import math, time, yaml, argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

try:
    import wandb
except ImportError:
    wandb = None

from data import build_loader
from ncsnpp2D import build_model_from_config


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(sd[k].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)


def cycle(loader):
    while True:
        for x in loader:
            yield x


def sample_sigma(B, sigma_min, sigma_max, device):
    u = torch.rand(B, device=device)
    return sigma_min * (sigma_max / sigma_min) ** u


def ve_dsm_loss_from_noise(model, xt, sigma, eps):
    B = xt.shape[0]
    score = model(xt, sigma)
    loss = ((score * sigma[:, None, None, None] + eps) ** 2)
    loss = loss.reshape(B, -1).mean(dim=1).mean()
    return loss, score


def ve_dsm_loss(model, x0, sigma_min, sigma_max):
    B = x0.shape[0]
    sigma = sample_sigma(B, sigma_min, sigma_max, x0.device)
    eps = torch.randn_like(x0)
    xt = x0 + sigma[:, None, None, None] * eps
    loss, score = ve_dsm_loss_from_noise(model, xt, sigma, eps)
    return loss, sigma, xt, score, eps


@torch.no_grad()
def ema_loss(model, ema, xt, sigma, eps):
    ema_model = deepcopy(model).to(xt.device)
    ema.copy_to(ema_model)
    ema_model.eval()
    loss, score = ve_dsm_loss_from_noise(ema_model, xt, sigma, eps)
    del ema_model
    return loss, score


@torch.no_grad()
def tweedie(model, xt, sigma):
    B = xt.shape[0]
    s = torch.full((B,), float(sigma), device=xt.device)
    return xt + (float(sigma) ** 2) * model(xt, s)


def mag(x):
    return torch.sqrt(x[:, 0:1].pow(2) + x[:, 1:2].pow(2) + 1e-12)


def img_grid(x, nrow=4):
    return vutils.make_grid(
        x.detach().float().cpu(),
        nrow=nrow,
        normalize=True,
        scale_each=True,
    )


@torch.no_grad()
def log_denoising(model, fixed, cfg, step):
    if wandb is None or not cfg["wandb"]["enable"]:
        return

    model.eval()
    B = fixed.shape[0]
    rows = [mag(fixed)]
    logs = {}

    for sig in cfg["visualization"]["fixed_noise_levels"]:
        sig = float(sig)
        noisy = fixed + sig * torch.randn_like(fixed)
        den = tweedie(model, noisy, sig)

        rows += [mag(noisy), mag(den), (mag(den) - mag(fixed)).abs()]
        logs[f"fixed/mse_noisy_sigma_{sig}"] = F.mse_loss(noisy, fixed).item()
        logs[f"fixed/mse_denoised_sigma_{sig}"] = F.mse_loss(den, fixed).item()

    logs["fixed/magnitude_grid"] = wandb.Image(
        img_grid(torch.cat(rows, 0), nrow=B),
        caption="GT, then noisy/denoised/error per sigma",
    )

    sig = float(cfg["visualization"]["fixed_noise_levels"][min(2, len(cfg["visualization"]["fixed_noise_levels"]) - 1)])
    noisy = fixed + sig * torch.randn_like(fixed)
    den = tweedie(model, noisy, sig)

    ri = torch.cat(
        [
            fixed[:, 0:1], fixed[:, 1:2],
            noisy[:, 0:1], noisy[:, 1:2],
            den[:, 0:1], den[:, 1:2],
        ],
        0,
    )

    logs["fixed/real_imag_grid"] = wandb.Image(
        img_grid(ri, nrow=B),
        caption=f"GT R/I, noisy R/I, denoised R/I, sigma={sig}",
    )

    wandb.log(logs, step=step)
    model.train()


def grad_stats(model, max_hist_elems=200000):
    grads = []
    total_sq = 0.0
    max_abs = 0.0
    mean_abs_sum = 0.0
    count = 0

    for p in model.parameters():
        if p.grad is None:
            continue

        g = p.grad.detach().float().flatten()

        total_sq += g.pow(2).sum().item()
        max_abs = max(max_abs, g.abs().max().item())
        mean_abs_sum += g.abs().sum().item()
        count += g.numel()

        grads.append(g.cpu())

    norm = math.sqrt(total_sq)
    mean_abs = mean_abs_sum / max(count, 1)

    if len(grads) == 0:
        return norm, max_abs, mean_abs, None

    flat = torch.cat(grads)

    if flat.numel() > max_hist_elems:
        idx = torch.randperm(flat.numel())[:max_hist_elems]
        flat = flat[idx]

    return norm, max_abs, mean_abs, flat.numpy()


def set_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


def linear_warmup_lr(
    step,
    base_lr,
    total_steps,
    warmup_frac=0.025,
    start_factor=0.1,
):
    warmup_steps = max(1, int(warmup_frac * total_steps))

    if step >= warmup_steps:
        return base_lr, 1.0, warmup_steps

    if warmup_steps == 1:
        factor = 1.0
    else:
        factor = start_factor + (1.0 - start_factor) * (step - 1) / (warmup_steps - 1)

    return base_lr * factor, factor, warmup_steps


def save(path, model, ema, opt, step, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "opt": opt.state_dict(),
            "step": step,
            "cfg": cfg,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ncsnpp2D.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    seed = int(cfg["project"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workdir = Path(cfg["project"]["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)

    _, loader = build_loader(
        batch_size=cfg["data"]["batch_size"],
        crop_size=cfg["data"]["crop_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=seed,
    )

    it = cycle(loader)

    model = build_model_from_config(cfg).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}; Length data loader: {len(loader)}.")

    ema = EMA(model, cfg["training"]["ema_decay"])

    base_lr = float(cfg["optim"]["lr"])
    total_steps = int(cfg["training"]["steps"])

    warmup_frac = 0.025
    warmup_start_factor = 0.1
    warmup_steps = max(1, int(warmup_frac * total_steps))

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr * warmup_start_factor,
        betas=(
            float(cfg["optim"]["beta1"]),
            float(cfg["optim"]["beta2"]),
        ),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    print(
        f"LR warmup: {base_lr * warmup_start_factor:.3e} -> {base_lr:.3e} "
        f"over {warmup_steps} steps."
    )

    if wandb is not None and cfg["wandb"]["enable"]:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity", None),
            name=cfg["wandb"].get("name", None),
            tags=cfg["wandb"].get("tags", None),
            config=cfg,
        )

        wandb.watch(
            model,
            log="gradients",
            log_freq=max(1000, cfg["training"]["log_every"]),
        )

    fixed = next(iter(loader))[: cfg["visualization"]["num_fixed_images"]]
    fixed = fixed.to(device).contiguous()

    sigma_min = float(cfg["sde"]["sigma_min"])
    sigma_max = float(cfg["sde"]["sigma_max"])
    grad_clip = float(cfg["optim"]["grad_clip"])

    t0 = time.time()

    for step in range(1, total_steps + 1):
        lr, lr_factor, warmup_steps = linear_warmup_lr(
            step=step,
            base_lr=base_lr,
            total_steps=total_steps,
            warmup_frac=warmup_frac,
            start_factor=warmup_start_factor,
        )

        set_lr(opt, lr)

        x0 = next(it).to(device, non_blocking=True).contiguous()

        opt.zero_grad(set_to_none=True)

        loss, sigma, xt, score, eps = ve_dsm_loss(
            model,
            x0,
            sigma_min,
            sigma_max,
        )

        loss.backward()

        gn_before, gmax_before, gmean_before, grad_hist_before = grad_stats(model)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        gn_after, gmax_after, gmean_after, grad_hist_after = grad_stats(model)

        opt.step()
        ema.update(model)

        if step % cfg["training"]["log_every"] == 0:
            e_loss, e_score = ema_loss(
                model,
                ema,
                xt.detach(),
                sigma.detach(),
                eps.detach(),
            )

            logs = {
                "train/loss": loss.item(),
                "train/ema_loss": e_loss.item(),
                "train/lr": opt.param_groups[0]["lr"],
                "train/lr_factor": lr_factor,
                "train/warmup_steps": warmup_steps,

                "grad/norm_before_clip": gn_before,
                "grad/norm_after_clip": gn_after,
                "grad/max_abs_before_clip": gmax_before,
                "grad/max_abs_after_clip": gmax_after,
                "grad/mean_abs_before_clip": gmean_before,
                "grad/mean_abs_after_clip": gmean_after,

                "train/sigma_mean": sigma.mean().item(),
                "train/sigma_min_batch": sigma.min().item(),
                "train/sigma_max_batch": sigma.max().item(),

                "data/x0_min": x0.min().item(),
                "data/x0_max": x0.max().item(),
                "data/x0_abs_mean": x0.abs().mean().item(),
                "data/xt_abs_mean": xt.abs().mean().item(),

                "model/score_abs_mean": score.abs().mean().item(),
                "model/ema_score_abs_mean": e_score.abs().mean().item(),

                "time/sec_per_step": (time.time() - t0) / cfg["training"]["log_every"],
            }

            if wandb is not None and cfg["wandb"]["enable"]:
                if grad_hist_after is not None:
                    logs["grad/hist_after_clip"] = wandb.Histogram(grad_hist_after)

                wandb.log(logs, step=step)

            t0 = time.time()

            print(
                f"step {step:07d} | "
                f"loss {logs['train/loss']:.3e} | "
                f"ema_loss {logs['train/ema_loss']:.3e} | "
                f"lr {logs['train/lr']:.3e} | "
                f"sigma {logs['train/sigma_mean']:.3e} | "
                f"gn {gn_after:.3e}"
            )

        if step % cfg["training"]["image_every"] == 0:
            ema_model = deepcopy(model).to(device)
            ema.copy_to(ema_model)

            log_denoising(
                ema_model,
                fixed,
                cfg,
                step,
            )

            del ema_model

        if step % cfg["training"]["ckpt_every"] == 0:
            save(
                workdir / f"ckpt_{step:07d}.pt",
                model,
                ema,
                opt,
                step,
                cfg,
            )

            save(
                workdir / "latest.pt",
                model,
                ema,
                opt,
                step,
                cfg,
            )

    save(
        workdir / "final.pt",
        model,
        ema,
        opt,
        total_steps,
        cfg,
    )

    if wandb is not None and cfg["wandb"]["enable"]:
        wandb.finish()


if __name__ == "__main__":
    main()