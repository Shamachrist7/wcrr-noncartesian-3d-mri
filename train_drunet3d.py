import os, time, argparse
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import numpy as np

from baselines.PostProcessing.data import build_loader
from baselines.drunet.drunet_base import DRUNet
from utils import ri_to_complex

def compute_mask(gt, threshold=0.05):
    mag = np.abs(gt)
    mask = mag > threshold * mag.max()
    return mask

def masked_psnr(gt, pred, mask):
    mse = np.mean((gt[mask]-pred[mask])**2)
    data_range = gt.max()
    return 20*np.log10(data_range/np.sqrt(mse))

def psnr(pred, gt):
    mask = compute_mask(ri_to_complex(gt).abs().detach().cpu().numpy())
    return masked_psnr(ri_to_complex(gt).abs().detach().cpu().numpy(), ri_to_complex(pred).abs().detach().cpu().numpy(), mask)

# def psnr(pred, gt):
#     mse = F.mse_loss(pred, gt).clamp_min(1e-12)
#     vmax = gt.detach().amax().clamp_min(1e-8)
#     return 20 * torch.log10(vmax / torch.sqrt(mse))


def grad_stats(model):
    total_sq, max_abs, mean_sum, n = 0.0, 0.0, 0.0, 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_sq += g.norm(2).item() ** 2
        max_abs = max(max_abs, g.abs().max().item())
        mean_sum += g.abs().sum().item()
        n += g.numel()
    return {
        "norm": total_sq ** 0.5,
        "max_abs": max_abs,
        "mean_abs": mean_sum / max(n, 1),
    }


@torch.no_grad()
def make_image(net, noise_level, fixed_zf, fixed_gt, z=None):
    net.eval()
    pred = net(fixed_zf, noise_level)

    if z is None:
        z = fixed_gt.shape[-3] // 2

    imgs = [
        ri_to_complex(fixed_zf).abs()[z],
        ri_to_complex(pred).abs()[z],
        ri_to_complex(fixed_gt).abs()[z],
        (ri_to_complex(pred).abs() - ri_to_complex(fixed_gt).abs())[z],
    ]
    titles = ["ZF", "DRUNet3D", "GT", "Abs error"]

    vmax = torch.quantile(ri_to_complex(fixed_gt).abs()[z].flatten(), 0.995).item()
    err_vmax = 0.2 * vmax

    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    for i in range(4):
        ax[i].imshow(
            imgs[i].detach().cpu(),
            cmap="gray",
            vmin=0,
            vmax=err_vmax if i == 3 else vmax,
        )
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    plt.tight_layout()
    return wandb.Image(fig)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ds, loader = build_loader(batch_size=args.batch_size, shuffle=True, cache=args.cache)

    # model and reconstructor (use deepinv's DRUNet 3D)
    net = DRUNet(in_channels=2, out_channels=2, pretrained=None, dim=3)
    net.load_state_dict(torch.load("weights/drunet/drunet_3d_complex_denoise.pth", map_location=device, weights_only=True))
    net.to(device)
    
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    )

    wandb.init(project=args.project, name=args.name, config=vars(args))
    wandb.watch(net, log=None)

    fixed_zf, fixed_gt = next(iter(loader))
    fixed_zf = fixed_zf.to(device)
    fixed_gt = fixed_gt.to(device)

    global_step = 0
    optim_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()

        epoch_loss_sum = 0.0
        epoch_n = 0

        opt.zero_grad(set_to_none=True)
        acc = {}
        acc_n = 0
        accum_target = args.accum_steps

        for it, (zf, gt) in enumerate(loader, start=1):
            global_step += 1

            if (it - 1) % args.accum_steps == 0:
                accum_target = min(args.accum_steps, len(loader) - it + 1)

            zf = zf.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            pred = net(zf, args.noise_level)
            loss = F.mse_loss(ri_to_complex(pred).abs(), ri_to_complex(gt).abs())

            (loss / accum_target).backward()

            with torch.no_grad():
                scalars = {
                    "train/loss_mse": loss.item(),
                    "train/psnr": float(psnr(pred.detach(), gt)),
                    "train/zf_min": zf.min().item(),
                    "train/zf_max": zf.max().item(),
                    "train/gt_min": gt.min().item(),
                    "train/gt_max": gt.max().item(),
                    "train/pred_min": pred.min().item(),
                    "train/pred_max": pred.max().item(),
                }

            for k, v in scalars.items():
                acc[k] = acc.get(k, 0.0) + v

            acc_n += 1
            epoch_loss_sum += loss.item()
            epoch_n += 1

            do_step = (it % args.accum_steps == 0) or (it == len(loader))

            if not do_step:
                continue

            g_before = grad_stats(net)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

            g_after = grad_stats(net)

            opt.step()
            opt.zero_grad(set_to_none=True)
            optim_step += 1

            log = {k: v / acc_n for k, v in acc.items()}
            log.update({
                "train/lr": opt.param_groups[0]["lr"],
                "train/global_step": global_step,
                "train/optim_step": optim_step,
                "train/accum_count": acc_n,

                "grad/before_norm": g_before["norm"],
                "grad/before_max_abs": g_before["max_abs"],
                "grad/before_mean_abs": g_before["mean_abs"],
                "grad/after_norm": g_after["norm"],
                "grad/after_max_abs": g_after["max_abs"],
                "grad/after_mean_abs": g_after["mean_abs"],
            })

            if torch.cuda.is_available():
                log["gpu/memory_allocated_GB"] = torch.cuda.memory_allocated() / 1e9
                log["gpu/memory_reserved_GB"] = torch.cuda.memory_reserved() / 1e9

            if optim_step % args.img_every == 0:
                log["fixed_slice"] = make_image(net, args.noise_level, fixed_zf, fixed_gt)

            wandb.log(log, step=optim_step)

            acc = {}
            acc_n = 0

        scheduler.step()

        epoch_loss = epoch_loss_sum / max(epoch_n, 1)
        epoch_time = time.time() - t0

        wandb.log(
            {
                "epoch/loss": epoch_loss,
                "epoch/time_sec": epoch_time,
                "epoch/lr_after_step": opt.param_groups[0]["lr"],
                "epoch": epoch,
                "epoch/optim_steps": optim_step,
            },
            step=optim_step,
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "optim_step": optim_step,
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "loss": epoch_loss,
        }

        torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))

        if epoch % 500 == 0:
            torch.save(ckpt, os.path.join(args.out_dir, f"ckp_epoch_{epoch}.pt"))

        print(
            f"epoch {epoch:03d} | loss {epoch_loss:.4e} | "
            f"lr {opt.param_groups[0]['lr']:.2e} | "
            f"optim_steps {optim_step} | time {epoch_time:.1f}s"
        )

    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--project", type=str, default="drunet3d_postprocessing")
    p.add_argument("--name", type=str, default="zf_to_gt")
    p.add_argument("--out_dir", type=str, default="weights/PostProcess")

    p.add_argument("--epochs", type=int, default=2000)#400
    p.add_argument("--batch_size", type=int, default=1)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_step", type=int, default=50) #epochs/8
    p.add_argument("--lr_gamma", type=float, default=1.0) #0.5

    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--noise_level", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accum_steps", type=int, default=4)

    p.add_argument("--img_every", type=int, default=40)
    p.add_argument("--cache", action="store_true")

    args = p.parse_args()
    train(args)