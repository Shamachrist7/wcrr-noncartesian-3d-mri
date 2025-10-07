import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from deepinv.loss.metric import PSNR
from deepinv.optim.utils import minres
from evaluation.nmAPG3d import reconstruct_nmAPG
import copy
from .utils.adabelief import AdaBelief

def grad_norm(model, norm_type=2):
    total = 0.0
    for p in model.parameters():
        if p.grad is None: 
            continue
        param_norm = p.grad.data.norm(norm_type)
        total += float(param_norm) ** norm_type
    return total ** (1.0 / norm_type)

def bilevel_training(
    regularizer,
#    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    physics,
    noise_generator,
    sigma_val,
    wandb_setup,
    fitting_only=False,
    epochs=100,
    mode="IFT",
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    minres_max_iter=5000,
    minres_tol=1e-5,
    jfb_step_size_factor=1.0,
    lr=0.005,
    lr_decay=0.99,
    momentum_optim=None,
    reg=False,
    reg_para=1e-5,
    reg_reduced=False,
    adabelief=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    validation_epochs=5,
    logger=None,
    dynamic_range_psnr=False,
    savestr=None,
    upper_loss=lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1),
):
    assert validation_epochs <= epochs, (
        "validation_epochs cannot be greater than epochs. "
        "If validation_epochs > epochs, no validation will occur, "
        "best_regularizer_state will remain unchanged, and the returned model will be identical to the initial state."
    )
    task = "Fitting" if fitting_only else "Training"
    wandb.init(
        # Set the project where this run will be logged
        project=wandb_setup["project"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{task} {wandb_setup["regularizer_name"]}",
        # Track hyperparameters and run metadata
        config={
        "lr": lr,
        "exponential lr decay factor": lr_decay,
        "architecture": f"Multi-noise level {wandb_setup["regularizer_name"]}",
        "dataset": "Calgary-Campinas",
        "epochs": epochs,
        })
    # To track gradients
    #wandb.watch(regularizer)

    def hessian_vector_product(
        sigma,
        x,
        v,
        data_fidelity,
        y,
        regularizer,
        lmbd,
        physics,
        diff=False,
        only_reg=False,
    ):
        x = x.requires_grad_(True)
        if only_reg:
            grad = lmbd * regularizer.grad(x, sigma)
        else:
            grad = data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x, sigma)
        dot = torch.dot(grad.view(-1), v.view(-1))
        hvp = torch.autograd.grad(dot, x, create_graph=diff)[0]
        if diff:
            return hvp
        return hvp.detach()

    def jac_vector_product(sigma, x, v, data_fidelity, y, regularizer, lmbd, physics):
        grad_lower_level = lambda x: data_fidelity.grad(
            x, y, physics
        ) + lmbd * regularizer.grad(x, sigma)
        for param in regularizer.parameters():
            if param.requires_grad:
                dot = torch.dot(grad_lower_level(x).view(-1), v.view(-1))
                if param.grad is None:
                    param.grad = -torch.autograd.grad(dot, param, create_graph=False)[
                        0
                    ].detach()
                else:
                    param.grad -= torch.autograd.grad(dot, param, create_graph=False)[
                        0
                    ].detach()
        return regularizer

    def jac_pow_loss(x, physics, M=50, tol=1e-2):
        hvp = torch.randint(low=0, high=1, size=x.shape).to(x) * 2 - 1
        hvp_old = hvp.clone()
        for i in range(M):
            hvp = hessian_vector_product(
                sigma,
                x,
                hvp,
                data_fidelity,
                y,
                regularizer,
                lmbd,
                physics,
                diff=False,
                only_reg=True,
            ).detach()
            hvp = torch.nn.functional.normalize(hvp, dim=[-2, -1], out=hvp)
            if torch.norm(hvp - hvp_old) / x.size(0) < tol:
                break
            hvp_old = hvp.clone()
        hvp = hvp.clone(memory_format=torch.contiguous_format).detach()
        hvp = hessian_vector_product(
            sigma,
            x,
            hvp,
            data_fidelity,
            y,
            regularizer,
            lmbd,
            physics,
            diff=True,
            only_reg=True,
        )
        norm_sq = torch.sum(hvp ** 2) / x.size(0)
        print(f"Jac_Loss: {norm_sq}")
        return torch.clip(norm_sq, min=200, max=None)

    if adabelief:
        momentum_optim = (0.5, 0.9) if momentum_optim is None else momentum_optim
        optimizer = AdaBelief(
            [
                {"params": regularizer.parameters(), "lr": lr},
            ],
            lr=lr,
            betas=(0.5, 0.9),
        )
    else:
        momentum_optim = (0.9, 0.999) if momentum_optim is None else momentum_optim
        optimizer = torch.optim.Adam(
            regularizer.parameters(), lr=lr, betas=momentum_optim
        )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []
    filter_snapshots = []  # To store filter evolution

    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    for epoch in range(epochs):
        # ---- Training ----
        
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0
        train_step = 0
        for x in (
            progress_bar := tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs} - Train",
                total=len(train_dataloader),
            )
        ):
            train_step += 1
            x = x.to(device).to(torch.float32)
            #sigma = noise_generator.step(x.shape[0])["sigma"] # drawing uniformly a random noise level for each image of in the batch separetly
            
            sigma_min, sigma_max, K = 0.01, 0.1, 12
            knots= torch.linspace(sigma_min, sigma_max, K)
            perm = torch.randperm(K)       # a random permutation of 0..K-1
            idx = perm[:x.shape[0]]         # first x.shape[0] distinct random indices
            sigma = knots[idx].to(device)                   # the randomly picked sub-tensor (unordered)
            
            y = physics.noise_model(physics.A(x), sigma=sigma)
            x_noisy = physics.A_dagger(y)

            x_recon, x_stats = reconstruct_nmAPG(
                sigma,
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol_train,
                verbose=verbose,
                x_init=x_noisy,
                return_stats=True,
            )

            optimizer.zero_grad()
            loss_fn = lambda x_in: upper_loss(x, x_in).mean()
            train_loss_epoch += loss_fn(x_recon).item()
            train_psnr_epoch += psnr(x_recon, x).mean().item()
            progress_bar.set_description(
                "used {0} of {1} steps, Loss: {2:.2E}, PSNR: {3:.2f}".format(
                    x_stats["steps"] + 1,
                    NAG_max_iter,
                    train_loss_epoch / train_step,
                    train_psnr_epoch / train_step,
                )
            )
            if x_stats["steps"] + 1 == NAG_max_iter:
                print("maxiter hit...")
                if logger is not None:
                    logger.info(f"maxiter hit in iteration {train_step}")

            x_recon = x_recon.detach()

            if reg and (train_step % 5) == 1:
                jac_loss = reg_para * jac_pow_loss(x_recon, physics)
                jac_loss.backward()

            if mode == "IFT":
                x_recon = x_recon.requires_grad_(True)
                grad_loss = torch.autograd.grad(
                    loss_fn(x_recon), x_recon, create_graph=False
                )[0].detach()

                q = minres(
                    lambda input: hessian_vector_product(
                        sigma,
                        x_recon.detach(),
                        input,
                        data_fidelity,
                        y,
                        regularizer,
                        lmbd,
                        physics,
                    ),
                    grad_loss,
                    max_iter=minres_max_iter,
                    tol=minres_tol,
                )

                regularizer = jac_vector_product(
                    sigma, x_recon, q, data_fidelity, y, regularizer, lmbd, physics
                )
            elif mode == "JFB":
                L = x_stats["L"]
                grad = data_fidelity.grad(
                    x_recon, y, physics
                ) + lmbd * regularizer.grad(x_recon, sigma)
                x_recon = x_recon - jfb_step_size_factor / L * grad
                loss = upper_loss(x_recon, x).mean()
                loss.backward()
            else:
                raise NameError("unknwon mode!")
            optimizer.step()
            if logger is not None and train_step % 10 == 0:
                logger.info(
                    f"Step {train_step}, Train PSNR {train_psnr_epoch/train_step}"
                )

        scheduler.step()
        mean_train_loss = train_loss_epoch / len(train_dataloader)
        mean_train_psnr = train_psnr_epoch / len(train_dataloader)
        loss_train.append(mean_train_loss)
        psnr_train.append(mean_train_psnr)
        
        print_str = f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}"
        print(print_str)
        if logger is not None:
            logger.info(print_str)
        wandb.log({"Epoch": epoch+1, "Gradient norm": grad_norm(regularizer), "Train Loss": mean_train_loss, "Train PSNR": mean_train_psnr})

        # Save checkpoints every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(regularizer.state_dict(),f"weights/bilevel_Denoising/{wandb_setup["regularizer_name"]}_bilevel_IFT_ckpt_{epoch + 1}.pt")

        # ---- Validation ----
        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_psnr_epoch = 0
                for x_val in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val"
                ):
                    x_val = x_val.to(device).to(torch.float32)
                    y_val = physics.noise_model(physics.A(x_val), sigma=sigma_val)

                    x_recon_val = reconstruct_nmAPG(
                        sigma_val,
                        y_val,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd,
                        NAG_step_size,
                        NAG_max_iter,
                        NAG_tol_val,
                        verbose=verbose,
                        x_init=y_val,
                    )

                    val_loss_epoch += upper_loss(x_val, x_recon_val).mean().item()
                    val_psnr_epoch += psnr(x_recon_val, x_val).mean().item()

                mean_val_loss = val_loss_epoch / len(val_dataloader)
                mean_val_psnr = val_psnr_epoch / len(val_dataloader)
                loss_val.append(mean_val_loss)
                psnr_val.append(mean_val_psnr)

                print_str = f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.2E}, PSNR: {mean_val_psnr:.2f}"
                print(print_str)

                if savestr is not None:
                    torch.save(
                        regularizer.state_dict(),
                        savestr + "_epoch_" + str(epoch) + ".pt",
                    )

                if logger is not None:
                    logger.info(print_str)
                wandb.log({"Val Loss": mean_val_loss, "Val PSNR": mean_val_psnr})

                # ---- Save best regularizer based on validation PSNR ----
                if mean_val_psnr > best_val_psnr:
                    best_val_psnr = mean_val_psnr
                    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)
    wandb.finish()

    return regularizer, loss_train, loss_val, psnr_train, psnr_val
