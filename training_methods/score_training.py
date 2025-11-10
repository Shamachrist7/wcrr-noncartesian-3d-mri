import torch
import wandb
from tqdm import tqdm
from deepinv.loss.metric import PSNR
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

def score_training(
    regularizer,
    train_dataloader,
    val_dataloader,
    physics,
    sigma_min,
    sigma_max,
    sigma_val,
    wandb_setup, # Dictionary
    epochs=100,
    lr=0.005,
    lr_decay=0.99,
    device="cuda" if torch.cuda.is_available() else "cpu",
    validation_epochs=20,
    logger=None,
    dynamic_range_psnr=False,
    savestr=None,
    loss_fn=lambda x,y: torch.sum((x-y)**2),
    adabelief=False,
):
    wandb.init(
        # Set the project where this run will be logged
        project=wandb_setup["project"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"Score {wandb_setup["regularizer_name"]}",
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
    
    if adabelief:
        optimizer = AdaBelief(
            [
                {"params": regularizer.parameters(), "lr": lr},
             ],
            lr=lr,
            betas=(0.5, 0.9),
        )
    else:
        optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    # number of knots
    K = regularizer.regularizer.scaling.K

    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []

    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    for epoch in range(epochs):
        
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0

        for x in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            x = x.to(device).to(torch.float32)
            #sigma = noise_generator.step(x.shape[0])["sigma"] # drawing uniformly a random noise level for each image of in the batch separetly
            
            knots= torch.linspace(sigma_min, sigma_max, K)
            perm = torch.randperm(K)       # a random permutation of 0..K-1
            idx = perm[:x.shape[0]]         # first x.shape[0] distinct random indices
            sigma = knots[idx].to(device)                   # the randomly picked sub-tensor (unordered)
            
            y = physics.noise_model(physics.A(x), sigma=sigma)
            xhat = y - regularizer.grad(y, sigma)
            loss = loss_fn(xhat, x)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss_epoch += loss.item()
            train_psnr_epoch += psnr(xhat, x).mean().item()

        scheduler.step()
        mean_train_loss = train_loss_epoch / len(train_dataloader)
        mean_train_psnr = train_psnr_epoch / len(train_dataloader)
        loss_train.append(mean_train_loss)
        psnr_train.append(mean_train_psnr)

        print_str = f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}"
        print(print_str)
        if logger is not None:
            logger.info(print_str)
        wandb.log({"Epoch": epoch+1, "Gradient norm":grad_norm(regularizer), "Train Loss": mean_train_loss, "Train PSNR": mean_train_psnr})


        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_psnr_epoch = 0
                for x_val in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val"
                ):
                    x_val = x_val.to(device).to(torch.float32)
                    y = physics.noise_model(physics.A(x_val), sigma=sigma_val)
                    xhat = y - regularizer.grad(y, sigma_val)
                    loss = loss_fn(xhat,x_val)

                    val_loss_epoch += loss.item()
                    val_psnr_epoch += psnr(xhat, x_val).mean().item()

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

                # save checkpoint whenever you validate
                torch.save(regularizer.state_dict(),f"weights/score_for_Denoising/{wandb_setup["regularizer_name"]}_score_training_ckpt_{epoch + 1}.pt")

                """# ---- Save best regularizer based on validation PSNR ----
                if mean_val_psnr > best_val_psnr:
                    best_val_psnr = mean_val_psnr
                    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)"""
    wandb.finish()

    return regularizer, loss_train, loss_val, psnr_train, psnr_val
