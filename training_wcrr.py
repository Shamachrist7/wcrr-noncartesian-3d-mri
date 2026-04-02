import torch
import deepinv as dinv
from deepinv.optim import L2
from training_methods import bilevel_training, score_training
from reg_architecture import WCRR3D
from data_processing import load_data
import argparse
import os

parser = argparse.ArgumentParser(description="Choosing the training setting")
parser.add_argument("--root", type=str, default='../../../../../../../../LOCAL/mri_data')
parser.add_argument("--regularizer_name", type=str, default="WCRR_500")
inp = parser.parse_args()
root = inp.root

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.random.manual_seed(0)  # make results deterministic

problem = "Denoising"
hypergradient_computation = "IFT"  # IFT or JFB
regularizer_name = inp.regularizer_name  # WCRR, WCRR_no_rotations
load_pretrain = False  # load pretrained weights given that they exist
load_parameter_fitting = (
    False  # load pretrained weights and learned regularization and scaling parameter
)
wandb_setup = {"project": regularizer_name + "_3D_Denoiser", "regularizer_name": regularizer_name}

sigma_min = 0.01
sigma_max = 0.1
sigma_val = 0.05

if regularizer_name == "WCRR_no_rotations": #non-rotation-invariant version
    pretrain_epochs = 1000
    pretrain_lr = 1e-2
    fitting_lr = 0.1
    adabelief = True
    epochs = 500
    lr = 1e-2
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-6
    regularizer = WCRR3D(
        weak_convexity=1.0,
        nb_channels=[2, 4, 8, 32],
        filter_sizes=[3, 3, 3],
        rotations=False,
    ).to(device)
elif regularizer_name == "WCRR_500":#"WCRR"
    pretrain_epochs = 1000
    pretrain_lr = 1e-2
    fitting_lr = 0.1
    adabelief = True
    epochs = 500
    lr = 1e-2
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-6
    regularizer = WCRR3D(
        weak_convexity=1.0,
        nb_channels=[2, 4, 8, 32],
        filter_sizes=[3, 3, 3],
        rotations=True,
    ).to(device)

lmbd = 1.0

if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir(f"weights/score_for_{problem}"):
    os.mkdir(f"weights/score_for_{problem}")
if not os.path.isdir(f"weights/score_parameter_fitting_for_{problem}"):
    os.mkdir(f"weights/score_parameter_fitting_for_{problem}")
if not os.path.isdir(f"weights/bilevel_{problem}"):
    os.mkdir(f"weights/bilevel_{problem}")

params = 0
for p in regularizer.parameters():
    params += p.numel()
print(params)

patch_size = 64
batch_size = 12
data_fidelity = L2(sigma=1.0)

if problem == "Denoising":
    train_dataloader = load_data(root, patch_size, batch_size, device=device, train=True, scale_fact=1e-6)
val_dataloader = load_data(root, patch_size, batch_size, device=device, train=False, scale_fact=1e-6)
pretrain_dataloader = train_dataloader
# Physics
physics = dinv.physics.DecomposablePhysics(device=device, noise_model=dinv.physics.noise.GaussianNoise(sigma=sigma_min))
sigma_val = torch.tensor([sigma_val], device=device).tile(batch_size)


if load_pretrain and not load_parameter_fitting:
    regularizer.load_state_dict(
        torch.load(
            f"weights/score_for_{problem}/{regularizer_name}_score_training_ckpt_1000.pt",
            weights_only=True
        )
    )
elif not load_parameter_fitting:
    for p in regularizer.parameters():
        p.requires_grad_(True)
    (
        regularizer,
        loss_train,
        loss_val,
        psnr_train,
        psnr_val,
    ) = score_training(
        regularizer,
        pretrain_dataloader,
        val_dataloader,
        wandb_setup=wandb_setup,
        physics=physics,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_val=sigma_val,
        epochs=pretrain_epochs,
        lr=pretrain_lr,
        lr_decay=0.1 ** (1 / pretrain_epochs),
        device=device,
        validation_epochs=100,
        adabelief=adabelief,
    )
    torch.save(
        regularizer.state_dict(),
        f"weights/score_for_{problem}/{regularizer_name}_score_training_for_{problem}.pt",
    )


if load_parameter_fitting:
    regularizer.load_state_dict(
        torch.load(
            f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_IFT_ckpt_20.pt",
            weights_only=True
        )
    )
else:
    for p in regularizer.parameters():
        p.requires_grad_(False)
    
    regularizer.beta.requires_grad_(True)
    regularizer.scaling.s_at_knots.requires_grad_(True)
    
    regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
        regularizer,
        data_fidelity,
        lmbd,
        train_dataloader,
        val_dataloader,
        wandb_setup=wandb_setup,
        fitting_only=True,
        physics=physics,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_val=sigma_val,
        epochs=20,
        mode=hypergradient_computation,
        NAG_step_size=1e-1,
        NAG_max_iter=1000,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-4,
        lr=fitting_lr,
        lr_decay=0.95,
        device=device,
        verbose=False,
        validation_epochs=5,
    )
    torch.save(
        regularizer.state_dict(),
        f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_{hypergradient_computation}_for_{problem}.pt",
    )

# bilevel training

for p in regularizer.parameters():
    p.requires_grad_(True)

if not jacobian_regularization:
    jacobian_regularization_parameter = 0.0

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    wandb_setup=wandb_setup,
    physics=physics,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    sigma_val=sigma_val,
    epochs=epochs,
    mode=hypergradient_computation,
    NAG_step_size=1e-1,
    NAG_max_iter=1500,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=lr,
    lr_decay=0.05 ** (1 / epochs),
    reg=jacobian_regularization,
    reg_para=jacobian_regularization_parameter,
    device=device,
    verbose=False,
    validation_epochs=100,
    adabelief=adabelief,
)

torch.save(
    regularizer.state_dict(),
    f"weights/bilevel_{problem}/{regularizer_name}_bilevel_{hypergradient_computation}_for_{problem}.pt",
)
