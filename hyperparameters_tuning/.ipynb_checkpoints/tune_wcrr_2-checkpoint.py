import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import wandb
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri
from reg_architectures import WCRR3D, ParameterLearningWrapper
from evaluation.nmAPG3d_evaluation import reconstruct_nmAPG
from deepinv.optim import L2
from deepinv.loss.metric import PSNR
from deepinv.optim.utils import minres
import warnings
import argparse
warnings.filterwarnings("ignore")

torch.random.manual_seed(0)  # make results deterministic

parser = argparse.ArgumentParser(description="Choosing the training setting")
parser.add_argument("--root", type=str, default='../../../../../../../../LOCAL/mri_data')
parser.add_argument("--regularizer_name", type=str, default="WCRR")
inp = parser.parse_args()
root = inp.root + "/Val/_images"
regularizer_name = inp.regularizer_name

data_fidelity = L2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "gpunufft"
scaler = 1e-6 
coils = 12 # number of coils in each volume
noise_level = 2e-3

wandb.init(project="tuning", name=f"tune_{regularizer_name.lower()}")

# Load trajectory and get the k-space locations
traj, traj_params = read_trajectory("trajectory.bin", dwell_time=0.01/2)
traj = traj.copy()
traj[traj < -0.5] = -0.5
traj[traj > 0.5] = 0.5
dim = traj_params["dimension"]
kspace_loc = traj.reshape(-1, dim)

# The 05 chosen volumes to tune hyperparameters on
volumes = ['e14120s11_P66048.7.h5.npy',
           'e14692s5_P14848.7.h5.npy',
           'e14110s3_P59904.7.h5.npy',
           'e15652s14_P51712.7.h5.npy',
           'e14531s6_P68096.7.h5.npy']
           
# Load the regularizer weights
if regularizer_name=="WCRR":
    regularizer = WCRR3D(weak_convexity=1.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=True).to(device)
    pretrained = "weights/bilevel_Denoising/WCRR_bilevel_IFT_ckpt_100.pt"
elif regularizer_name=="CRR":
    regularizer = WCRR3D(weak_convexity=0.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=True).to(device)
    pretrained = "weights/bilevel_Denoising/CRR_bilevel_IFT_ckpt_100.pt"
else:
    raise ValueError("Wrong regularizer name!")
regularizer.load_state_dict(torch.load(pretrained, weights_only=True, map_location=device))
regularizer.eval()

# Parameter learning wrapper
regularizer = ParameterLearningWrapper(regularizer, device=device).to(device)

# Parameters of the nmAPG solver
step_size = 1e-1
max_iter = 100
tol = 1e-3  # tolerance for the relative error (stopping criterion)

def grad_norm(model, norm_type=2):
    total = 0.0
    for p in model.parameters():
        if p.grad is None: 
            continue
        param_norm = p.grad.data.norm(norm_type)
        total += float(param_norm) ** norm_type
    return total ** (1.0 / norm_type)

def center_crop_3d(x, crop_size = 160):
    """
    x: tensor of shape [C, D, H, W]
    crop_size: int (e.g., 128)
    """
    C, D, H, W = x.shape
    
    d1 = (D - crop_size) // 2
    h1 = (H - crop_size) // 2
    w1 = (W - crop_size) // 2

    return x[:, d1:d1+crop_size, h1:h1+crop_size, w1:w1+crop_size]

# Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
x_0 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[0])),-1, 0)) #[coils,H,W,D] and complex dtype
x_0 = center_crop_3d(x_0)
x_1 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[1])),-1, 0))
x_1 = center_crop_3d(x_1)
x_2 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[2])),-1, 0))
x_2 = center_crop_3d(x_2)
x_3 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[3])),-1, 0))
x_3 = center_crop_3d(x_3)
x_4 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[4])),-1, 0))
x_4 = center_crop_3d(x_4)

# Forward NUFFT that takes coil images -> k-space & simulate the corruption
F_raw = get_operator(backend)(kspace_loc, x_0.shape[1:], n_coils=coils, density=True)
y_np_0 = F_raw.op(x_0) # simulates the undersampled kspace volume y. The zero-filled recon comes from it
y_np_0 = y_np_0 + noise_level * torch.randn_like(y_np_0)
y_np_1 = F_raw.op(x_1)
y_np_1 = y_np_1 + noise_level * torch.randn_like(y_np_1)
y_np_2 = F_raw.op(x_2)
y_np_2 = y_np_2 + noise_level * torch.randn_like(y_np_2)
y_np_3 = F_raw.op(x_3)
y_np_3 = y_np_3 + noise_level * torch.randn_like(y_np_3)
y_np_4 = F_raw.op(x_4)
y_np_4 = y_np_4 + noise_level * torch.randn_like(y_np_4)

# GRAPPA reconstruct the center of k-space and append the data, basis for our regularizers
new_kspace_loc_0, y_grappa_0 = do_grappa_and_append_data(kspace_loc, y_np_0, traj_params, af=(2, 2))
new_kspace_loc_1, y_grappa_1 = do_grappa_and_append_data(kspace_loc, y_np_1, traj_params, af=(2, 2))
new_kspace_loc_2, y_grappa_2 = do_grappa_and_append_data(kspace_loc, y_np_2, traj_params, af=(2, 2))
new_kspace_loc_3, y_grappa_3 = do_grappa_and_append_data(kspace_loc, y_np_3, traj_params, af=(2, 2))
new_kspace_loc_4, y_grappa_4 = do_grappa_and_append_data(kspace_loc, y_np_4, traj_params, af=(2, 2))

# Build reconstruction operator that ESTIMATES smaps from y_grappa (For each volume)
E_est_0 = get_operator(backend)(
    new_kspace_loc_0,
    x_0.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_0},
    density=True,
    use_gpu_direct=True,
)
physics_0 = MRINUFFTPhysicsRI(E_est_0)
y_0 = torch.from_numpy(y_grappa_0).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_0 = physics_0.A_adjoint(y_0)

E_est_1 = get_operator(backend)(
    new_kspace_loc_1,
    x_1.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_1},
    density=True,
    use_gpu_direct=True,
)
physics_1 = MRINUFFTPhysicsRI(E_est_1)
y_1 = torch.from_numpy(y_grappa_1).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_1 = physics_1.A_adjoint(y_1)

E_est_2 = get_operator(backend)(
    new_kspace_loc_2,
    x_2.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_2},
    density=True,
    use_gpu_direct=True,
)
physics_2 = MRINUFFTPhysicsRI(E_est_2)
y_2 = torch.from_numpy(y_grappa_2).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_2 = physics_2.A_adjoint(y_2)

E_est_3 = get_operator(backend)(
    new_kspace_loc_3,
    x_3.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_3},
    density=True,
    use_gpu_direct=True,
)
physics_3 = MRINUFFTPhysicsRI(E_est_3)
y_3 = torch.from_numpy(y_grappa_3).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_3 = physics_3.A_adjoint(y_3)

E_est_4 = get_operator(backend)(
    new_kspace_loc_4,
    x_4.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_4},
    density=True,
    use_gpu_direct=True,
)
physics_4 = MRINUFFTPhysicsRI(E_est_4)
y_4 = torch.from_numpy(y_grappa_4).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_4 = physics_4.A_adjoint(y_4)

# Reference/Ground Truth (Adjoint coil combination)
smaps_0 = torch.from_numpy(E_est_0.smaps)
x_gt_0 = torch.sum(torch.conj(smaps_0) * x_0, axis=0)
x_gt_ri_0 = complex_to_ri(x_gt_0).to(device)
#reference_0 = torch.abs(x_gt_0) # Magnitude

smaps_1 = torch.from_numpy(E_est_1.smaps)
x_gt_1 = torch.sum(torch.conj(smaps_1) * x_1, axis=0)
x_gt_ri_1 = complex_to_ri(x_gt_1).to(device)
#reference_1 = torch.abs(torch.from_numpy(x_gt_1)) # Magnitude

smaps_2 = torch.from_numpy(E_est_2.smaps)
x_gt_2 = torch.sum(torch.conj(smaps_2) * x_2, axis=0)
x_gt_ri_2 = complex_to_ri(x_gt_2).to(device)
#reference_2 = torch.abs(torch.from_numpy(x_gt_2)) # Magnitude

smaps_3 = torch.from_numpy(E_est_3.smaps)
x_gt_3 = torch.sum(torch.conj(smaps_3) * x_3, axis=0)
x_gt_ri_3 = complex_to_ri(x_gt_3).to(device)
#reference_3 = torch.abs(torch.from_numpy(x_gt_3)) # Magnitude

smaps_4 = torch.from_numpy(E_est_4.smaps)
x_gt_4 = torch.sum(torch.conj(smaps_4) * x_4, axis=0)
x_gt_ri_4 = complex_to_ri(x_gt_4).to(device)
#reference_4 = torch.abs(torch.from_numpy(x_gt_4)) # Magnitude
        
# Objective function for Optuna
lr = 1e-3 #1e-3 #1e-2
epochs = 100 #20
lmbd = 1.0
sigma_init = 0.1
sigma = torch.tensor([sigma_init], device=device)
upper_loss = lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1)
optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr, betas=(0.5, 0.9))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

def hessian_vector_product(sigma, x, v, data_fidelity, y, regularizer, lmbd, physics, diff=False, only_reg=False):
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
    grad_lower_level = lambda x: data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x, sigma)
    for param in regularizer.parameters():
        if param.requires_grad:
            dot = torch.dot(grad_lower_level(x).view(-1), v.view(-1))
            if param.grad is None:
                param.grad = -torch.autograd.grad(dot, param, create_graph=False)[0].detach()
            else:
                param.grad -= torch.autograd.grad(dot, param, create_graph=False)[0].detach()
    return regularizer


for epoch in range(epochs):
    loss_ = 0.0
    psnr_ = 0.0
    for y, physics, x_adj_ri, x_gt_ri in [(y_0, physics_0, x_adj_ri_0, x_gt_ri_0), (y_1, physics_1, x_adj_ri_1, x_gt_ri_1), (y_2, physics_2, x_adj_ri_2, x_gt_ri_2), (y_3, physics_3, x_adj_ri_3, x_gt_ri_3), (y_4, physics_4, x_adj_ri_4, x_gt_ri_4)]:
        print(f"lamda = {regularizer.lamda}, beta = {regularizer.beta}")
        x_rec_ri = reconstruct_nmAPG(
                    sigma,
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
                    lmbd,
                    step_size,
                    max_iter,
                    tol,
                    verbose=True,
                    x_init=x_adj_ri,
                    return_stats=False)
        optimizer.zero_grad()            
        loss_fn = lambda x_in: upper_loss(x_gt_ri, x_in).mean()
        loss_ += loss_fn(x_rec_ri).item()
        psnr_ += PSNR(max_pixel=None)(torch.abs(ri_to_complex(x_rec_ri)), torch.abs(ri_to_complex(x_gt_ri))).mean().item()
        x_rec_ri = x_rec_ri.detach()
        x_rec_ri = x_rec_ri.requires_grad_(True)
        grad_loss = torch.autograd.grad(loss_fn(x_rec_ri), x_rec_ri, create_graph=False)[0].detach()

        q = minres(lambda input: hessian_vector_product(
                    sigma,
                    x_rec_ri.detach(),
                    input,
                    data_fidelity,
                    y,
                    regularizer,
                    lmbd,
                    physics,
                ),
                grad_loss,
                max_iter=1000,
                tol=1e-4,#1e-5
            )

        regularizer = jac_vector_product(sigma, x_rec_ri, q, data_fidelity, y, regularizer, lmbd, physics)
        optimizer.step()
        torch.cuda.empty_cache()
        
    scheduler.step()
    avg_loss = loss_ / len(volumes)
    avg_psnr = psnr_ / len(volumes)
    wandb.log({"Epoch": epoch+1, "Loss": avg_loss, "PSNR": avg_psnr, "Gradient norm": grad_norm(regularizer)})
    
    if (epoch + 1) % 10 == 0:
        torch.save(regularizer.state_dict(), f"weights/{regularizer_name}_tuned_ckpt_{epoch+1}.pt")
wandb.finish()
