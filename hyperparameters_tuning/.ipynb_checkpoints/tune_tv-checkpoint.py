import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim, sum_of_squares, _load_volumes, PSNR_MRI, L2_precon
from deepinv.optim.prior import TVPrior#, WaveletPrior
from deepinv.optim import ADMM#, HQS
from baselines.drunet.drunet_base import DRUNet
import deepinv as dinv
from mrinufft import get_density
import gc
import os
import time
import argparse
import warnings
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

seed = 0
torch.random.manual_seed(seed)  # make results deterministic

parser = argparse.ArgumentParser(description="Choosing the training setting")
parser.add_argument("--root", type=str, default='../../../../../../../../LOCAL/mri_data')
inp = parser.parse_args()
root = inp.root + "/Val/_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "gpunufft"
scaler = 1e-6 
coils = 12 # number of coils in each volume
noise_level = 2e-3

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
           'e14531s6_P68096.7.h5.npy',
           'e14110s3_P59904.7.h5.npy',
           'e15652s14_P51712.7.h5.npy'] 
           
# TVPrior
prior = TVPrior(def_crit=1e-4)

# Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
x_0 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[0])),-1, 0)) #[coils,H,W,D] and complex dtype
x_1 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[1])),-1, 0))
x_2 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[2])),-1, 0))
x_3 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[3])),-1, 0))
x_4 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[4])),-1, 0))

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
density_0 = get_density("pipe", new_kspace_loc_0, x_0.shape[1:], backend=backend, num_iterations=10).astype(np.float32)
weights_0 = torch.from_numpy(density_0).to(device)
data_fidelity_0 = L2_precon(weights_0) # custom data fidelity
E_est_0 = get_operator(backend)(
    new_kspace_loc_0,
    x_0.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_0},
    density=False,
    use_gpu_direct=True,
)
physics_0 = MRINUFFTPhysicsRI(E_est_0)
y_0 = torch.from_numpy(y_grappa_0).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_0 = physics_0.A_adjoint(y_0)

density_1 = get_density("pipe", new_kspace_loc_1, x_1.shape[1:], backend=backend, num_iterations=10).astype(np.float32)
weights_1 = torch.from_numpy(density_1).to(device)
data_fidelity_1 = L2_precon(weights_1) # custom data fidelity
E_est_1 = get_operator(backend)(
    new_kspace_loc_1,
    x_1.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_1},
    density=False,
    use_gpu_direct=True,
)
physics_1 = MRINUFFTPhysicsRI(E_est_1)
y_1 = torch.from_numpy(y_grappa_1).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_1 = physics_1.A_adjoint(y_1)

density_2 = get_density("pipe", new_kspace_loc_2, x_2.shape[1:], backend=backend, num_iterations=10).astype(np.float32)
weights_2 = torch.from_numpy(density_2).to(device)
data_fidelity_2 = L2_precon(weights_2) # custom data fidelity
E_est_2 = get_operator(backend)(
    new_kspace_loc_2,
    x_2.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_2},
    density=False,
    use_gpu_direct=True,
)
physics_2 = MRINUFFTPhysicsRI(E_est_2)
y_2 = torch.from_numpy(y_grappa_2).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_2 = physics_2.A_adjoint(y_2)

density_3 = get_density("pipe", new_kspace_loc_3, x_3.shape[1:], backend=backend, num_iterations=10).astype(np.float32)
weights_3 = torch.from_numpy(density_3).to(device)
data_fidelity_3 = L2_precon(weights_3) # custom data fidelity
E_est_3 = get_operator(backend)(
    new_kspace_loc_3,
    x_3.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_3},
    density=False,
    use_gpu_direct=True,
)
physics_3 = MRINUFFTPhysicsRI(E_est_3)
y_3 = torch.from_numpy(y_grappa_3).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_3 = physics_3.A_adjoint(y_3)

density_4 = get_density("pipe", new_kspace_loc_4, x_4.shape[1:], backend=backend, num_iterations=10).astype(np.float32)
weights_4 = torch.from_numpy(density_4).to(device)
data_fidelity_4 = L2_precon(weights_4) # custom data fidelity
E_est_4 = get_operator(backend)(
    new_kspace_loc_4,
    x_4.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_4},
    density=False,
    use_gpu_direct=True,
)
physics_4 = MRINUFFTPhysicsRI(E_est_4)
y_4 = torch.from_numpy(y_grappa_4).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_4 = physics_4.A_adjoint(y_4)

# Reference/Ground Truth (Adjoint coil combination)
smaps_0 = torch.from_numpy(E_est_0.smaps)
x_gt_0 = torch.sum(torch.conj(smaps_0) * x_0, axis=0)
x_gt_ri_0 = complex_to_ri(x_gt_0)
reference_0 = torch.abs(x_gt_0) # Magnitude

smaps_1 = torch.from_numpy(E_est_1.smaps)
x_gt_1 = torch.sum(torch.conj(smaps_1) * x_1, axis=0)
x_gt_ri_1 = complex_to_ri(x_gt_1)
reference_1 = torch.abs(x_gt_1) # Magnitude

smaps_2 = torch.from_numpy(E_est_2.smaps)
x_gt_2 = torch.sum(torch.conj(smaps_2) * x_2, axis=0)
x_gt_ri_2 = complex_to_ri(x_gt_2)
reference_2 = torch.abs(x_gt_2) # Magnitude

smaps_3 = torch.from_numpy(E_est_3.smaps)
x_gt_3 = torch.sum(torch.conj(smaps_3) * x_3, axis=0)
x_gt_ri_3 = complex_to_ri(x_gt_3)
reference_3 = torch.abs(x_gt_3) # Magnitude

smaps_4 = torch.from_numpy(E_est_4.smaps)
x_gt_4 = torch.sum(torch.conj(smaps_4) * x_4, axis=0)
x_gt_ri_4 = complex_to_ri(x_gt_4)
reference_4 = torch.abs(x_gt_4) # Magnitude

del density_0, weights_0, density_1, weights_1, density_2, weights_2, density_3, weights_3, density_4, weights_4
torch.cuda.empty_cache()
        
# Optimization
k = 0
best_psnr = -float("inf")

for lmbd in [1e-3, 1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3, 1.7e-3, 1.8e-3, 1.9e-3, 2e-3]:
    
    avg_psnr = 0.0
    for y, physics, x_adj_ri, reference, data_fidelity in [(y_0, physics_0, x_adj_ri_0, reference_0, data_fidelity_0), (y_1, physics_1, x_adj_ri_1, reference_1, data_fidelity_1), (y_2, physics_2, x_adj_ri_2, reference_2, data_fidelity_2), (y_3, physics_3, x_adj_ri_3, reference_3, data_fidelity_3), (y_4, physics_4, x_adj_ri_4, reference_4, data_fidelity_4)]:
        model = ADMM(
            prior=prior,
            data_fidelity=data_fidelity,
            g_first=False,
            stepsize=1.0,
            lambda_reg=lmbd,
            max_iter=40,
            crit_conv="residual",
            thres_conv=1e-3,
            verbose=True,
            early_stop=True,
            show_progress_bar = True,
        )
        with torch.no_grad():
            x_rec_ri = model(y, physics).detach().cpu()
        recon  = torch.abs(ri_to_complex(x_rec_ri)) # Magnitude of the reconstruction
        avg_psnr += psnr(recon, reference)
    avg_psnr /= len(volumes)
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        lmbd_best = lmbd
        k_best = k
    print(f"Trial {k}: psnr {avg_psnr.item():.2f} with value lmbd={lmbd}")
    print(f"Best trial so far {k_best}, with value lmbd={lmbd_best} and PSNR={best_psnr.item():.2f}")
    k += 1

print(f"Overall best trial {k_best}, with value lmbd={lmbd_best} and PSNR={best_psnr.item():.2f}")
