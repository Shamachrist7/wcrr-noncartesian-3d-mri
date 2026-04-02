import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import cupy as cp
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, L2_precon, compute_mask, masked_psnr
from deepinv.optim.prior import WaveletPrior
from deepinv.optim import FISTA
from mrinufft.extras.smaps import get_smaps
import gc
import os
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
backend = "cufinufft"
scaler = 1e-6 
coils = 12 # number of coils in each volume
noise_level = 2e-3
data_fidelity = L2_precon(weights=torch.tensor(1.0))

# Load trajectory and get the k-space locations
traj, traj_params = read_trajectory("gs.bin", dwell_time=0.01/2)
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
           
# Parameters of the FISTA solver
max_iter = 500
tol = 5e-3  # tolerance for the relative error (stopping criterion)
prior = WaveletPrior(level=4, wv="db4", p=1, wvdim=3)

# Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
x_0 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[0])),-1, 0)) #[coils,H,W,D] and complex dtype
x_1 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[1])),-1, 0))
x_2 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[2])),-1, 0))
x_3 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[3])),-1, 0))
x_4 = torch.from_numpy(scaler * np.moveaxis(np.load(os.path.join(root, volumes[4])),-1, 0))

# Forward NUFFT that takes coil images -> k-space & simulate the corruption
F_raw = get_operator(backend)(kspace_loc, x_0.shape[1:], n_coils=coils, density=True, squeeze_dims=True)
y_0 = F_raw.op(x_0) # simulates the undersampled kspace volume y. The zero-filled recon comes from it
y_0 = y_0 + noise_level * torch.randn_like(y_0)
y_1 = F_raw.op(x_1)
y_1 = y_1 + noise_level * torch.randn_like(y_1)
y_2 = F_raw.op(x_2)
y_2 = y_2 + noise_level * torch.randn_like(y_2)
y_3 = F_raw.op(x_3)
y_3 = y_3 + noise_level * torch.randn_like(y_3)
y_4 = F_raw.op(x_4)
y_4 = y_4 + noise_level * torch.randn_like(y_4)

# GRAPPA reconstruct the center of k-space and append the data, basis for our regularizers
new_kspace_loc_0, y_grappa_0 = do_grappa_and_append_data(kspace_loc, y_0, traj_params, af=(2, 2))
new_kspace_loc_1, y_grappa_1 = do_grappa_and_append_data(kspace_loc, y_1, traj_params, af=(2, 2))
new_kspace_loc_2, y_grappa_2 = do_grappa_and_append_data(kspace_loc, y_2, traj_params, af=(2, 2))
new_kspace_loc_3, y_grappa_3 = do_grappa_and_append_data(kspace_loc, y_3, traj_params, af=(2, 2))
new_kspace_loc_4, y_grappa_4 = do_grappa_and_append_data(kspace_loc, y_4, traj_params, af=(2, 2))

# Build reconstruction operator that ESTIMATES smaps from y (zero-filled) (For each volume)
Smaps_0 = get_smaps("espirit")(
    kspace_loc,
    x_0.shape[1:],
    kspace_data=cp.asarray(y_0),
    density=F_raw.density,
    backend=backend,
    decim=4,
)
E_est_0 = get_operator(backend)(
    kspace_loc,
    x_0.shape[1:],
    n_coils=coils,
    smaps=Smaps_0.get(),
    squeeze_dims=True,
)
physics_0 = MRINUFFTPhysicsRI(E_est_0)
y_grappa_0 = torch.from_numpy(y_grappa_0).to(device) # ACS reconstructed with grappa (In the k-space)
### Grappa + DCp recon
nufft_grappa_0 = get_operator(backend)(new_kspace_loc_0, x_0.shape[1:], n_coils=coils, smaps=Smaps_0.get(), density=True, squeeze_dims=True)
dcp_grappa_ri_0 = complex_to_ri(nufft_grappa_0.adj_op(y_grappa_0))


Smaps_1 = get_smaps("espirit")(
    kspace_loc,
    x_0.shape[1:],
    kspace_data=cp.asarray(y_1),
    density=F_raw.density,
    backend=backend,
    decim=4,
)
E_est_1 = get_operator(backend)(
    kspace_loc,
    x_0.shape[1:],
    n_coils=coils,
    smaps=Smaps_1.get(),
    squeeze_dims=True,
)
physics_1 = MRINUFFTPhysicsRI(E_est_1)
y_grappa_1 = torch.from_numpy(y_grappa_1).to(device) # ACS reconstructed with grappa (In the k-space)
### Grappa + DCp recon
nufft_grappa_1 = get_operator(backend)(new_kspace_loc_1, x_0.shape[1:], n_coils=coils, smaps=Smaps_1.get(), density=True, squeeze_dims=True)
dcp_grappa_ri_1 = complex_to_ri(nufft_grappa_1.adj_op(y_grappa_1))


Smaps_2 = get_smaps("espirit")(
    kspace_loc,
    x_0.shape[1:],
    kspace_data=cp.asarray(y_2),
    density=F_raw.density,
    backend=backend,
    decim=4,
)
E_est_2 = get_operator(backend)(
    kspace_loc,
    x_0.shape[1:],
    n_coils=coils,
    smaps=Smaps_2.get(),
    squeeze_dims=True,
)
physics_2 = MRINUFFTPhysicsRI(E_est_2)
y_grappa_2 = torch.from_numpy(y_grappa_2).to(device) # ACS reconstructed with grappa (In the k-space)
### Grappa + DCp recon
nufft_grappa_2 = get_operator(backend)(new_kspace_loc_2, x_0.shape[1:], n_coils=coils, smaps=Smaps_2.get(), density=True, squeeze_dims=True)
dcp_grappa_ri_2 = complex_to_ri(nufft_grappa_2.adj_op(y_grappa_2))


Smaps_3 = get_smaps("espirit")(
    kspace_loc,
    x_0.shape[1:],
    kspace_data=cp.asarray(y_3),
    density=F_raw.density,
    backend=backend,
    decim=4,
)
E_est_3 = get_operator(backend)(
    kspace_loc,
    x_0.shape[1:],
    n_coils=coils,
    smaps=Smaps_3.get(),
    squeeze_dims=True,
)
physics_3 = MRINUFFTPhysicsRI(E_est_3)
y_grappa_3 = torch.from_numpy(y_grappa_3).to(device) # ACS reconstructed with grappa (In the k-space)
### Grappa + DCp recon
nufft_grappa_3 = get_operator(backend)(new_kspace_loc_3, x_0.shape[1:], n_coils=coils, smaps=Smaps_3.get(), density=True, squeeze_dims=True)
dcp_grappa_ri_3 = complex_to_ri(nufft_grappa_3.adj_op(y_grappa_3))


Smaps_4 = get_smaps("espirit")(
    kspace_loc,
    x_0.shape[1:],
    kspace_data=cp.asarray(y_4),
    density=F_raw.density,
    backend=backend,
    decim=4,
)
E_est_4 = get_operator(backend)(
    kspace_loc,
    x_0.shape[1:],
    n_coils=coils,
    smaps=Smaps_4.get(),
    squeeze_dims=True,
)
physics_4 = MRINUFFTPhysicsRI(E_est_4)
y_grappa_4 = torch.from_numpy(y_grappa_4).to(device) # ACS reconstructed with grappa (In the k-space)
### Grappa + DCp recon
nufft_grappa_4 = get_operator(backend)(new_kspace_loc_4, x_0.shape[1:], n_coils=coils, smaps=Smaps_4.get(), density=True, squeeze_dims=True)
dcp_grappa_ri_4 = complex_to_ri(nufft_grappa_4.adj_op(y_grappa_4))

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

# Compute the operator norm of the forward operator, which is needed to set the stepsize of FISTA. We compute it with the power method, which consists in applying the forward and adjoint operators iteratively on a random vector until convergence. The value returned is an upper bound of the true operator norm, which guarantees convergence of FISTA.
opnorm = physics_0.compute_sqnorm(torch.randn_like(x_gt_ri_0, device=device), max_iter=20).item()

del F_raw, Smaps_0, Smaps_1, Smaps_2, Smaps_3, Smaps_4, nufft_grappa_0, nufft_grappa_1, nufft_grappa_2, nufft_grappa_3, nufft_grappa_4, E_est_0, E_est_1, E_est_2, E_est_3, E_est_4, new_kspace_loc_0, new_kspace_loc_1, new_kspace_loc_2, new_kspace_loc_3, new_kspace_loc_4, y_grappa_0, y_grappa_1, y_grappa_2, y_grappa_3, y_grappa_4, smaps_0, smaps_1, smaps_2, smaps_3, smaps_4, x_gt_0, x_gt_1, x_gt_2, x_gt_3, x_gt_4, x_gt_ri_0, x_gt_ri_1, x_gt_ri_2, x_gt_ri_3, x_gt_ri_4, x_0, x_1, x_2, x_3, x_4
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
        
# Optimization
k = 0
best_psnr = -float("inf")

for lmbd in [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]:
    # Define the model
    model = FISTA(
            prior=prior,
            data_fidelity=data_fidelity,
            stepsize=1/opnorm,
            lambda_reg=lmbd,
            max_iter=max_iter,
            thres_conv=tol,
            verbose=True,
            early_stop=True,
            show_progress_bar = True,
        )
    avg_psnr = 0.0
    for y, physics, dcp_grappa_ri, reference in [(y_0, physics_0, dcp_grappa_ri_0, reference_0), (y_1, physics_1, dcp_grappa_ri_1, reference_1), (y_2, physics_2, dcp_grappa_ri_2, reference_2), (y_3, physics_3, dcp_grappa_ri_3, reference_3), (y_4, physics_4, dcp_grappa_ri_4, reference_4)]:
        
        with torch.no_grad():
            x_rec_ri = model(y.to(device), physics, init=(dcp_grappa_ri.to(device), dcp_grappa_ri.to(device)), x_gt=None, compute_metrics=False).detach().cpu()
        recon  = torch.abs(ri_to_complex(x_rec_ri)) # Magnitude of the reconstruction
        mask = compute_mask(reference.numpy())
        avg_psnr += masked_psnr(reference.numpy(), recon.numpy(), mask)
    avg_psnr /= len(volumes)
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        lmbd_best = lmbd
        k_best = k
    print(f"Trial {k}: psnr {avg_psnr:.2f} with value lmbd={lmbd}")
    print(f"Best trial so far {k_best}, with value lmbd={lmbd_best} and PSNR={best_psnr:.2f}")
    k += 1

print(f"Overall best trial {k_best}, with value lmbd={lmbd_best} and PSNR={best_psnr:.2f}")

