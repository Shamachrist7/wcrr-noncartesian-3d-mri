import torch
import numpy as np
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim, power_iteration_L_RI
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import WaveletPrior
from deepinv.optim import L2
import deepinv as dinv
import optuna
import os
import warnings
import argparse
warnings.filterwarnings("ignore")

torch.random.manual_seed(0)  # make results deterministic

parser = argparse.ArgumentParser(description="Choosing the training setting")
parser.add_argument("--root", type=str, default='../../../../../../../../LOCAL/mri_data')
inp = parser.parse_args()
root = inp.root + "/Val/_images"

data_fidelity = L2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "gpunufft"
scaler = 1e-6 
coils = 12 # number of coils in each volume

# Load trajectory and get the k-space locations
traj, traj_params = read_trajectory("trajectory.bin", dwell_time=0.01/2)
traj = traj.copy()
traj[traj < -0.5] = -0.5
traj[traj > 0.5] = 0.5
dim = traj_params["dimension"]
kspace_loc = traj.reshape(-1, dim)

# The 03 chosen volumes to tune hyperparameters on
volumes = ['e14120s11_P66048.7.h5.npy',
           'e14692s5_P14848.7.h5.npy',
           'e14531s6_P68096.7.h5.npy']

# Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
x_0 = scaler * np.moveaxis(np.load(os.path.join(root, volumes[0])),-1, 0) #[coils,H,W,D] and complex dtype
x_1 = scaler * np.moveaxis(np.load(os.path.join(root, volumes[1])),-1, 0)
x_2 = scaler * np.moveaxis(np.load(os.path.join(root, volumes[2])),-1, 0)

# Forward NUFFT that takes coil images -> k-space
F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True)
y_np_0 = F_raw.op(x_0) # simulates the undersampled kspace volume y. The zero-filled recon comes from it
y_np_1 = F_raw.op(x_1)
y_np_2 = F_raw.op(x_2)

# GRAPPA reconstruct the center of k-space and append the data, basis for our regularizers
new_kspace_loc_0, y_grappa_0 = do_grappa_and_append_data(kspace_loc, y_np_0, traj_params, af=(2, 2))
new_kspace_loc_1, y_grappa_1 = do_grappa_and_append_data(kspace_loc, y_np_1, traj_params, af=(2, 2))
new_kspace_loc_2, y_grappa_2 = do_grappa_and_append_data(kspace_loc, y_np_2, traj_params, af=(2, 2))

# Build reconstruction operator that ESTIMATES smaps from y_grappa (For each volume)
E_est_0 = get_operator(backend)(
    new_kspace_loc_0,
    x_0.shape[1:],
    n_coils=coils,
    smaps={"name": "low_frequency", "kspace_data": y_grappa_0},
    density=True,
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
)
physics_2 = MRINUFFTPhysicsRI(E_est_2)
y_2 = torch.from_numpy(y_grappa_2).to(device) # ACS reconstructed with grappa (In the k-space)
x_adj_ri_2 = physics_2.A_adjoint(y_2)

# Reference/Ground Truth (Adjoint coil combination)
x_gt_0 = np.sum(np.conj(E_est_0.smaps) * x_0, axis=0)
reference_0 = torch.abs(torch.from_numpy(x_gt_0)) # Magnitude

x_gt_1 = np.sum(np.conj(E_est_1.smaps) * x_1, axis=0)
reference_1 = torch.abs(torch.from_numpy(x_gt_1)) # Magnitude

x_gt_2 = np.sum(np.conj(E_est_2.smaps) * x_2, axis=0)
reference_2 = torch.abs(torch.from_numpy(x_gt_2)) # Magnitude
        
# Objective function for Optuna
def objective(trial):

    # Define hyperparameters to tune
    lamda = trial.suggest_float('lamda', 1e-5, 1e-1, log=True)
    wv = trial.suggest_categorical('wv', ["haar", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"])
    level = trial.suggest_int('level', 2, 8)
    
    # Lipschitz bound estimated with power iteration
    L = 1.58170747756958 #power_iteration_L_RI(physics, x_gt_ri.shape, iters=100, device=device)
    stepsize = 1.0 / L
    # Parameters of ADMM solver
    params_algo = {"stepsize": stepsize, "lambda": lamda}
    # l1-wavelets prior
    prior = WaveletPrior(level=level, wv=wv, p=1, wvdim=3, device=device)
  
    model = optim_builder(
        iteration="ADMM",
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=100,
        early_stop=True,
        verbose=True,
        params_algo=params_algo,
        thres_conv=1e-3,
    )
    model.fixed_point.show_progress_bar = True

    with torch.no_grad():
        avg_psnr = 0.0
        for y, physics, reference in [(y_0, physics_0, reference_0), (y_1, physics_1, reference_1), (y_2, physics_2, reference_2)]:
            x_rec_ri = model(y, physics)
	    recon  = torch.abs(ri_to_complex(x_rec_ri)).detach().cpu() # Magnitude of the reconstruction
	    avg_psnr += psnr(recon, reference)
	avg_psnr /= len(volumes)
    return avg_psnr

# Optuna study to tune hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
# Print the best hyperparameters
print(f'Best trial: {study.best_trial.value}')
print('Best hyperparameters: ')
for key, value in study.best_trial.params.items():
    print(f'{key}: {value}')
