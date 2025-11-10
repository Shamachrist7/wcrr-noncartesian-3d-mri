import torch
import numpy as np
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim
from reg_architectures import ParameterLearningWrapper, WCRR3D
from evaluation.nmAPG3d_evaluation import reconstruct_nmAPG
from deepinv.optim import L2
import optuna
import os
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
           
# Load the WCRR regularizer weights
reg = WCRR3D(weak_convexity=1.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=True).to(device)
if regularizer_name=="WCRR":
    pretrained = "weights/bilevel_Denoising/WCRR_SP_bilevel_IFT_ckpt_1000.pt"
elif regularizer_name=="CRR":
    pretrained = "weights/bilevel_Denoising/CRR_bilevel_IFT_ckpt_500.pt"
else:
    raise ValueError("Wrong regularizer name!")
regularizer = ParameterLearningWrapper(reg, device=device)
regularizer.load_state_dict(torch.load(pretrained, weights_only=True, map_location=device))
regularizer.eval()

# Parameters of the nmAPG solver
step_size = 1e-1
max_iter = 100
tol = 1e-4  # tolerance for the relative error (stopping criterion)

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
    lmbd = trial.suggest_float('lmbd', 1e-3, 1e-1, log=True) # can be tuned but in [0.0, 1.0], to keep a rho<=1 weakly convex regularizer and conserve an overall convex objective
    sigma = trial.suggest_categorical('sigma', [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]) # can be tuned in [0.01, 0.1] (range for which the regularizer has been trained)
    sigma = torch.tensor([sigma], device=device)

    with torch.no_grad():
        avg_psnr = 0.0
        for y, physics, x_adj_ri, reference in [(y_0, physics_0, x_adj_ri_0, reference_0), (y_1, physics_1, x_adj_ri_1, reference_1), (y_2, physics_2, x_adj_ri_2, reference_2)]:
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
