import torch
import numpy as np
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim
from reg_architectures import ParameterLearningWrapper, WCRR3D
from evaluation import reconstruct_nmAPG
from deepinv.optim import L2
import optuna
import os
import warnings
warnings.filterwarnings("ignore")

data_fidelity = L2()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
backend = "gpunufft"
scaler = 1e-6 

# Load trajectory and get the k-space locations
traj, traj_params = read_trajectory("trajectory.bin", dwell_time=0.01/2)
traj = traj.copy()
traj[traj < -0.5] = -0.5
traj[traj > 0.5] = 0.5
dim = traj_params["dimension"]
kspace_loc = traj.reshape(-1, dim)

# The 05 chosen volumes to validate on
volumes = ['e14120s11_P66048.7.h5.npy',
         'e14692s5_P14848.7.h5.npy',
         'e14531s6_P68096.7.h5.npy',
         'e14691s3_P06656.7.h5.npy',
         'e14584s5_P30208.7.h5.npy']
root = "../../../../../../../LOCAL/mri_data/Val/_images" # Here, replace with the root directory of those volumes

# Load the WCRR regularizer weights
reg = WCRR3D(weak_convexity=1.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=True).to(device)
pretrained = "weights/bilevel_Denoising/WCRR_3by3_32_bilevel_IFT_ckpt_100.pt"
regularizer = ParameterLearningWrapper(reg, device=device)
regularizer.load_state_dict(torch.load(pretrained, weights_only=True, map_location=device))
regularizer.eval()

# Parameters of the nmAPG solver
step_size = 1e-1
max_iter = 30 # set to 30 only to speed up tuning, set to 100 for real evaluation
tol = 1e-4  # tolerance for the relative error (stopping criterion)

# Objective function for Optuna
def objective(trial):

    # Define hyperparameters to tune
    lmbd = trial.suggest_float('lmbd', 1e-4, 1e-1, log=True) # can be tuned but in [0.0, 1.0], to keep a rho<=1 weakly convex regularizer and conserve an overall convex objective
    sigma = trial.suggest_float('sigma', 0.01, 0.1) # can be tuned in [0.01, 0.1] (range for which the regularizer has been trained)
    sigma = torch.tensor([sigma], device=device)
    regularizer_scale = trial.suggest_float('regularizer_scale', 0.0, 1.0)
    regularizer_scale = torch.tensor(regularizer_scale, device=device)
    regularizer.scale = torch.nn.Parameter(regularizer_scale)

    avg_psnr = 0.0
    for i in range(len(volumes)):

        # Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
        x = scaler * np.moveaxis(np.load(os.path.join(root, volumes[i])),-1, 0) #[coils,H,W,D] and complex dtype
        coils = x.shape[0] # number of coils in the volumes

        # Forward NUFFT that takes coil images -> k-space
        F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils)
        y_np = F_raw.op(x) # simulates the undersampled kspace volume y. The zero-filled recon comes from it

        # GRAPPA reconstruct the center of k-space and append the data, basis for our regularizers
        new_kspace_loc, y_grappa = do_grappa_and_append_data(kspace_loc, y_np, traj_params, af=(2, 2))

        # Build reconstruction operator that ESTIMATES smaps from y_grappa
        E_est = get_operator(backend)(
            new_kspace_loc,
            x.shape[1:],
            n_coils=coils,
            smaps={"name": "low_frequency", "kspace_data": y_grappa},
        )

        physics = MRINUFFTPhysicsRI(E_est)

        y = torch.from_numpy(y_grappa).to(device) # ACS reconstructed with grappa (In the k-space)
        x_adj_ri = physics.A_adjoint(y)
        # Reference/Ground Truth (Adjoint coil combination)
        x_gt = np.sum(np.conj(E_est.smaps) * x, axis=0)
        x_gt_ri = complex_to_ri(torch.from_numpy(x_gt)).to(device) # In the RI space

        with torch.no_grad():
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
	                x_init=x_adj_ri, # Initialize with the adjoint of GRAPPA recon (and not the adjoint of the zero-filled)
	                multi_coil_mri=True,
	                return_stats=False,
	                )
        avg_psnr += psnr(x_rec_ri, x_gt_ri).item()
    
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
