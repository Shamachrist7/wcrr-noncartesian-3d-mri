import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim, simulate_acs_data, sum_of_squares
from reg_architectures import ParameterLearningWrapper, WCRR3D
from evaluation.nmAPG3d_evaluation import reconstruct_nmAPG
import deepinv as dinv
import time
from deepinv.optim import L2
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

wandb.init(
        # Set the project where this run will be logged
        project="10vols_results", #project name
        name="wcrr_recons", #run name, originaly "wcrr_recons"
        config={
        "Algorithm": "nmAPG",
        "lamda": 5e-3,
        "sigma": 0.1,
        "max_iter": 100,
        })
os.makedirs("savings", exist_ok=True)

data_fidelity = L2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "gpunufft"
scaler = 1e-6 

# Load trajectory and get the k-space locations
traj, traj_params = read_trajectory("trajectory.bin", dwell_time=0.01/2)
traj = traj.copy()
traj[traj < -0.5] = -0.5
traj[traj > 0.5] = 0.5
dim = traj_params["dimension"]
kspace_loc = traj.reshape(-1, dim)

# Choose your multi-coil volume
volumes = [ 'e15521s3_P33280.7.h5.npy',
            'e15652s4_P45056.7.h5.npy',
            'e16673s3_P24576.7.h5.npy',
            'e14583s3_P21504.7.h5.npy',
            'e14110s3_P59904.7.h5.npy',
            'e15652s14_P51712.7.h5.npy',
            'e16673s13_P31744.7.h5.npy',
            'e15183s3_P52224.7.h5.npy',
            'e14498s5_P60928.7.h5.npy',
            'e14258s3_P76800.7.h5.npy']

# Parameters of nmAPG solver
step_size = 1e-1
max_iter = 100
tol = 1e-4  # tolerance for the relative error (stopping criterion)

# Define the WCRR prior
reg = WCRR3D( 
    weak_convexity=1.0, 
    nb_channels=[2,4,8,32],
    filter_sizes=[3, 3, 3],
    rotations=True,
).to(device)
if regularizer_name=="WCRR":
    pretrained = "weights/bilevel_Denoising/WCRR_SP_bilevel_IFT_ckpt_1000.pt"
elif regularizer_name=="CRR":
    pretrained = "weights/bilevel_Denoising/CRR_bilevel_IFT_ckpt_500.pt"
else:
    raise ValueError("Wrong regularizer name!")
regularizer = ParameterLearningWrapper(reg, device=device)
regularizer.load_state_dict(torch.load(pretrained, weights_only=True, map_location=device))
regularizer.eval()

# Hyperparameters
lmbd = 5e-3
sigma = 0.1
sigma = torch.tensor([sigma], device=device)

for i, volume in enumerate(volumes):
    # Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
    x = scaler * np.moveaxis(np.load(os.path.join(root, volume)),-1, 0) #[coils,H,W,D] and complex dtype
    coils = x.shape[0] # number of coils in the volume

    # Forward NUFFT that takes coil images -> k-space
    F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True)
    y_np = F_raw.op(x) # simulates the undersampled kspace volume y. The zero-filled recon comes from it

    # GRAPPA reconstruct the center of k-space and append the data, basis for our regularizers
    acs = None#simulate_acs_data(x)
    new_kspace_loc, y_grappa = do_grappa_and_append_data(kspace_loc, y_np, traj_params, af=(2, 2), acs=acs)

    # Build reconstruction operator that ESTIMATES smaps from y_grappa
    E_est = get_operator(backend)(
        new_kspace_loc,
        x.shape[1:],
        n_coils=coils,
        smaps={"name": "low_frequency", "kspace_data": y_grappa},
        density=True,
    )

    physics = MRINUFFTPhysicsRI(E_est)

    y = torch.from_numpy(y_grappa).to(device) # ACS reconstructed with grappa (In the k-space)
    x_adj_ri = physics.A_adjoint(y)
    # Reference/Ground Truth (Adjoint coil combination)
    x_gt = np.sum(np.conj(E_est.smaps) * x, axis=0)
    x_gt_ri = complex_to_ri(torch.from_numpy(x_gt)).to(device) # In the RI space

    # Reconstruction
    with torch.no_grad():
        x_rec_ri, stats = reconstruct_nmAPG(
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
		        x_init=x_adj_ri, #Initialize as GRAPPA recon
		        return_stats=True)                  

    # Compute the magnitude before saving
    reference = torch.abs(ri_to_complex(x_gt_ri)).detach().cpu()
    grappa_recon  = torch.abs(ri_to_complex(x_adj_ri)).detach().cpu()
    wcrr_recon  = torch.abs(ri_to_complex(x_rec_ri)).detach().cpu()
    # Handle the zero-filled recon
    x_zf = F_raw.adj_op(y_np) #zero-filled reconstruction
    zf_recon = torch.from_numpy(sum_of_squares(x_zf)) # Magnitude

    wandb.log({"volume_idx": i, "psnr_zf": psnr(zf_recon, reference), "ssim_zf": ssim(zf_recon, reference), "psnr_grappa": psnr(grappa_recon, reference), "ssim_grappa": ssim(grappa_recon, reference), "psnr_wcrr": psnr(wcrr_recon, reference), "ssim_wcrr": ssim(wcrr_recon, reference)})
    if i < 5:
        torch.save(reference, f"savings/volume_{i}_gt.pt")
        torch.save(zf_recon, f"savings/volume_{i}_zf.pt")
        torch.save(grappa_recon, f"savings/volume_{i}_grappa.pt")
        torch.save(wcrr_recon, f"savings/volume_{i}_wcrr.pt")
        
print("Reconstructions finished!")
wandb.finish()
