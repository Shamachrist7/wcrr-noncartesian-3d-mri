import torch
import numpy as np
import matplotlib.pyplot as plt
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim
from reg_architectures import ParameterLearningWrapper, WCRR3D
from evaluation.nmAPG3d_evaluation import reconstruct_nmAPG
from deepinv.optim import L2
import os
import warnings
warnings.filterwarnings("ignore")

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
volume = 'e14120s11_P66048.7.h5.npy'
root = "../../../../../../../LOCAL/mri_data/Val/_images" # Here, replace with the root directory of that volume

# Load the WCRR regularizer weights
reg = WCRR3D(weak_convexity=0.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=True).to(device)
pretrained = "weights/bilevel_Denoising/CRR_3by3_32_bilevel_IFT_ckpt_100.pt"
regularizer = ParameterLearningWrapper(reg, device=device)
regularizer.load_state_dict(torch.load(pretrained, weights_only=True, map_location=device))
regularizer.eval()

# Parameters of the nmAPG solver
step_size = 1e-1
max_iter = 100
tol = 1e-4  # tolerance for the relative error (stopping criterion)

# Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
x = scaler * np.moveaxis(np.load(os.path.join(root, volume)),-1, 0) #[coils,H,W,D] and complex dtype
coils = x.shape[0] # number of coils in the volume

# Forward NUFFT that takes coil images -> k-space
F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True)
y_np = F_raw.op(x) # simulates the undersampled kspace volume y. The zero-filled recon comes from it

# GRAPPA reconstruct the center of k-space and append the data, basis for our regularizers
new_kspace_loc, y_grappa = do_grappa_and_append_data(kspace_loc, y_np, traj_params, af=(2, 2))

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
        
# (Tuned) hyperparameters
lmbd = 5e-3 # Regularization strength
sigma = 0.1 # Denoising power
sigma = torch.tensor([sigma], device=device)

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
                return_stats=False,
                    )
                    

# Compute the magnitude for plotting
reference = torch.abs(ri_to_complex(x_gt_ri)).detach().cpu()
grappa_recon  = torch.abs(ri_to_complex(x_adj_ri)).detach().cpu()
crr_recon  = torch.abs(ri_to_complex(x_rec_ri)).detach().cpu()

# -----------------------------------
# Plots (mid-slice)
# -----------------------------------
mid =  x.shape[-1] // 2

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(reference[..., mid], cmap='gray'); plt.title('Reference'); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(grappa_recon[..., mid],  cmap='gray'); plt.title(f'Grappa, psnr:{psnr(x_adj_ri, x_gt_ri)}'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(crr_recon[..., mid],  cmap='gray'); plt.title(f'CRR, psnr:{psnr(x_rec_ri, x_gt_ri)}'); plt.axis('off')
plt.tight_layout(); plt.show()
