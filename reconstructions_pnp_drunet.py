import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim, simulate_acs_data, sum_of_squares
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from baselines.drunet.drunet_base import DRUNet
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
inp = parser.parse_args()
root = inp.root + "/Val/_images"

wandb.init(
        # Set the project where this run will be logged
        project="10vols_results",
        name="drunet_recons",
        config={
        "Algorithm": "ADMM",
        "stepsize": 10.0,
        "g_param": 0.01,
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

# Load the DRUNet weights
drunet = DRUNet(in_channels=2, out_channels=2, dim=3, pretrained=None).to(device)
weights = "weights/drunet/drunet_3d_complex_denoise.pth"#"weights/drunet/ckpts/drunet_supervised_denoising_raw_0_sigma_0.01_0.1/25-07-30-12:13:56/ckp_11999.pth.tar"
state_dict = torch.load(weights, map_location=device)#['state_dict']
#state_dict = {k.replace('backbone_net.', ''): v for k, v in state_dict.items()}
drunet.load_state_dict(state_dict)
denoiser = drunet.eval().to(device)
prior = PnP(denoiser=denoiser)

# Parameters of ADMM solver
params_algo = {
    "stepsize": 10.0,
    "g_param":  0.01,
}
max_iter = 100
model = optim_builder(
    iteration="ADMM",#"HQS"
    prior=prior,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    early_stop=True,
    verbose=True,
    params_algo=params_algo,
    thres_conv=1e-4,
)
model.fixed_point.show_progress_bar = True

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
        x_rec_ri, metrics = model(y, physics, x_gt=x_gt_ri, compute_metrics=True)                  

    # Compute the magnitude before saving
    reference = torch.abs(ri_to_complex(x_gt_ri)).detach().cpu()
    grappa_recon  = torch.abs(ri_to_complex(x_adj_ri)).detach().cpu()
    drunet_recon  = torch.abs(ri_to_complex(x_rec_ri)).detach().cpu()
    # Handle the zero-filled recon
    x_zf = F_raw.adj_op(y_np) #zero-filled reconstruction
    zf_recon = torch.from_numpy(sum_of_squares(x_zf)) # Magnitude

    wandb.log({"volume_idx": i, "psnr_zf": psnr(zf_recon, reference), "ssim_zf": ssim(zf_recon, reference), "psnr_grappa": psnr(grappa_recon, reference), "ssim_grappa": ssim(grappa_recon, reference), "psnr_drunet": psnr(drunet_recon, reference), "ssim_drunet": ssim(drunet_recon, reference)})
    if i < 5:
        torch.save(reference, f"savings/volume_{i}_gt.pt")
        torch.save(zf_recon, f"savings/volume_{i}_zf.pt")
        torch.save(grappa_recon, f"savings/volume_{i}_grappa.pt")
        torch.save(drunet_recon, f"savings/volume_{i}_drunet2.pt")
        
print("Reconstructions finished!")
wandb.finish()
