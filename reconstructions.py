import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim, sum_of_squares, _load_volumes, PSNR_MRI, L2_precon
from reg_architectures import WCRR3D
from deepinv.optim.prior import PnP, TVPrior, WaveletPrior
from deepinv.optim import ADMM#, HQS
from baselines.drunet.drunet_base import DRUNet
from evaluation.nmAPG3d_evaluation import reconstruct_nmAPG
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

seed = 0 # 0, 1, 2, 3, 4 (run each of them  to be able to compute the confidence interval)
torch.random.manual_seed(seed)  # make results deterministic

parser = argparse.ArgumentParser(description="reconstructions")
parser.add_argument("--root", type=str, default='../../../../../../../../LOCAL/mri_data')
parser.add_argument("--method", type=str, default="WCRR")
parser.add_argument("--coil", type=int, default=12)
inp = parser.parse_args()
coil = inp.coil # 12 or 32
method = inp.method # "wcrr", "tv", "wv", "drunet", "wcrr_no_rot"
root = inp.root + f"/Test/{coil}coil"

wandb.init(
        # Set the project where this run will be logged
        project=f"{coil}coil_results_seed_{seed}", # project name
        name=f"{method.lower()}_recons",
        config={
        "max_iter": 40,
        "noise_level": 2e-3,
        })
os.makedirs(f"savings_{coil}coil", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "gpunufft"
scaler = 1e-6 # data normalizer
noise_level = 2e-3
max_iter = 30 # Maximum number of iterations
thres_conv = 1e-3 # convergence threshold

# Load trajectory and get the k-space locations
traj, traj_params = read_trajectory("trajectory.bin", dwell_time=0.01/2)
traj = traj.copy()
traj[traj < -0.5] = -0.5
traj[traj > 0.5] = 0.5
dim = traj_params["dimension"]
kspace_loc = traj.reshape(-1, dim)

# your 50 multi-coil volumes (12 or 32 coils)
volumes = sorted([fn for fn in os.listdir(root) if fn.endswith(".h5")])[:15]

##### PnP-DRUNet pior and hyperparameters #####
if method.lower()=="drunet":
    drunet = DRUNet(in_channels=2, out_channels=2, dim=3, pretrained=None).to(device)
    drunet.load_state_dict(torch.load("weights/drunet/drunet_3d_complex_denoise.pth", map_location=device, weights_only=True))
    prior_drunet = PnP(denoiser=drunet.eval())
    stepsize_drunet = 2.5 #22.0
    sigma_drunet = 0.01
##### l1-wavelet prior and hyperparameters #####
if method.lower()=="wv":
    prior_wv = WaveletPrior(level=4, wv="db4", p=1, wvdim=3, device=device)
    stepsize_wv = 1.0
    lmbd_wv = 3.8e-2
##### TV prior and hyperparameters #####
if method.lower()=="tv":
    prior_tv = TVPrior(def_crit=1e-4)
    stepsize_tv = 1.0
    lmbd_tv = 1.3e-3
##### (Rotation invariant) WCRR prior and hyperparameters #####
if method.lower()=="wcrr":
    WCRR = WCRR3D(weak_convexity=1.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=True).to(device)
    WCRR.load_state_dict(torch.load("weights/bilevel_Denoising/WCRR_bilevel_IFT_ckpt_100.pt", weights_only=True, map_location=device))
    WCRR.eval()
    lmbd_wcrr = 0.07
    sigma_wcrr = 0.035
    sigma_wcrr = torch.tensor([sigma_wcrr], device=device)
##### (Not rotation invariant) WCRR_no_rot prior and hyperparameters #####
if method.lower()=="wcrr_no_rot":
    WCRR_no_rot = WCRR3D(weak_convexity=1.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=False).to(device)
    WCRR_no_rot.load_state_dict(torch.load("weights/bilevel_Denoising/WCRR_no_rotations_bilevel_IFT_ckpt_100.pt", weights_only=True, map_location=device))
    WCRR_no_rot.eval()
    lmbd_wcrr_no_rot = 0.1
    sigma_wcrr_no_rot = 0.025
    sigma_wcrr_no_rot = torch.tensor([sigma_wcrr_no_rot], device=device)

# reconstructions

for i, volume in enumerate(volumes):
    try:
        # Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
        x = torch.from_numpy(scaler * np.moveaxis(_load_volumes(os.path.join(root, volume)),-1, 0)) #[coils,H,W,D] and complex dtype
        coils = x.shape[0] # number of coils in the volume
    
        # Forward NUFFT that takes coil images -> k-space
        print(f"Simulation of the undersampled measurement {i+1}!")
        print("Start ... ")
        F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True)
        y_np = F_raw.op(x) # simulates the undersampled kspace volume y. The zero-filled recon comes from it
        z = (
            torch.randn_like(y_np)
            + 1j * torch.randn_like(y_np)
        )
        y_np = y_np + noise_level * z  
        print("Succesfully simulated!")
    
        # GRAPPA reconstruct the center of k-space, basis for our regularizers
        t1_grappa = time.time()
        new_kspace_loc, y_grappa = do_grappa_and_append_data(kspace_loc, y_np, traj_params, af=(2, 2), acs=None)
        dt1_grappa = time.time() - t1_grappa
    
        # Build reconstruction operator that ESTIMATES smaps from y_grappa
        print(f"Operator definition, DCp weights and smaps estimation from measurement {i+1}!")
        print("Start ... ")
    
        density = get_density("pipe", new_kspace_loc, x.shape[1:], backend=backend, num_iterations=10).astype(np.float32)
        weights = torch.from_numpy(density).to(device)
        data_fidelity = L2_precon(weights) # custom data fidelity
        
        E_est = get_operator(backend)(
            new_kspace_loc,
            x.shape[1:],
            n_coils=coils,
            smaps={"name": "low_frequency", "kspace_data": y_grappa},
            density= False,
            use_gpu_direct=True,
        )
        physics = MRINUFFTPhysicsRI(E_est)
        print("Succesfull!")
        
        y = torch.from_numpy(y_grappa).to(device) # ACS reconstructed with grappa (In the k-space)
        x_adj_ri = physics.A_adjoint(y) # Grappa adjoint without DCp (Initialization for all our iterative solvers)
    
        # Reference/Ground Truth (Adjoint coil combination)
        smaps = torch.from_numpy(E_est.smaps)
        x_gt = torch.sum(torch.conj(smaps) * x, axis=0)
        x_gt_ri = complex_to_ri(x_gt).to(device) # In the RI space
        reference = torch.abs(ri_to_complex(x_gt_ri)).detach().cpu()
        # zero-filled + DCp recon
        t1_zf = time.time()
        x_zf = F_raw.adj_op(y_np) #zero-filled reconstruction
        dt_zf = time.time() - t1_zf
        zf_recon = sum_of_squares(x_zf) # Its magnitude
        # Grappa + DCp recon
        t2_grappa = time.time()
        dcp_x_adj_ri = physics.A_adjoint(weights * y).detach().cpu()
        dt2_grappa = time.time() - t2_grappa
        grappa_recon  = torch.abs(ri_to_complex(dcp_x_adj_ri))#.detach().cpu() # Its magnitude
        #del F_raw, E_est, density, weights, new_kspace_loc, y_grappa # To free gpunufft
        #torch.cuda.empty_cache()
        # TV recon
        if method.lower()=="tv":
            solver_tv = ADMM(
                prior=prior_tv,
                g_first=False,
                data_fidelity=data_fidelity,
                stepsize=stepsize_tv,
                lambda_reg=lmbd_tv,
                max_iter=max_iter,
                crit_conv="residual",
                thres_conv=thres_conv,
                verbose=True,
                early_stop=True,
                show_progress_bar = True,
            )
            with torch.no_grad():
                t1_tv = time.time()
                x_rec_ri_tv = solver_tv(y, physics, x_gt=x_gt_ri, compute_metrics=False).detach().cpu()
                dt = time.time() - t1_tv
            recon  = torch.abs(ri_to_complex(x_rec_ri_tv))#.detach().cpu() # Its magnitude
        # l1-wavelet recon
        if method.lower()=="wv":
            solver_wv = ADMM(
                prior=prior_wv,
                g_first=False,
                data_fidelity=data_fidelity,
                stepsize=stepsize_wv,
                lambda_reg=lmbd_wv,
                max_iter=max_iter,
                crit_conv="residual",
                thres_conv=thres_conv,
                verbose=True,
                early_stop=True,
                show_progress_bar = True,
            )
            with torch.no_grad():
                t1_wv = time.time()
                x_rec_ri_wv = solver_wv(y, physics, x_gt=x_gt_ri, compute_metrics=False).detach().cpu()
                dt = time.time() - t1_wv
            recon  = torch.abs(ri_to_complex(x_rec_ri_wv))#.detach().cpu() # Its magnitude
        # PnP-DRUNet recon
        if method.lower()=="drunet":
            solver_drunet = ADMM(
                prior=prior_drunet,
                data_fidelity=data_fidelity,
                g_first=False,
                stepsize=stepsize_drunet,
                sigma_denoiser=sigma_drunet,
                max_iter=max_iter,
                crit_conv="residual",
                thres_conv=thres_conv,
                verbose=True,
                early_stop=True,
                show_progress_bar = True,
            )
            with torch.no_grad():
                t1_drunet = time.time()
                x_rec_ri_drunet = solver_drunet(y, physics, x_gt=x_gt_ri, compute_metrics=False).detach().cpu()
                dt = time.time() - t1_drunet
            recon  = torch.abs(ri_to_complex(x_rec_ri_drunet))#.detach().cpu() # Its magnitude
        # WCRR recon
        if method.lower()=="wcrr":
            with torch.no_grad():
                t1_wcrr = time.time()
                x_rec_ri_wcrr = reconstruct_nmAPG(
                        sigma_wcrr,
                        y,
                        physics,
                        data_fidelity,
                        WCRR,
                        lmbd_wcrr,
                        1e-1, # Stepsize_nmAPG (can be anything)
                        max_iter,
                        thres_conv,
                        verbose=True,
                        x_init=x_adj_ri, # Initialize as GRAPPA adj (without DCp)
                        x_gt=x_gt_ri,
                        return_stats=False,
                        ).detach().cpu()
                dt = time.time() - t1_wcrr                 
            recon  = torch.abs(ri_to_complex(x_rec_ri_wcrr))#.detach().cpu() # Its magnitude
        # WCRR_no_rot recon
        if method.lower()=="wcrr_no_rot":
            with torch.no_grad():
                t1_wcrr_no_rot = time.time()
                x_rec_ri_wcrr_no_rot = reconstruct_nmAPG(
                        sigma_wcrr_no_rot,
                        y,
                        physics,
                        data_fidelity,
                        WCRR_no_rot,
                        lmbd_wcrr_no_rot,
                        1e-1, # Stepsize_nmAPG (can be anything)
                        max_iter,
                        thres_conv,
                        verbose=True,
                        x_init=x_adj_ri, # Initialize as GRAPPA adj (without DCp)
                        x_gt=x_gt_ri,
                        return_stats=False,
                        ).detach().cpu()
                dt = time.time() - t1_wcrr_no_rot                 
            recon  = torch.abs(ri_to_complex(x_rec_ri_wcrr_no_rot))#.detach().cpu() # Its magnitude
        # Log all the metrics to weights and biases (psnr, ssim and time)
        wandb.log({"volume_idx": i, "psnr_zf": psnr(zf_recon, reference), "ssim_zf": ssim(zf_recon, reference), "psnr_grappa": psnr(grappa_recon, reference), "ssim_grappa": ssim(grappa_recon, reference), f"psnr_{method.lower()}": psnr(recon, reference), f"ssim_{method.lower()}": ssim(recon, reference), "time_zf": dt_zf, "time_grappa": dt1_grappa+dt2_grappa, f"time_{method.lower()}": dt})
        if i < 10:
            torch.save(reference, f"savings_{coil}coil/volume_{i}_gt.pt")
            torch.save(zf_recon, f"savings_{coil}coil/volume_{i}_zf.pt")
            torch.save(grappa_recon, f"savings_{coil}coil/volume_{i}_grappa.pt")
            torch.save(recon, f"savings_{coil}coil/volume_{i}_{method.lower()}.pt")
    finally:
        # 1) Break references to gpuNUFFT operators & physics (most important)
        # physics holds E_est internally, so deleting E_est alone is not enough.
        for name in ["F_raw", "E_est", "physics", "data_fidelity", "solver_tv", "solver_wv", "solver_drunet"]:
            if name in locals():
                del locals()[name]

        # 2) Delete big tensors (GPU + CPU if huge)
        for name in [
            "x", "y_np", "y_grappa", "y",
            "weights", "density",
            "x_adj_ri", "dcp_x_adj_ri",
            "x_zf", "x_gt", "x_gt_ri", "smaps",
            "x_rec_ri_tv", "x_rec_ri_wv", "x_rec_ri_drunet",
            "x_rec_ri_wcrr", "x_rec_ri_wcrr_no_rot",
            "tv_recon", "wv_recon", "drunet_recon",
            "wcrr_recon", "wcrr_no_rot_recon",
            "reference", "zf_recon", "grappa_recon",
            "new_kspace_loc", "drunet",
        ]:
            if name in locals():
                del locals()[name]

        # 3) Run Python GC (frees Python objects so CUDA tensors can be released)
        gc.collect()

        # 4) Ask CUDA to release cached blocks back to the driver (helps with fragmentation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()          # optional but helps make freeing deterministic
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
print("Reconstructions finished!")
wandb.finish()