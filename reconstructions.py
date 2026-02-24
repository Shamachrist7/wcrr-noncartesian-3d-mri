import torch
import numpy as np
import cupy as cp
import wandb
import matplotlib.pyplot as plt
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.extras.smaps import cartesian_espirit, coil_compression
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim, sum_of_squares, _load_volumes, PSNR_MRI, L2_precon, normalize_kspace
from reg_architectures import WCRR3D
from deepinv.optim.prior import PnP, TVPrior, WaveletPrior
from deepinv.optim import ADMM#, HQS
from baselines.drunet.drunet_base import DRUNet
from baselines.ncpdnet import NCPDNET
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
parser.add_argument("--folder", type=str, default="savings")
parser.add_argument("--coil", type=int, default=12)
parser.add_argument("--simulation", type=int, default=1)
parser.add_argument("--compress_coil", type=float, default=-1)
parser.add_argument("--volume_id", type=int, default=-1)
parser.add_argument("--traj", type=str, default="trajectory.bin")
inp = parser.parse_args()
coil = inp.coil # 12 or 32
method = inp.method # "wcrr", "tv", "wv", "drunet", "wcrr_no_rot", "ncpdnet"
if inp.simulation:
    root = inp.root + f"/Test/{coil}coil"
else:
    root = inp.root

wandb.init(
        # Set the project where this run will be logged
        project=f"{coil}coil_results_seed_{seed}", # project name
        name=f"{method.lower()}_recons",
        config={
        "max_iter": 40,
        "noise_level": 2e-3,
        })
volume_id = inp.volume_id
start_dir = inp.folder
os.makedirs(f"{start_dir}_{coil}coil", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "gpunufft"
scaler = 1e-6 # data normalizer
noise_level = 2e-3
max_iter = 30 # Maximum number of iterations
thres_conv = 1e-3 # convergence threshold

if inp.simulation:
    volumes = sorted([fn for fn in os.listdir(root) if fn.endswith(".h5")])[:15]
    # Load trajectory and get the k-space locations
    traj, traj_params = read_trajectory(inp.traj, dwell_time=0.01/2)
    traj = traj.copy()
    traj[traj < -0.5] = -0.5
    traj[traj > 0.5] = 0.5
    dim = traj_params["dimension"]
    kspace_loc = traj.reshape(-1, dim)
else:
    # your 50 multi-coil volumes (12 or 32 coils)
    volumes = sorted([fn for fn in os.listdir(root) if fn.endswith(".dat")])

volumes = [volumes[inp.volume_id]] if inp.volume_id != -1 else volumes
print(volumes)

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
##### NC-PDNet #####
if method.lower()=="ncpdnet":
    class DummyNUFFT:
        def op(self, x):
            return x  
        def adj_op(self, kspace, dcomp=None):
            return torch.zeros_like(kspace)
    dummy_nufft = DummyNUFFT()
    ncpdnet = NCPDNET(nufft_op=dummy_nufft, image_net_type="ImageNetUnet", base_filters=16, num_stages=3, n_primal=2, n_iter=6, activation="silu", dim=3, complex_recon=True, normalize_input=True)
    weights = torch.load("./weights/ncpdnet/ncpdnet_weights.pth", map_location=device, weights_only=True)
    ncpdnet.load_state_dict(weights["state_dict"])

# reconstructions

for i, volume in enumerate(volumes):
    if not inp.simulation:
        from mrinufft.extras.cartesian import fft, ifft
        import cupy as cp
        y_np, data_header = _load_volumes(os.path.join(root, volume))
        if inp.compress_coil > 0:
            y_np, V = coil_compression(y_np, 0.9)
        y_np = y_np * 1e3 #/ 0.9 # Scale real data to same scale as simulations
        C, *XYZ = data_header['ref'].shape
        if inp.compress_coil > 0:
            x = torch.tensor(ifft((cp.asarray(V) @ fft(cp.asarray(data_header['ref'])).reshape(C, -1)).reshape(V.shape[0], *XYZ)), dtype=torch.complex64).cpu()
        else:
            x = torch.tensor(data_header['ref'], dtype=torch.complex64)
        traj, traj_params = read_trajectory(os.path.join(root, "traj", data_header['trajectory_name']), dwell_time=0.01/data_header['oversampling_factor'])
        caipi_delta = 1
        y_np = add_phase_to_kspace_with_shifts(
            y_np, 
            traj.reshape(-1, traj_params["dimension"]),
            normalized_shifts=(
                np.array(data_header["shifts"])
                / np.array(traj_params["FOV"])
                * np.array(traj_params["img_size"])
                / 1000
            ),
        )
        traj[traj < -0.5] = -0.5
        traj[traj > 0.5] = 0.5
        kspace_loc = traj.reshape(-1, traj_params["dimension"])
        # @Shama you can use ESPIRiT like this to get the smaps from the acs data in data_header['acs'], we started doing external ACS acquisitions for all current acquisitions
        C, *XYZ = data_header['acs'].shape
        if inp.compress_coil > 0:
            smaps = cartesian_espirit(cp.array((V @ data_header['acs'].reshape(C, -1)).reshape(V.shape[0], *XYZ), dtype=cp.complex64), traj_params['img_size'], decim=4, crop=0).get()
        else:
            smaps = cartesian_espirit(cp.asarray(data_header['acs'], dtype=cp.complex64), traj_params['img_size'], decim=4, crop=0).get()
        # Clean up cupy memory
        cp._default_memory_pool.free_all_blocks()
        coils = y_np.shape[0]
        F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True)
    else:
        caipi_delta = 0
        # Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
        x = torch.from_numpy(scaler * np.moveaxis(_load_volumes(os.path.join(root, volume)),-1, 0)) #[coils,H,W,D] and complex dtype
        coils = x.shape[0] # number of coils in the volume
        # Forward NUFFT that takes coil images -> k-space
        print(f"Simulation of the undersampled measurement {i+1}!")
        print("Start ... ")
        F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True)
        y_np = F_raw.op(x) # simulates the undersampled kspace volume y. The zero-filled recon comes from it
        y_np = y_np + noise_level * torch.randn_like(y_np)
        print("Succesfully simulated!")
    # GRAPPA reconstruct the center of k-space, basis for our regularizers
    grappa_recon_done = True
    t1_grappa = time.time()
    try:
        new_kspace_loc, y_grappa = do_grappa_and_append_data(kspace_loc, y_np, traj_params, af=(2, 2), acs=None if inp.simulation else data_header['acs'], caipi_delta=caipi_delta)
    except:
        grappa_recon_done = False
        print("GRAPPA reconstruction failed, trying SENSE")
        new_kspace_loc, y_grappa = kspace_loc, y_np
    dt1_grappa = time.time() - t1_grappa
    # Build reconstruction operator that ESTIMATES smaps from y_grappa
    print(f"Operator definition, DCp weights and smaps estimation from measurement {i+1}!")
    print("Start ... ")
    density = get_density("pipe", new_kspace_loc, traj_params['img_size'], backend=backend, max_iter=10).astype(np.float32)
    weights = torch.from_numpy(density).to(device)
    data_fidelity = L2_precon(weights) # custom data fidelity
    
    E_est = get_operator(backend)(
        new_kspace_loc,
        x.shape[1:],
        n_coils=coils,
        smaps={"name": "low_frequency", "kspace_data": y_grappa} if inp.simulation else smaps,
        density= False,
        use_gpu_direct=True,
    )
    physics = MRINUFFTPhysicsRI(E_est)
    print("Succesfull!")
    # Reference/Ground Truth (Adjoint coil combination)    
    smaps = torch.from_numpy(E_est.smaps)
    y = torch.from_numpy(y_grappa).to(device) # ACS reconstructed with grappa (In the k-space)
    if grappa_recon_done:
        x_adj_ri = physics.A_adjoint(y) # Grappa adjoint without DCp (Initialization for all our iterative solvers)
    else:
         # If GRAPPA failed, we fall back to the pinv of the estimated NUFFT operator
        x_adj_ri = physics.A_dagger(y).to(device)
    # Reference/Ground Truth (Adjoint coil combination)    smaps = torch.from_numpy(E_est.smaps)
    x_gt = torch.sum(torch.conj(smaps) * x, axis=0)
    x_gt_ri = complex_to_ri(x_gt).to(device) # In the RI space
    reference = torch.abs(ri_to_complex(x_gt_ri)).detach().cpu()
    # Grappa + DCp recon
    t2_grappa = time.time()
    if grappa_recon_done:
        dcp_x_adj_ri = physics.A_adjoint(weights * y).detach().cpu()
    else:
        dcp_x_adj_ri = x_adj_ri
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
            x_rec_ri_tv = solver_tv(y, physics, init=(dcp_x_adj_ri, dcp_x_adj_ri), compute_metrics=False).detach().cpu()
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
            x_rec_ri_wv = solver_wv(y, physics, x_gt=x_gt_ri, init=(dcp_x_adj_ri, dcp_x_adj_ri), compute_metrics=False).detach().cpu()
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
            x_rec_ri_drunet = solver_drunet(y, physics, x_gt=x_gt_ri, init=(dcp_x_adj_ri, dcp_x_adj_ri), compute_metrics=False).detach().cpu()
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
                    x_init=dcp_x_adj_ri, # Initialize as GRAPPA adj (without DCp)
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
                    x_init=dcp_x_adj_ri, # Initialize as GRAPPA adj (without DCp)
                    x_gt=x_gt_ri,
                    return_stats=False,
                    ).detach().cpu()
            dt = time.time() - t1_wcrr_no_rot                 
        recon  = torch.abs(ri_to_complex(x_rec_ri_wcrr_no_rot))#.detach().cpu() # Its magnitude
                # NC-PDnet recon
    if method.lower()=="ncpdnet":
        # NC-PDNet is trained with Density compensation
        yn, norm_fact = normalize_kspace(y_grappa, E_est.samples) #normalize wrt energy of central region
        y = torch.from_numpy(yn).to(device)
        if grappa_recon_done:
            smaps_new = {"name": "low_frequency", "kspace_data": y_grappa}
        else:
            from mrinufft.extras.smaps import _crop_or_pad
            smaps_new = ifft(_crop_or_pad(cp.asarray(data_header['acs'], dtype=cp.complex64), (coils, *E_est.shape))).get()
            smaps_new = smaps_new / (np.linalg.norm(smaps_new, axis=0) + 1e-10)
        ncpdnet.update_nufft_op(
            get_operator(backend)(
                E_est.samples, 
                E_est.shape, 
                n_coils=coils, 
                density=True,
                smaps=smaps_new,
                use_gpu_direct=True,
                squeeze_dims=False, #preserve batch dim 
                )
            )
        ncpdnet.to(device).eval()
        with torch.no_grad():
            t1_ncpdnet = time.time()
            recon = ncpdnet(y.unsqueeze(0)).squeeze().detach().cpu() 
            recon = torch.abs(recon) * norm_fact
            dt = time.time() - t1_ncpdnet
    # Log all the metrics to weights and biases (psnr, ssim and time)
    wandb.log({"volume_idx": i if volume_id == -1 else volume_id, "psnr_grappa": psnr(grappa_recon, reference), "ssim_grappa": ssim(grappa_recon, reference), f"psnr_{method.lower()}": psnr(recon, reference), f"ssim_{method.lower()}": ssim(recon, reference), "time_grappa": dt1_grappa+dt2_grappa, f"time_{method.lower()}": dt})
    if i < 10:
        torch.save(reference, f"{start_dir}_{coil}coil/volume_{i if volume_id == -1 else volume_id}_gt.pt")
        torch.save(grappa_recon, f"{start_dir}_{coil}coil/volume_{i if volume_id == -1 else volume_id}_grappa.pt")
        torch.save(recon, f"{start_dir}_{coil}coil/volume_{i if volume_id == -1 else volume_id}_{method.lower()}.pt")
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
        "x_zf", "x_gt", "_gt_ri", "smaps",
        "x_rec_ri_tv", "x_rec_ri_wv", "x_rec_ri_drunet",
        "x_rec_ri_wcrr", "x_rec_ri_wcrr_no_rot",
        "tv_recon", "wv_recon", "drunet_recon",
        "wcrr_recon", "wcrr_no_rot_recon",
        "reference", "grappa_recon",
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
