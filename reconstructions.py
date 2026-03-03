import torch
import numpy as np
import cupy as cp
import wandb
from mrinufft import get_operator
from mrinufft.io import read_trajectory
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.extras.smaps import cartesian_espirit, coil_compression
from baselines.grappa_reconstruction import do_grappa_and_append_data
from utils import MRINUFFTPhysicsRI, ri_to_complex, complex_to_ri, psnr, ssim, _load_volumes, L2_precon, normalize_kspace, get_acs_locations, get_DPIR_params
from reg_architectures import WCRR3D
from deepinv.optim.prior import PnP, WaveletPrior
from deepinv.optim import HQS, FISTA
from baselines.drunet.drunet_base import DRUNet
from baselines.ncpdnet import NCPDNET
from baselines.TV import PDHG_TV
from mrinufft.extras.cartesian import fft, ifft
from mrinufft.extras.smaps import get_smaps
from evaluation.nmAPG3d_evaluation import reconstruct_nmAPG
import deepinv as dinv
import gc
import os
import time
import argparse
import warnings
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

torch.random.manual_seed(0)  # make results deterministic

parser = argparse.ArgumentParser(description="reconstructions")
parser.add_argument("--root", type=str, default='../../../../../../../../LOCAL/mri_data')
parser.add_argument("--method", type=str, default="WCRR")
parser.add_argument("--folder", type=str, default="savings")
parser.add_argument("--coil", type=int, default=12)
parser.add_argument("--simulation", type=int, default=1)
parser.add_argument("--compress_coil", type=float, default=-1)
parser.add_argument("--volume_id", type=int, default=-1)
parser.add_argument("--traj", type=str, default="gs.bin") # "gs.bin" or "caipi.bin"
parser.add_argument("--smaps_on_gpu", type=bool, default=True) # Whether to compute the smaps on GPU (with cupy) or CPU (with numpy). If True, make sure to have enough GPU memory, especially for the 32-coil data. If False, it will be much slower but can be run on CPU if GPU memory is not sufficient.
parser.add_argument("--smaps_precomputation", type=bool, default=False) # Whether to only precompute the smaps and save them, without doing the reconstructions. This can be useful to avoid recomputing the smaps every time you want to test a new method or hyperparameters. If True, make sure to set the correct root directory and run this once before running the reconstructions with smaps_precomputation=False and loading the smaps from disk in utils.py.
parser.add_argument("--precomputed_smaps_available", type=bool, default=True) # Whether the precomputed smaps are already available on disk. If True, make sure to set the correct smaps_dir in utils.py to load them.
parser.add_argument("--init", type=str, default="grappa") # "grappa" or "sense"

inp = parser.parse_args()
coil = inp.coil # 12 or 32 or different for prospective users
method = inp.method # "wcrr", "tv", "wv", "drunet", "wcrr_no_rot", "ncpdnet"
if inp.simulation:
    root = inp.root + f"/Test/{coil}coil"
else:
    root = inp.root
    
if inp.smaps_precomputation or inp.precomputed_smaps_available:
    smaps_dir = root + f"/first_15_smaps/{coil}coil_{inp.traj[:-4]}/"
    os.makedirs(smaps_dir, exist_ok=True)
 
if not inp.smaps_precomputation:    
    wandb.init(
            # Set the project where this run will be logged
            project=f"{coil}coil_results_{inp.traj[:-4]}", # project name
            name=f"{method.lower()}_recons",
            config={
            "max_iter": 200,
            "noise_level": 2e-3,
            })
volume_id = inp.volume_id
start_dir = inp.folder
os.makedirs(f"{start_dir}_{coil}coil_{inp.traj[:-4]}", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = "cufinufft"
scaler = 1e-6 # data normalizer
noise_level = 2e-3
max_iter = 200 # Maximum number of iterations
data_fidelity = L2_precon(weights=torch.tensor(1.0))

if inp.simulation:
    volumes = sorted([fn for fn in os.listdir(root) if fn.endswith(".h5")])[:15]
    # Load trajectory and get the k-space locations
    traj, traj_params = read_trajectory(inp.traj, dwell_time=0.01/2)
    traj = traj.copy()
    traj[traj < -0.5] = -0.5
    traj[traj > 0.5] = 0.5
    dim = traj_params["dimension"]
    kspace_loc = traj.reshape(-1, dim)
    if inp.traj.lower() == "caipi.bin":
        acs_loc = get_acs_locations(img_size=traj_params['img_size'])
        kspace_loc = np.vstack([kspace_loc, acs_loc])
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
    sigma_denoiser, stepsize, num_iter = get_DPIR_params(num_iter=1, sigma_init=0.01, lmbd=2e-3, device=device)
    solver_drunet = HQS(
            prior=prior_drunet,
            data_fidelity=data_fidelity,
            stepsize=stepsize,
            sigma_denoiser=sigma_denoiser,
            max_iter=num_iter,
            verbose=True,
            show_progress_bar = True,
        )
##### l1-wavelet prior and hyperparameters #####
if method.lower()=="wv":
    prior_wv = WaveletPrior(level=4, wv="db4", p=1, wvdim=3)
    lmbd = 3e-3
    tol = 5e-3 #1e-3
##### TV prior and hyperparameters #####
if method.lower()=="tv":
    lmbd = 0.3
    tol = 1e-3
##### (Rotation invariant) WCRR prior and hyperparameters #####
if method.lower()=="wcrr":
    WCRR = WCRR3D(weak_convexity=1.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=True).to(device)
    WCRR.load_state_dict(torch.load("weights/bilevel_Denoising/WCRR_bilevel_IFT_ckpt_100.pt", weights_only=True, map_location=device))
    WCRR.eval()
    lmbd = 5e-3
    sigma = 0.06
    sigma = torch.tensor([sigma], device=device)
    tol = 1e-2
##### (Not rotation invariant) WCRR_no_rot prior and hyperparameters #####
if method.lower()=="wcrr_no_rot":
    WCRR_no_rot = WCRR3D(weak_convexity=1.0, nb_channels=[2,4,8,32], filter_sizes=[3, 3, 3], rotations=False).to(device)
    WCRR_no_rot.load_state_dict(torch.load("weights/bilevel_Denoising/WCRR_no_rotations_bilevel_IFT_ckpt_100.pt", weights_only=True, map_location=device))
    WCRR_no_rot.eval()
    lmbd = 5e-3  # Not definitive
    sigma = 0.06 # Not definitive
    sigma = torch.tensor([sigma], device=device)
    tol = 1e-2
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
        y, data_header = _load_volumes(os.path.join(root, volume))
        if inp.compress_coil > 0:
            y, V = coil_compression(y, 0.9)
        y = y  * 1e3 #/ 0.9 # Scale real data to same scale as simulations
        C, *XYZ = data_header['ref'].shape
        if inp.compress_coil > 0:
            x = torch.tensor(ifft((cp.asarray(V) @ fft(cp.asarray(data_header['ref'])).reshape(C, -1)).reshape(V.shape[0], *XYZ)), dtype=torch.complex64).cpu()
        else:
            x = torch.tensor(data_header['ref'], dtype=torch.complex64)
        traj, traj_params = read_trajectory(os.path.join(root, "traj", data_header['trajectory_name']), dwell_time=0.01/data_header['oversampling_factor'])
        caipi_delta = 1
        y = add_phase_to_kspace_with_shifts(
            y, 
            traj.reshape(-1, traj_params["dimension"]),
            normalized_shifts=(
                np.array(data_header["shifts"])
                / np.array(traj_params["FOV"])
                * np.array(traj_params["img_size"])
                / 1000
            ),
        )
        y = torch.from_numpy(y)
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
        coils = y.shape[0]
        F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True, squeeze_dims=True) # NUFFT operator that simulates the measurement (takes coil images as input, outputs k-space)
    else:
        caipi_delta = 0
        # Calgary volumes are under the format [D,H,W,coils], and we convert to [coils,H,W,D] so that NUFFT works
        x = torch.from_numpy(scaler * np.moveaxis(_load_volumes(os.path.join(root, volume)),-1, 0)) #[coils,H,W,D] and complex dtype
        coils = x.shape[0] # number of coils in the volume
        # Forward NUFFT that takes coil images -> k-space
        print(f"Simulation of the undersampled measurement {i}!")
        print("Start ... ")
        F_raw = get_operator(backend)(kspace_loc, x.shape[1:], n_coils=coils, density=True, squeeze_dims=True) # NUFFT operator that simulates the measurement (takes coil images as input, outputs k-space)
        y = F_raw.op(x) # simulates the undersampled kspace volume y. The zero-filled recon comes from it
        y = y + noise_level * torch.randn_like(y)
        print("Succesfully simulated!")
        
        if not inp.precomputed_smaps_available:
            print(f"smaps estimation from measurement {i}!")
            print("Start ... ")
            if inp.smaps_on_gpu==True:
                smaps = get_smaps("espirit")(
                    kspace_loc,
                    x.shape[1:],
                    kspace_data=cp.asarray(y),
                    density=F_raw.density,
                    backend=backend,
                    decim=4,
                ).get()
            else:
                smaps = get_smaps("espirit")(
                    kspace_loc,
                    x.shape[1:],
                    kspace_data=y.numpy(),
                    density=F_raw.density,
                    backend=backend,
                    decim=4,
                )
            # Clean up cupy memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            cp._default_memory_pool.free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            if inp.smaps_precomputation:
                np.save(smaps_dir + f"volume_{i if volume_id == -1 else volume_id}_smaps.npy", smaps)
                continue # Skip the reconstructions if we are only precomputing the smaps
        else:
            print(f"Loading precomputed smaps for volume {i}!")
            print("Start ... ")
            smaps = np.load(smaps_dir + f"volume_{i if volume_id == -1 else volume_id}_smaps.npy")
    
    # Operator definition
    E_est = get_operator(backend)(
        kspace_loc,
        x.shape[1:],
        n_coils=coils,
        smaps=smaps,
        squeeze_dims=True,
    )
    physics = MRINUFFTPhysicsRI(E_est)
    print("Succesfull!")
    # compute initialization (GRAPPA or SENSE)
    if inp.init.lower() == "grappa":
        try:
            t1_grappa = time.time()
            new_kspace_loc, y_grappa = do_grappa_and_append_data(kspace_loc, y, traj_params, af=(2, 2), acs=None if inp.simulation else data_header['acs'], caipi_delta=caipi_delta)
            y_grappa = torch.from_numpy(y_grappa) # ACS reconstructed with grappa (In the k-space)
            # Grappa + DCp recon
            nufft_grappa = get_operator(backend)(new_kspace_loc, x.shape[1:], n_coils=coils, smaps=smaps, density=True, squeeze_dims=True)
            grappa_ri = complex_to_ri(nufft_grappa.adj_op(y_grappa))
            dt_init = time.time() - t1_grappa
            del nufft_grappa # free memory
        except:
            print("GRAPPA reconstruction failed! Please, use SENSE as initialization for this trajectory!")
            break
    elif inp.init.lower() == "sense":
        t1_sense = time.time()
        sense_ri = physics.A_dagger(y)
        dt_init = time.time() - t1_sense
    
    init = sense_ri if inp.init.lower() == "sense" else grappa_ri # Initialization for all our iterative methods
    init_recon = torch.abs(ri_to_complex(init))
    # Reference/Ground Truth (Adjoint coil combination)
    smaps = torch.from_numpy(smaps)
    x_gt = torch.sum(torch.conj(smaps) * x, axis=0)
    x_gt_ri = complex_to_ri(x_gt) # In the RI space
    reference = torch.abs(ri_to_complex(x_gt_ri))
    
    # Clean memory before proper reconstructions
    del F_raw, E_est,
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    cp._default_memory_pool.free_all_blocks()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    # TV recon
    if method.lower()=="tv":
        op_norm = physics.compute_sqnorm(torch.randn_like(x_gt_ri, device=device), max_iter=20).item()
        solver_tv = PDHG_TV(
            lambda_reg=lmbd,
            max_iter=max_iter,
            lipschitz= op_norm,
            data_fidelity=data_fidelity,
            stopping_criterion=tol,
        )
        with torch.no_grad():
            t1_tv = time.time()
            x_rec_ri_tv = solver_tv(y.to(device), physics, init=init.to(device), compute_metrics=False).detach().cpu()
            dt = time.time() - t1_tv
        recon  = torch.abs(ri_to_complex(x_rec_ri_tv)) # Its magnitude
    # l1-wavelet recon
    if method.lower()=="wv":
        op_norm = physics.compute_sqnorm(torch.randn_like(x_gt_ri, device=device), max_iter=20).item()
        solver_wv = FISTA(
            prior=prior_wv,
            data_fidelity=data_fidelity,
            stepsize=1.0 / op_norm,
            lambda_reg=lmbd,
            max_iter=max_iter,
            thres_conv=tol,
            verbose=True,
            early_stop=True,
            show_progress_bar=True,
        )
        with torch.no_grad():
            t1_wv = time.time()
            x_rec_ri_wv = solver_wv(y.to(device), physics, init=(init.to(device), init.to(device)), compute_metrics=False).detach().cpu()
            dt = time.time() - t1_wv
        recon  = torch.abs(ri_to_complex(x_rec_ri_wv)) # Its magnitude
    # PnP-DRUNet recon
    if method.lower()=="drunet":
        with torch.no_grad():
            t1_drunet = time.time()
            x_rec_ri_drunet = solver_drunet(y.to(device), physics, init=init.to(device), compute_metrics=False).detach().cpu()
            dt = time.time() - t1_drunet
        recon  = torch.abs(ri_to_complex(x_rec_ri_drunet)) # Its magnitude
    # WCRR recon
    if method.lower()=="wcrr":
        with torch.no_grad():
            t1_wcrr = time.time()
            x_rec_ri_wcrr = reconstruct_nmAPG(
                    sigma,
                    y.to(device),
                    physics,
                    data_fidelity,
                    WCRR,
                    lmbd,
                    1e-1, # Stepsize_nmAPG (can be anything)
                    max_iter,
                    tol,
                    verbose=True,
                    x_init=init.to(device),
                    return_stats=False,
                    ).detach().cpu()
            dt = time.time() - t1_wcrr                 
        recon  = torch.abs(ri_to_complex(x_rec_ri_wcrr)) # Its magnitude
    # WCRR_no_rot recon
    if method.lower()=="wcrr_no_rot":
        with torch.no_grad():
            t1_wcrr_no_rot = time.time()
            x_rec_ri_wcrr_no_rot = reconstruct_nmAPG(
                    sigma,
                    y.to(device),
                    physics,
                    data_fidelity,
                    WCRR_no_rot,
                    lmbd,
                    1e-1, # Stepsize_nmAPG (can be anything)
                    max_iter,
                    tol,
                    verbose=True,
                    x_init=init.to(device),
                    return_stats=False,
                    ).detach().cpu()
            dt = time.time() - t1_wcrr_no_rot                 
        recon  = torch.abs(ri_to_complex(x_rec_ri_wcrr_no_rot)) # Its magnitude
    # NC-PDnet recon
    if method.lower()=="ncpdnet":
        # NC-PDNet is trained with Density compensation
        yn, norm_fact = normalize_kspace(y_grappa, E_est.samples) #normalize wrt energy of central region
        y = torch.from_numpy(yn).to(device)
        x = ri_to_complex(x_adj_ri).to(device)[None, None] / norm_fact
        E_est.squeeze_dims = False # preserve batch dim for ncpdnet
        ncpdnet.update_nufft_op(E_est)
        ncpdnet.to(device).eval()
        with torch.no_grad():
            t1_ncpdnet = time.time()
            recon = ncpdnet(y.unsqueeze(0), x).squeeze().detach().cpu() 
            recon = torch.abs(recon) * norm_fact
            dt = time.time() - t1_ncpdnet
    # Log all the metrics to weights and biases (psnr, ssim and time)
    wandb.log({"volume_idx": i if volume_id == -1 else volume_id, f"psnr_{inp.init.lower()}": psnr(init_recon, reference), f"ssim_{inp.init.lower()}": ssim(init_recon, reference), f"psnr_{method.lower()}": psnr(recon, reference), f"ssim_{method.lower()}": ssim(recon, reference), f"time_{inp.init.lower()}": dt_init, f"time_{method.lower()}": dt})
    if i < 10:
        torch.save(reference, f"{start_dir}_{coil}coil_{inp.traj[:-4]}/volume_{i if volume_id == -1 else volume_id}_gt.pt")
        torch.save(init_recon, f"{start_dir}_{coil}coil_{inp.traj[:-4]}/volume_{i if volume_id == -1 else volume_id}_{inp.init.lower()}.pt")
        torch.save(recon, f"{start_dir}_{coil}coil_{inp.traj[:-4]}/volume_{i if volume_id == -1 else volume_id}_{method.lower()}.pt")
    # 1) Break references to gpuNUFFT operators & physics (most important)
    # physics holds E_est internally, so deleting E_est alone is not enough.



    for name in ["F_raw", "E_est", "physics", "solver_tv", "solver_wv"]:
        if name in locals():
            del locals()[name]
    # 2) Delete big tensors (GPU + CPU if huge)
    for name in [
        "x", "y_grappa", "y",
        "x_adj_ri", "dcp_x_adj_ri",
        "x_zf", "x_gt", "x_gt_ri", "smaps",
        "x_rec_ri_tv", "x_rec_ri_wv", "x_rec_ri_drunet",
        "x_rec_ri_wcrr", "x_rec_ri_wcrr_no_rot",
        "tv_recon", "wv_recon", "drunet_recon",
        "wcrr_recon", "wcrr_no_rot_recon",
        "reference", "grappa_recon", "sense_recon",
        "new_kspace_loc", "reference", "init_recon", "recon",
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
