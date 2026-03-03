from reg_architectures import WCRR3D
from deepinv.physics import Denoising, GaussianNoise
from deepinv.optim import L2
from deepinv.loss.metric import PSNR, SSIM
from evaluation import reconstruct_nmAPG

import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_processing import load_data
from torch.utils.data import DataLoader
import torchvision
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.random.manual_seed(0)  # make results deterministic

def free_gpu():
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
free_gpu()

psnr = PSNR()
ssim = SSIM()

noise_level =  0.05
physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
data_fidelity = L2(sigma=1.0)

# Parameters for the Nesterov Algorithm, might also be problem dependent...
NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)


train = False
scale_fact = 1e-6
patch_size = 0
batch_size = 1
transform = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(train_patch_size),
        #RandomCrop3D(train_patch_size),
        # function for rescaling
        torchvision.transforms.Lambda(lambda x: x * scale_fact),
])

root = '../../../../../../../../LOCAL/mri_data'

#dataset = Calgary3D(root=root, transform=transform)

#dataloader = DataLoader(dataset,
#                        batch_size=_batch_size,
#                        shuffle=False,
#                        drop_last=True,
#                        pin_memory=True if torch.cuda.is_available() else False,
#                        num_workers=4)

dataloader = load_data(root, patch_size, batch_size, device=device, train=train, scale_fact=1e-6, crop3D=False)

reg_name = "WCRR_5by5by5_32" # "WCRR_no_rot", "WCRR" or "WCRR_3by3by3_64" or "WCRR_5by5by5_32"
eval_method = "prox" # "prox" or "score"

# Define regularizer
if reg_name == "WCRR_no_rot":
    regularizer = WCRR3D(
        weak_convexity=1.0, 
        nb_channels=[2,4,8,32],
        filter_sizes=[3, 3, 3],
        rotations=False,
    ).to(device)
    if eval_method == "prox":
        pretrained = "weights/bilevel_Denoising/WCRR_no_rotations_bilevel_IFT_ckpt_100.pt"
    elif eval_method == "score":
        pretrained = "weights/score_for_Denoising/WCRR_no_rotations_score_training_ckpt_1000.pt"
    else:
        raise ValueError("Unknown evaluation method!")
        
elif reg_name == "WCRR":
    regularizer = WCRR3D(
        weak_convexity=1.0, 
        nb_channels=[2,4,8,32],
        filter_sizes=[3, 3, 3],
        rotations=True,
        device=device,
    ).to(device)
    if eval_method == "prox":
        pretrained = "weights/bilevel_Denoising/WCRR_bilevel_IFT_ckpt_100.pt"
    elif eval_method == "score":
        pretrained = "weights/score_for_Denoising/WCRR_score_training_ckpt_1000.pt"
    else:
        raise ValueError("Unknown evaluation method!")
    
elif reg_name == "WCRR_3by3by3_64":
    regularizer = WCRR3D(
        weak_convexity=1.0, 
        nb_channels=[2,4,8,64],
        filter_sizes=[3, 3, 3],
        rotations=True,
        device=device,
    ).to(device)
    if eval_method == "prox":
        pretrained = "weights/bilevel_Denoising/WCRR_3by3by3_64_bilevel_IFT_ckpt_100.pt"
    elif eval_method == "score":
        pretrained = "weights/score_for_Denoising/WCRR_3by3by3_64_score_training_ckpt_1000.pt"
    else:
        raise ValueError("Unknown evaluation method!")
    
elif reg_name == "WCRR_5by5by5_32":
    regularizer = WCRR3D(
        weak_convexity=1.0, 
        nb_channels=[2,4,8,32],
        filter_sizes=[5, 5, 5],
        rotations=True,
        device=device,
    ).to(device)
    if eval_method == "prox":
        pretrained = "weights/bilevel_Denoising/WCRR_5by5by5_32_bilevel_IFT_ckpt_100.pt"
    elif eval_method == "score":
        pretrained = "weights/score_for_Denoising/WCRR_5by5by5_32_score_training_ckpt_1000.pt"
    else:
        raise ValueError("Unknown evaluation method!")    
                  
else:
    raise ValueError("Unknown model!")
        

#regularizer = ParameterLearningWrapper(reg, device=device)
lmbd = 1.0
regularizer.load_state_dict(torch.load(pretrained, weights_only=True, map_location=device))
regularizer.eval()

params = 0
for p in regularizer.parameters():
    params += p.numel()
print(params)

"""reg4score = ParameterLearningWrapper(reg, device=device)
lmbd = 1.0
reg4score.load_state_dict(torch.load("weights/score_for_Denoising/WCRR_score_training_for_Denoising.pt", weights_only=True, map_location=device))
reg4score.eval()"""
sigma = torch.tensor([noise_level], device=device)
with torch.no_grad():
    torch.cuda.reset_peak_memory_stats()  # clears the high‐water mark
    t1 = time.time()
    avg_psnr = 0.0
    avg_ssim = 0.0
    for i, x in enumerate(tqdm(dataloader, desc="Inference")):
        x = x.to(device)
        y = physics(x)
        if eval_method == "prox":
            x_recon = reconstruct_nmAPG(
                        sigma,
                        y,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd,
                        NAG_step_size,
                        NAG_max_iter,
                        NAG_tol,
                        verbose=True,
                        x_init=y,#-reg4score.grad(y,sigma),
                        )
        elif eval_method == "score":
            x_recon = y - lmbd * regularizer.grad(y, sigma)
        else:
            raise ValueError("Unknown evaluation method!")
        #if i == 0:
        #    torch.save(x_recon,f"savings/Vol_{i}_{reg_name}_{eval_method}.pt")
        print(f"Volume {i+1}, PSNR = {psnr(x, x_recon).item()} dB")
        #break
        avg_psnr += psnr(x, x_recon).item()
        avg_ssim += ssim(x, x_recon).item()
    avg_psnr = avg_psnr / len(dataloader)
    avg_ssim = avg_ssim / len(dataloader)
    t2 = time.time()
    # Right after inference, query the peak:
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_GB   = peak_bytes / (1024**3) # bytes to Giga-bytes
    print(f"Peak GPU memory usage: {peak_GB:.2f} GB")
    print(f"Inference time: {(t2-t1)/60:.2f} minutes")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.2f}")
