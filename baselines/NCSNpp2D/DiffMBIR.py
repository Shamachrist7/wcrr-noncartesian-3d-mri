import torch
from tqdm import tqdm

def mag(x):
    return torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + 1e-12)

def dztdz(x):
    y = torch.zeros_like(x)
    y[..., 0]  = x[..., 0]  - x[..., 1]
    y[..., -1] = x[..., -1] - x[..., -2]
    y[..., 1:-1] = 2*x[..., 1:-1] - x[..., :-2] - x[..., 2:]
    return y

@torch.no_grad()
def denoise_volume_2d(model, x, sigma, batch_slices=16):
    # x: [1,2,H,W,D]
    B, C, H, W, D = x.shape
    xs = x[0].permute(3, 0, 1, 2).contiguous()  # [D,2,H,W]
    out = []

    for i in range(0, D, batch_slices):
        xb = xs[i:i+batch_slices]
        s = torch.full((xb.shape[0],), float(sigma), device=x.device)
        score = model(xb, s)
        out.append(xb + sigma**2 * score)

    xs = torch.cat(out, 0)
    return xs.permute(1, 2, 3, 0)[None].contiguous()  # [1,2,H,W,D]

def data_consistency_loss(x, y, physics):
    r = physics.A(x) - y
    return 0.5 * (r.abs()**2).mean()

def diffmbir(
    model, physics, y, x_init,
    sigma_max=0.015,sigma_min=0.002,num_steps=6,
    dc_steps=3,
    dc_lr=1.0,
    z_weight=1e-2,
):
    sigmas = torch.exp(torch.linspace(
        torch.log(torch.tensor(sigma_max)),
        torch.log(torch.tensor(sigma_min)),
        num_steps,
    )).tolist()
    
    model.eval()
    x = x_init.clone()
    #dc_hist = []

    for sigma in tqdm(sigmas):
        # 2D diffusion prior, slice-wise
        x = denoise_volume_2d(model, x, sigma)

        # MBIR / data-consistency refinement
        for _ in range(dc_steps):
            grad_dc = physics.A_adjoint(physics.A(x) - y)
            grad_z = dztdz(x)
            x = x - dc_lr * (grad_dc + z_weight * grad_z)

        #dc_hist.append(data_consistency_loss(x, y, physics).item())

    return x#, dc_hist