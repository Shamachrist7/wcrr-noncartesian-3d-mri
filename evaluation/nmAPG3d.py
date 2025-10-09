import torch
import numpy as np
from typing import Callable, Tuple
import inspect
from torch.amp import autocast

# Implements Algorithm 4 (non-monotone APG) from:
# Huan Li, Zhouchen Lin
# Accelerated Proximal Gradient Methods for Nonconvex Programming (NeurIPS 2015)

def nmAPG(
    sigma: torch.Tensor,
    x0: torch.Tensor,                    # [B, C, D, H, W]
    y: torch.Tensor,                     # measurement or target, shape [B, ...]
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    f_and_nabla: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    max_iter: int = 200,
    L_init: float = 1.0,
    tol: float = 1e-4,
    rho: float = 0.9,
    delta: float = 0.1,
    eta: float = 0.8,
    verbose: bool = False,
    use_amp: bool = False,                # use mixed precision to save memory
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Non-monotone APG solver for 3D volumes x of shape [B,C,D,H,W].
    Returns optimized x, final L, and iteration count.
    Mixed precision (autocast) around f and f_and_nabla to reduce GPU usage.
    """
    # Optionally cast to half precision
    if use_amp:
        x0 = x0.half()
        y = y.half()

    # Initialize variables
    x = x0.clone()
    x_old = x.clone()
    z = x0.clone()
    t = 1.0
    t_old = 0.0
    q = 1.0
    # initial energy
    with autocast('cuda', enabled=use_amp):
        c = f(x, y, sigma)

    # Lipschitz per sample
    L = torch.full((x.shape[0], 1, 1, 1, 1), L_init, device=x.device, dtype=x.dtype)
    L_old = L.clone()

    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(0, x.shape[0], device=x.device)

    grad = torch.zeros_like(x)
    x_bar = torch.zeros_like(x)
    x_bar_old = x_bar.clone()
    grad_old = grad.clone()

    # Main loop
    for i in range(max_iter):
        assert not torch.any(
            torch.isnan(x)
        ), "Numerical errors! Some values became NaN!"
        # Extrapolation
        x_bar[idx] = (
            x[idx]
            + (t_old / t) * (z[idx] - x[idx])
            + ((t_old - 1) / t) * (x[idx] - x_old[idx])
        )
        x_old.copy_(x)

        # Compute energy and gradient
        with torch.no_grad(), autocast('cuda', enabled=use_amp):
            energy, grad_vals = f_and_nabla(x_bar[idx], y[idx], sigma[idx])
        grad[idx] = grad_vals

        # # Lipschitz Update (Barzilai-Borwein style step)
        if i > 0:
            dxg = grad[idx] - grad_old[idx]
            rr = (dxg * dxg).sum((1,2,3,4), keepdim=True)
            L[idx] = torch.clip(
                rr
                / (dxg * (x_bar[idx] - x_bar_old[idx]))
                .sum((1, 2, 3, 4), keepdim=True)
                .clip(min=1e-6, max=None),  # alpha_y = <s,r>/<r,r> in paper, Eq 150
                min=1.0,
                max=1e6,
            )  # clips for stability --> on a long term we can adjust min-clip based on the spectral norm of physics.A

        # Line search on z
        for _ in range(150):
            #print(f" Is there NaN in Grad? {torch.any(torch.isnan(grad))}")
            z[idx] = x_bar[idx] - grad[idx] / (L[idx] + 1e-6) # +1e-6 for numerical stability
            dxx = z[idx] - x_bar[idx]
            bound = torch.max(
                energy[:, None, None, None, None],
                c[idx, None, None, None, None]
            ) - delta * (dxx * dxx).sum((1,2,3,4), keepdim=True)
            with autocast('cuda', enabled=use_amp):
                energy_new = f(z[idx], y[idx], sigma[idx])
            if torch.all(energy_new <= bound.view(-1)):
                break
            mask = energy_new[:, None, None, None, None] <= bound
            L[idx] = torch.where(mask, L[idx], L[idx] / rho)

        # Non-monotone correction
        idx2 = ((energy_new[:] >= (c[idx] - delta * (dxx * dxx).sum((1,2,3,4)))).nonzero().view(-1))
        if idx2.nelement() > 0:
            sel = idx[idx2]
            with torch.no_grad(), autocast('cuda', enabled=use_amp):
                gradx = nabla(x[sel], y[sel], sigma[sel])
            if i > 0:
                dxg2 = gradx - grad_old[sel]
                rr2 = (dxg2 * dxg2).sum((1,2,3,4), keepdim=True)
                L[sel] = torch.clip(
                    rr2
                    / (dxg2 * (x[sel] - x_bar_old[sel]))
                    .sum((1, 2, 3, 4), keepdim=True)
                    .clip(min=1e-6, max=None),
                    min=1.0,
                    max=1e6,
                )
            L_old.copy_(L)
            # Line search on v
            for _ in range(150):
                v = x[sel] - gradx / (L[sel] + 1e-6) # +1e-6 for numerical stability
                dv = v - x[sel]
                bound2 = c[sel, None, None, None, None] - delta * (dv * dv).sum((1,2,3,4), keepdim=True)
                with autocast('cuda', enabled=use_amp):
                    energy_new2 = f(v, y[sel], sigma[sel])
                if torch.all(energy_new2 <= bound2.view(-1) * (1 + 1e-4)):
                    break
                mask2 = energy_new2[:, None, None, None, None] <= bound2
                L[sel] = torch.where(mask2, L[sel], L[sel] / rho,)
            x[idx] = z[idx]
            better = (energy_new2 <= energy_new[idx2]).nonzero().view(-1)
            x[sel[better]] = v[better]
        else:
            x[idx] = z[idx]

        # Residuals
        if i > 0:
            res[idx] = torch.norm(x[idx] - x_old[idx], p=2, dim=(1, 2, 3, 4)) / torch.norm(
                x[idx], p=2, dim=(1, 2, 3, 4)
            )

        assert not torch.any(
            torch.isnan(res)
        ), "Numerical errors! Some values became NaN!"
        condition = res >= tol
        idx = condition.nonzero().view(-1)  # Update which data to still iterate on

        if torch.max(res) < tol:
            if verbose:
                print(f"Converged in iter {i}, tol {torch.max(res).item():.6f}")
            break
            
        # Extrapolation & non-monotone params
        t_old = t
        t = (np.sqrt(4.0*t_old**2 + 1.0) + 1.0) / 2.0
        q_old = q
        q = eta * q + 1.0
        c[idx] = (eta * q_old * c[idx] + f(x[idx], y[idx], sigma[idx])) / q
        x_bar_old.copy_(x_bar)
        grad_old.copy_(grad)

    if verbose and (torch.max(res) >= tol):
        print(f"max iter reached, tol {torch.max(res).item():.6f}")
    # Cast back to float
    if use_amp:
        x = x.float()
    return x, L, i


def reconstruct_nmAPG(
    sigma: torch.Tensor, # noise level (Denoising strength)
    y: torch.Tensor,
    physics,
    data_fidelity,
    regularizer,
    lamda: float,
    step_size: float,
    max_iter: int,
    tol: float,
    x_init: torch.Tensor = None,
    detach_grads: bool = True,
    verbose: bool = False,
    return_stats: bool = False,
    use_amp: bool = False,
) -> torch.Tensor:
    """Wrapper using nmAPG for 3D volumes"""
    # Cast to half if using amp
    if use_amp:
        y = y.half()
        regularizer = regularizer.half()

    if x_init is not None:
        x0 = x_init.detach().clone()
    else:
        with autocast('cuda', enabled=use_amp):
            x0 = physics.A_dagger(y)

    def energy(val, y_in, sigma):
        with torch.no_grad():
            en = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val, sigma) # For training, batch-friendly
        return en.reshape(-1)

    def gradf(val, y_in, sigma):
        with torch.no_grad():
            g = data_fidelity.grad(val, y_in, physics) + lamda * regularizer.grad(val, sigma)
        return g

    energy_and_grad = lambda v, yv, sigma: (energy(v, yv, sigma), gradf(v, yv, sigma))

    # example energies
    rec, L, steps = nmAPG(
        sigma,
        x0=x0,
        y=y,
        f=energy,
        nabla=gradf,
        f_and_nabla=energy_and_grad,
        max_iter=max_iter,
        L_init=1/step_size,
        tol=tol,
        verbose=verbose,
        use_amp=use_amp,
    )
    
    stats = dict(L=L.detach(), steps=steps)
    if return_stats:
        return rec, stats
    return rec
