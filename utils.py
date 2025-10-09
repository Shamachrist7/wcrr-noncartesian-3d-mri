import numpy as np
import torch
from mrinufft import get_operator
import deepinv as dinv
from deepinv.optim.utils import conjugate_gradient
from deepinv.loss.metric import PSNR, SSIM


def sum_of_squares(img_channels: np.ndarray) -> np.ndarray:
    """Combines complex channels with square root sum of squares.

    :param img_channels: Complex channels
    :return: Combined image
    """
    sos = np.sqrt((np.abs(img_channels) ** 2).sum(axis=0))
    return sos

# -----------------------------------
# Utils: RI (Real-Imaginary)<->complex conversions
# -----------------------------------
def ri_to_complex(x_ri: torch.Tensor) -> torch.Tensor:
    """
    x_ri: [1,2,H,W,D] float -> complex tensor [H,W,D]
    """
    assert x_ri.ndim == 5 and x_ri.shape[0] == 1 and x_ri.shape[1] == 2, \
        f"Expected [1,2,H,W,D], got {tuple(x_ri.shape)}"
    real = x_ri[0, 0]
    imag = x_ri[0, 1]
    return (real + 1j * imag).to(torch.complex64)

def complex_to_ri(x_c: torch.Tensor) -> torch.Tensor:
    """
    x_c: complex tensor [H,W,D] -> [1,2,H,W,D] float
    """
    x_c = x_c.to(torch.complex64)
    real = torch.real(x_c).unsqueeze(0).unsqueeze(0)
    imag = torch.imag(x_c).unsqueeze(0).unsqueeze(0)
    return torch.cat([real, imag], dim=1).to(torch.float32)
    
# -----------------------------------
# Complex magnitude PSNR & SSIM (for both, x_rec & x_ref are assumed to be in the RI space)
# -----------------------------------
psnr = lambda x_rec, x_ref: PSNR(max_pixel=None)(torch.abs(ri_to_complex(x_rec)).unsqueeze(0), torch.abs(ri_to_complex(x_ref)).unsqueeze(0))
ssim = lambda x_rec, x_ref: SSIM(max_pixel=None)(torch.abs(ri_to_complex(x_rec)).unsqueeze(0), torch.abs(ri_to_complex(x_ref)).unsqueeze(0))

# -----------------------------------
# Physics wrapper for DeepInv in RI variable space
# -----------------------------------
class MRINUFFTPhysicsRI(dinv.physics.Physics):
    """
    DeepInv Physics operating on x in RI space [1,2,H,W,D].
    Internally converts to complex for mrinufft, and back.
    """
    def __init__(self, E):
        super().__init__()
        self.E = E

    def A(self, x_ri: torch.Tensor) -> torch.Tensor:
        # [1,2,H,W,D] -> complex -> numpy -> NUFFT
        x_c = ri_to_complex(x_ri).detach().cpu().numpy()
        y_np = self.E.op(x_c)  # complex numpy with shape like [Nsamples, ncoils] (backend-dependent)
        return torch.from_numpy(y_np).to(x_ri.device)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # complex torch (measurement domain) -> numpy -> adjoint NUFFT -> complex image -> RI 2ch
        y_np = y.detach().cpu().numpy()
        x_np = self.E.adj_op(y_np)  # complex numpy [H,W,D]
        x_c = torch.from_numpy(x_np).to(y.device)
        return complex_to_ri(x_c)

    # >>> KEY OVERRIDES to avoid torch.func.vjp on numpy-bridged A <<< (It works here because, it's a linear operator)
    def A_vjp(self, x, v):
        """Vector–Jacobian product for linear A is A^H v."""
        return self.A_adjoint(v)

    def A_jvp(self, x, v):
        """Jacobian–vector product for linear A is A v."""
        return self.A(v)

    def prox_l2(self, v, y, gamma: float, **kwargs):
        """
        Solve (I + gamma A^H A) x = v + gamma A^H y  with CG.
        v, return x have shape [B,2,H,W,D] (RI).
        y is complex torch tensor in the measurement domain.
        """

        b = v + gamma * self.A_adjoint(y)

        def M(z):
            Az  = self.A(z)
            AHAz = self.A_adjoint(Az)
            return z + gamma * AHAz
      
        x = conjugate_gradient(M, b, max_iter=2000, tol=1e-4)
        return x

# -----------------------------------
# Spectral norm estimate (L) in RI space for stepsize
# --------------------------------------
def power_iteration_L_RI(physics: MRINUFFTPhysicsRI, shape_ri, iters=20, device="cpu"):
    """
    Estimate L ≈ ||A||^2 for the linear map x_ri -> A(x_ri), with x_ri in RI space [1,2,H,W,D].
    """
    z = torch.randn(*shape_ri, device=device)
    z = z / (torch.linalg.vector_norm(z) + 1e-12)
    val = torch.tensor(1.0, device=device)
    for _ in range(iters):
        w = physics.A_adjoint(physics.A(z))   # [1,2,H,W,D]
        val = torch.tensordot(z, w, dims=len(z.shape))  # <z, w> in R
        z = w / (torch.linalg.vector_norm(w) + 1e-12)
    return float(max(val.item(), 1e-8))
