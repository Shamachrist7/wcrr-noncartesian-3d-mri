import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from deepinv.optim import Prior
import torchcde


class LinearSpline(nn.Module):
    """
    Learn N functions alpha_i(σ) = exp( s_i(σ) ) / (σ + eps),
    where s_i is a natural cubic spline over σ.

    Args
    ----
    N : int
        Number of alphas.
    K : int
        Number of spline knots (>=2).
    sigma_min, sigma_max : float
        Range of σ where knots are placed.
    eps : float
        Small constant in denominator for stability (e.g., 1e-5).
    domain : {'linear','log'}
        If 'log', knots are spaced uniformly in log(σ), and evaluation
        is done on log(σ) too (good when σ spans orders of magnitude).
    init : float
        Initial value for s_i at all knots.
    """

    def __init__(self, N: int = 32, K: int = 12,
                 sigma_min: float = 0.01, sigma_max: float = 0.1,
                 eps: float = 1e-5, init: float = 0.0):
        super().__init__()
        assert K >= 2, "Need at least 2 knots."
        assert sigma_max > sigma_min > 0.0

        self.N = N
        self.K = K
        self.eps = float(eps)

        # --- fixed knot locations (registered buffers, no grads) ---
        t = torch.linspace(sigma_min, sigma_max, K)
        self.register_buffer("t_knots", t)  # shape [K]

        # --- learnable spline values at knots: s_i(t_k) ---
        # Shape [1, K, N]: batch=1, time=K, channels=N
        self.s_at_knots = nn.Parameter(torch.full((1, K, N), float(init)))

    def _build_spline(self):
        """
        Build a LinearSpline object from current knot values.
        torchcde expects data of shape [B, T, C] with strictly increasing T.
        """
        # coefficients are computed with gradients flowing to s_at_knots
        coeffs = torchcde.linear_interpolation_coeffs(
            self.s_at_knots, t=self.t_knots
        )
        return torchcde.LinearInterpolation(coeffs)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all N alphas at a batch of noise levels.

        Parameters
        ----------
        sigma : Tensor, shape [B] or [B,1]
            Per-sample noise levels (must be > 0).

        Returns
        -------
        alphas : Tensor, shape [B, N]
        """
        sigma = sigma.view(-1)  # [B]
        
        spline = self._build_spline()              # linear spline s(t)
        s_vals = spline.evaluate(sigma)           # shape [1, B, N]
        s_vals = s_vals.squeeze(0)                 # -> [B, N]

        alphas = torch.exp(s_vals) / (sigma.view(-1, 1) + self.eps)
        return alphas.view(len(sigma), self.N, 1, 1, 1)

class ZeroMean3D(nn.Module):
    """Enforcing zero mean on 3D filters improves performance"""
    def forward(self, x):
        return x - torch.mean(x, dim=(1, 2, 3, 4), keepdim=True)

class WCRR3D(Prior):
    def __init__(
        self,
        weak_convexity,
        tanh=False,
        nb_channels=[2, 4, 8, 16],  # channels per layer
        filter_sizes=[5, 5, 5],      # 3D kernel sizes
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=None,
        rotations=True,
    ):
        super(WCRR3D, self).__init__()

        self.nb_filters = nb_channels[-1]
        self.filter_size = sum(filter_sizes) - len(filter_sizes) + 1

        # Build a cascade of 3D convolutions
        self.filters = nn.Sequential(
            *[
                nn.Conv3d(
                    nb_channels[i],
                    nb_channels[i + 1],
                    filter_sizes[i],
                    padding=filter_sizes[i] // 2,
                    bias=False,
                )
                for i in range(len(filter_sizes))
            ]
        )
        P.register_parametrization(self.filters[0], "weight", ZeroMean3D())

        # 3D Dirac for Lipschitz estimation
        sz = 2 * self.filter_size - 1
        # infer input channels from first conv layer
        in_ch = self.filters[0].in_channels
        self.dirac = torch.zeros(1, in_ch, sz, sz, sz, device=device)
        center = self.filter_size - 1
        for c in range(in_ch):
            self.dirac[0, c, center, center, center] = 1.0
            
        self.scaling = LinearSpline(N=self.nb_filters, K=12, sigma_min=0.01, sigma_max=0.1, eps=1e-5) #nn.Parameter(torch.zeros(1, self.nb_filters, 1, 1, 1, device=device))
        
        self.beta = nn.Parameter(torch.tensor(4.0, device=device))

        self.weak_cvx = weak_convexity
        self.tanh = tanh
        self.rotations = rotations

        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def smooth_l1(self, x):
        if self.tanh:
            x_abs = torch.abs(x)
            return torch.log((torch.exp(x - x_abs) + torch.exp(-x - x_abs)) / 2) + x_abs
        return torch.clip(x**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(x), 1.0) - 1.0

    def grad_smooth_l1(self, x):
        if self.tanh:
            return torch.tanh(x)
        return torch.clip(x, -1.0, 1.0)

    def get_conv_lip(self):
        imp = self.filters(self.dirac)
        for filt in reversed(self.filters):
            imp = F.conv_transpose3d(imp, filt.weight, padding=filt.padding)
        lip = torch.fft.fftn(imp.float()).abs().max()
        return lip.to(imp.dtype)

    def conv(self, x):
        lip = torch.sqrt(self.get_conv_lip())
        return self.filters(x / lip)

    def conv_transpose(self, x):
        lip = torch.sqrt(self.get_conv_lip())
        out = x / lip
        for filt in reversed(self.filters):
            out = F.conv_transpose3d(out, filt.weight, padding=filt.padding)
        return out

    def grad(self, x, sigma): # sigma --> 1D tensor with size equals the batch size of x

        beta_sp = torch.exp(self.beta)
        scale_sp = self.scaling(sigma)
        
        def grad_R(x):
            g = self.conv(x) * scale_sp
            g = self.grad_smooth_l1(beta_sp * g) - self.grad_smooth_l1(g) * self.weak_cvx
            g = g / scale_sp
            return self.conv_transpose(g)
        if self.rotations:
            x_DH = torch.rot90(x, k=1, dims=(-3,-2))
            x_DW = torch.rot90(x, k=1, dims=(-3,-1))
            x_HW = torch.rot90(x, k=1, dims=(-2,-1))
            grad_cost = grad_R(x) + torch.rot90(grad_R(x_DH), k=-1, dims=(-3,-2)) + torch.rot90(grad_R(x_DW), k=-1, dims=(-3,-1)) + torch.rot90(grad_R(x_HW), k=-1, dims=(-2,-1))
        return grad_cost/4 if self.rotations else grad_R(x)

    def g(self, x, sigma): # sigma --> 1D tensor with size equals the batch size of x

        beta_sp = torch.exp(self.beta)
        scale_sp = self.scaling(sigma)
        
        def R(x):
            r = self.conv(x) * scale_sp
            r = self.smooth_l1(beta_sp * r) / beta_sp - self.smooth_l1(r) * self.weak_cvx
            r = r / scale_sp**2
            return r.sum(dim=(1, 2, 3, 4))
        if self.rotations:
            x_DH = torch.rot90(x, k=1, dims=(-3,-2))
            x_DW = torch.rot90(x, k=1, dims=(-3,-1))
            x_HW = torch.rot90(x, k=1, dims=(-2,-1))
            cost = R(x) + R(x_DH) + R(x_DW) + R(x_HW)
        return cost/4 if self.rotations else R(x)

    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        return super()._apply(fn)




class WCRR3D_eval:
    def __init__(
        self,
        conv_lip,
        beta,
        scaling_sigma,
        filters,
        effective_filters=False,
        rotations=True,
        weak_cvx=1.0,
    ):
        super(WCRR3D_eval, self).__init__()
        
        self.filters = filters
        self.scaling_sigma = scaling_sigma
        self.beta = beta
        self.lip = torch.sqrt(conv_lip)
        self.rotations = rotations
        self.weak_cvx = weak_cvx
        self.effective_filters = effective_filters

    def smooth_l1(self, x):
        return torch.clip(x**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(x), 1.0) - 1.0

    def grad_smooth_l1(self, x):
        return torch.clip(x, -1.0, 1.0)

    def conv(self, x):
        if self.effective_filters:
            return F.conv3d(x / self.lip, self.filters, bias=None, stride=1, padding=3)
        return self.filters(x / self.lip)

    def conv_transpose(self, x):
        if self.effective_filters:
            return F.conv_transpose3d(x / self.lip, self.filters, bias=None, stride=1, padding=3)
        out = x / self.lip
        for filt in reversed(self.filters):
            out = F.conv_transpose3d(out, filt.weight, padding=filt.padding)
        return out

    def grad(self, x, sigma): # sigma --> 1D tensor with size equals the batch size of x
        
        def grad_R(x):
            g = self.conv(x) * self.scaling_sigma
            g = self.grad_smooth_l1(self.beta * g) - self.grad_smooth_l1(g) * self.weak_cvx
            g = g / self.scaling_sigma
            return self.conv_transpose(g)
        if self.rotations:
            x_DH = torch.rot90(x, k=1, dims=(-3,-2))
            x_DW = torch.rot90(x, k=1, dims=(-3,-1))
            x_HW = torch.rot90(x, k=1, dims=(-2,-1))
            grad_cost = grad_R(x) + torch.rot90(grad_R(x_DH), k=-1, dims=(-3,-2)) + torch.rot90(grad_R(x_DW), k=-1, dims=(-3,-1)) + torch.rot90(grad_R(x_HW), k=-1, dims=(-2,-1))
        return grad_cost/4 if self.rotations else grad_R(x)

    def g(self, x, sigma): # sigma --> 1D tensor with size equals the batch size of x
        
        def R(x):
            r = self.conv(x) * self.scaling_sigma
            r = self.smooth_l1(self.beta * r) / self.beta - self.smooth_l1(r) * self.weak_cvx
            r = r / self.scaling_sigma**2
            return r.sum(dim=(1, 2, 3, 4))
        if self.rotations:
            x_DH = torch.rot90(x, k=1, dims=(-3,-2))
            x_DW = torch.rot90(x, k=1, dims=(-3,-1))
            x_HW = torch.rot90(x, k=1, dims=(-2,-1))
            cost = R(x) + R(x_DH) + R(x_DW) + R(x_HW)
        return cost/4 if self.rotations else R(x)
    






