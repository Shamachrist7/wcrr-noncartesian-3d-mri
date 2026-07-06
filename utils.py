import numpy as np
import h5py
import torch
import torch.nn as nn
from deepinv.physics import LinearPhysics
from deepinv.optim.utils import conjugate_gradient
from deepinv.loss.metric.metric import Metric
from deepinv.optim.data_fidelity import DataFidelity
from mrinufft.io import read_arbgrad_rawdat
from skimage.metrics import structural_similarity
import nibabel as nib

def fft(x):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=(-3, -2, -1)), norm='ortho', axes=(-3, -2, -1)), axes=(-3, -2, -1))

def get_acs_locations(acs_shape=(24, 24), img_size=(256,240,176)):
    mask = np.zeros(img_size, dtype=bool)
    mask[:, img_size[1]//2-acs_shape[0]//2:img_size[1]//2+acs_shape[0]//2, img_size[2]//2-acs_shape[1]//2:img_size[2]//2+acs_shape[1]//2] = True
    loc = np.asarray(np.nonzero(mask)).T / img_size - 0.5
    return loc

def simulate_acs_data(x, acs_shape=(24, 24)):
    kspace = fft(x)
    return kspace[:, :, kspace.shape[-2]//2 - acs_shape[0]//2 : kspace.shape[-2]//2 + acs_shape[0]//2, 
                  kspace.shape[-1]//2 - acs_shape[1]//2 : kspace.shape[-1]//2 + acs_shape[1]//2]

def sum_of_squares(img_channels: torch.tensor) -> torch.tensor:
    """Combines complex channels with square root sum of squares.

    :param img_channels: Complex channels
    :return: Combined image
    """
    sos = torch.sqrt((torch.abs(img_channels) ** 2).sum(dim=0))
    return sos



# -----------------------------------------------------------------
# Preprocess the kspace volume and return the image domain version
# -----------------------------------------------------------------
def _load_volumes(filename, sr = 0.85 ):
    if filename.endswith('.dat'):
        kspace_data, data_header = read_arbgrad_rawdat(filename)
        data_header['ref'] = nib.load(filename.replace('.dat', '_ref.nii.gz')).get_fdata(dtype=np.complex64)
        return (kspace_data.astype(np.complex64).reshape(kspace_data.shape[0], -1), data_header)
    with h5py.File(filename,'r') as h5obj :
        kspace_hybrid = h5obj['kspace'][:]

    # Explicit zero-filling after 85% in the slice-encoded direction
    Nz = kspace_hybrid.shape[2]
    Nz_sampled = int(np.ceil(Nz*sr))
    kspace_hybrid[:,:,Nz_sampled:,:] = 0
    kspace_hybrid = kspace_hybrid[:, :, :, ::2] + 1j * kspace_hybrid[:, :, :, 1::2] 
    images = np.fft.ifft2(np.fft.ifftshift(kspace_hybrid, axes=[1,2]),axes=[1,2]) #from x,ky,kz to x,y,z
    # Crop around center
    VOLUME_SIZE = (256, 218, 170)

    if images.shape[-2] != VOLUME_SIZE[-1]:
        D = (images.shape[-2] - VOLUME_SIZE[-1]) // 2
        images = images[:, :, D:-D,:]
    images = images.astype(np.complex64)

    return images


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
# Complex magnitude (masked) PSNR & SSIM (for both, pred & gt are assumed to be the torch.Tensor Magnitudes of the volumes)
# -----------------------------------
def psnr(pred, gt):
    mask = compute_mask(gt.numpy())
    return masked_psnr(gt.numpy(), pred.numpy(), mask)

def ssim(pred, gt):
    mask = compute_mask(gt.numpy())
    return masked_ssim(gt.numpy(), pred.numpy(), mask)

def nmse(pred, gt):
    gt_np = gt.numpy()
    pred_np = pred.numpy()
    mask = compute_mask(gt_np)
    return masked_nmse(gt_np, pred_np, mask)

# helper to get the psnr history  during reconstructions from start to end
class PSNR_MRI(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def metric(self, x_net, x, *args, **kwargs):
        mask = compute_mask(torch.abs(ri_to_complex(x)).cpu().numpy())
        curr_psnr = masked_psnr(torch.abs(ri_to_complex(x)).cpu().numpy(), torch.abs(ri_to_complex(x_net)).cpu().numpy(), mask)
        return torch.tensor(curr_psnr)

# -----------------------------------
# Physics wrapper for DeepInv in RI variable space
# -----------------------------------
class MRINUFFTPhysicsRI(LinearPhysics):
    """
    DeepInv Physics operating on x in RI space [1,2,H,W,D].
    Internally converts to complex for mrinufft, and back.
    """
    def __init__(self, E):
        super().__init__()
        self.E = E # nufft operator

    def A(self, x_ri: torch.Tensor) -> torch.Tensor:
        # [1,2,H,W,D] -> complex -> numpy -> NUFFT
        x_c = ri_to_complex(x_ri)#.detach().cpu().numpy()
        y = self.E.op(x_c)  # complex numpy with shape like [Nsamples, ncoils] (backend-dependent)
        return y.unsqueeze(0) #.to(x_ri.device)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # complex torch (measurement domain) -> numpy -> adjoint NUFFT -> complex image -> RI 2ch
        x_c = self.E.adj_op(y)  # complex numpy [H,W,D]
        return complex_to_ri(x_c)#.to(y.device)

    def prox_l2_precon(self, v, y, gamma: float, weights=1.0, tol=1e-4, **kwargs):
        """
        Solve (I + gamma A^H weights A) x = v + gamma A^H weights y  with CG.
        v, return x have shape [B,2,H,W,D] (RI).
        y is complex torch tensor in the measurement domain.
        """

        b = v + gamma * self.A_adjoint(weights * y)

        def M(z):
            Az  = self.A(z)
            AHWAz = self.A_adjoint(weights * Az)
            return z + gamma * AHWAz
      
        x = conjugate_gradient(M, b, tol=tol)
        return x
    
    def A_dagger(self, y, x_init=None, max_iter=10):
        return complex_to_ri(self.E.pinv_solver(y, max_iter=max_iter, x0=x_init))

# Custom L2 data_fidelity preconditionned with the density compensation weights
class L2_precon(DataFidelity):
    r"""
    Implementation of the data-fidelity as the normalized :math:`\ell_2` norm

    .. math::

        f(x) = \frac{1}{2}\|\weights^{0.5}(forw{x}-y)\|^2
        Its gradient
        And its proximity operator.

    .. doctest::

    """

    def __init__(self, weights=torch.tensor(1.0)):
        super().__init__()
        self.weights = weights # density compensation weights

    def fn(
        self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs
    ) -> torch.Tensor:
        return 0.5 * (torch.abs(torch.sqrt(self.weights) * (physics.A(x) - y))**2).sum()

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs
    ) -> torch.Tensor:
        return physics.A_adjoint(self.weights * (physics.A(x) - y))

    def prox(self, x: torch.Tensor, y: torch.Tensor, physics, *args, gamma: float | torch.Tensor = 1.0, tol=1e-4, **kwargs
    ) -> torch.Tensor:
        r"""
        Proximal operator of :math:`\gamma \datafid{Ax}{y} = \frac{\gamma}{2\sigma^2}\|Ax-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \datafidname}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \datafidname} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|Au-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator :math:`\operatorname{prox}_{\gamma \datafidname}(x)`.
        """
        return physics.prox_l2_precon(x, y, gamma, self.weights, tol=tol)

def normalize_kspace(kspace_data: torch.Tensor, kspace_loc, thresh=0.05):
    """
    Normalize k-space data by the average energy of the central region.

    Parameters:
    - kspace_loc: torch.Tensor of shape (M, 3), the 3D k-space coordinates.
    - kspace_data: torch.Tensor of shape (N_coils, M), the complex k-space values.
    - thresh: float, optional. Threshold radius for selecting the central region.
    If 0.0, only the loc closest to the center is used.

    Returns:
    - kspace_data_norm: torch.Tensor, normalized k-space data.
    - normalization_fact: float, the normalization factor used.
    """
    dist_to_center = torch.linalg.norm(kspace_loc, dim=-1)
    central_reg = torch.zeros_like(dist_to_center, dtype=torch.bool)
    if thresh==0.0:
        central_reg[torch.argmin(dist_to_center)] = True
    else:
        central_reg[dist_to_center <= thresh] = True
    combined_energy = sum_of_squares(kspace_data) # SoS instead of Abs
    normalization_fact = torch.mean(combined_energy[central_reg])
    return kspace_data/normalization_fact , normalization_fact



# -----------------------------------
# Computes the DPIR parameters (denoiser noise level and stepsize per iteration) based on the noise level of the input image and the regularization parameter lambda.
# -----------------------------------
def get_DPIR_params(num_iter=8, sigma=2e-3, lmbd=5.5):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    :param str, torch.device device: Device to run the algorithm, either "cpu" or "cuda". Default is "cpu".
    :return: tuple(list with denoiser noise level per iteration, list with stepsize per iteration, iterations).
    """
    sigma_init=0.015
    sigma_denoiser = torch.logspace(
        torch.log10(torch.tensor(sigma_init, dtype=torch.float32)),
        torch.log10(torch.tensor(sigma, dtype=torch.float32)),
        steps=num_iter,
        dtype=torch.float32,
    )

    stepsize = lmbd * (sigma_denoiser / sigma) ** 2

    return sigma_denoiser, stepsize, num_iter



# -----------------------------------
# Computes the effective filters
# -----------------------------------
def compute_effective_filters(conv_seq: nn.Sequential):
    """
    Compute the effective 3D filters of a sequential stack of Conv3d layers
    by passing a Dirac impulse through the network.

    Parameters
    ----------
    conv_seq : nn.Sequential
        A sequence of 3D convolutional layers with padding that preserves spatial dimensions.

    Returns
    -------
    torch.Tensor
        Effective filters of shape [out_channels, in_channels, R, R, R],
        where R is the overall receptive field.
    """
    # Determine input/output channels and kernel sizes
    in_channels = conv_seq[0].in_channels
    out_channels = conv_seq[-1].out_channels

    # Compute overall receptive field
    rf = 1
    for layer in conv_seq:
        if isinstance(layer, nn.Conv3d):
            k = layer.kernel_size[0]  # assume cubic kernels
            rf += (k - 1)
    # rf is now the size of the effective kernel
    L = rf
    pad = L // 2

    # Create Dirac impulse input: shape [1, in_channels, L, L, L]
    impulse = torch.zeros(1, in_channels, L, L, L)
    center = pad
    # Set an impulse in each input channel
    for c in range(in_channels):
        impulse[0, c, center, center, center] = 1.0

    # Run through the conv sequence
    conv_seq = conv_seq.eval()
    with torch.no_grad():
        out = conv_seq(impulse)

    # out shape: [1, out_channels, L, L, L]
    filters = out[0]  # [out_channels, L, L, L]
    # reshape to [out_channels, in_channels, L, L, L]
    # Actually, because impulse had in_channels separate impulses, 
    # out[0, o, :, :, :] holds sum over inputs; we need one impulse per input channel separately:
    # So rerun per input channel:
    eff_filters = torch.zeros(out_channels, in_channels, L, L, L)
    with torch.no_grad():
        for c in range(in_channels):
            imp_c = torch.zeros_like(impulse)
            imp_c[0, c, center, center, center] = 1.0
            out_c = conv_seq(imp_c)[0]  # [out_channels, L, L, L]
            eff_filters[:, c] = out_c
    return eff_filters



# -----------------------------------
#  Masked metrics
# -----------------------------------

def compute_mask(gt, threshold=0.05):

    mag = np.abs(gt)

    mask = mag > threshold * mag.max()

    return mask


def masked_psnr(gt, pred, mask):

    mse = np.mean((gt[mask]-pred[mask])**2)

    data_range = gt.max()

    return 20*np.log10(data_range/np.sqrt(mse))


def masked_ssim(gt, pred, mask):

    vals = []

    for z in range(gt.shape[-1]):

        if mask[..., z].sum() < 10:
            continue

        vals.append(
            structural_similarity(
                (gt*mask)[..., z],
                (pred*mask)[..., z],
                data_range=(gt*mask)[..., z].max()
            )
        )

    return np.mean(vals)


def masked_nmse(gt, pred, mask):
    gt = gt[mask]
    pred = pred[mask]
    return ((gt - pred) ** 2).sum() / ((gt ** 2).sum() + 1e-12)
