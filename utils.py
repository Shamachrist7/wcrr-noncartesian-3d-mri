import numpy as np
import h5py
import torch
import deepinv as dinv
from deepinv.optim.utils import conjugate_gradient
from deepinv.loss.metric import PSNR, SSIM
from deepinv.loss.metric.metric import Metric
from deepinv.optim.data_fidelity import DataFidelity
from mrinufft.io import read_arbgrad_rawdat
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

def sum_of_squares(img_channels: np.ndarray) -> np.ndarray:
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
# Complex magnitude PSNR & SSIM (for both, x_rec & x_ref are assumed to be the Magnitudes of the volumes)
# -----------------------------------
psnr = lambda x_rec, x_ref: PSNR(max_pixel=None)(x_rec.unsqueeze(0), x_ref.unsqueeze(0))
ssim = lambda x_rec, x_ref: SSIM(max_pixel=None)(x_rec.unsqueeze(0), x_ref.unsqueeze(0))

# helper to get the psnr history  during reconstructions from start to end
class PSNR_MRI(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def metric(self, x_net, x, *args, **kwargs):
        return PSNR(max_pixel=None)(torch.abs(ri_to_complex(x_net)).unsqueeze(0).cpu(), torch.abs(ri_to_complex(x)).unsqueeze(0).cpu())

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
        self.E = E # nufft operator

    def A(self, x_ri: torch.Tensor) -> torch.Tensor:
        # [1,2,H,W,D] -> complex -> numpy -> NUFFT
        x_c = ri_to_complex(x_ri)#.detach().cpu().numpy()
        y = self.E.op(x_c)  # complex numpy with shape like [Nsamples, ncoils] (backend-dependent)
        return y#.to(x_ri.device)

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

    def __init__(self, weights):
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

def normalize_kspace(kspace_data, kspace_loc, thresh=0.05):
    """
    Normalize k-space data by the average energy of the central region.

    Parameters:
    - kspace_loc: np.ndarray of shape (M, 3), the 3D k-space coordinates.
    - kspace_data: np.ndarray of shape (N_coils, M), the complex k-space values.
    - thresh: float, optional. Threshold radius for selecting the central region.
    If 0.0, only the loc closest to the center is used.

    Returns:
    - kspace_data_norm: np.ndarray, normalized k-space data.
    - normalization_fact: float, the normalization factor used.
    """
    dist_to_center = np.linalg.norm(kspace_loc, axis=-1)
    central_reg = np.zeros_like(dist_to_center, dtype=bool)
    if thresh==0.0:
        central_reg[np.argmin(dist_to_center)] = True
    else:
        central_reg[dist_to_center <= thresh] = True
    combined_energy = np.sqrt(np.sum(np.abs(kspace_data)**2, axis=0)) #SoS instead of Abs
    normalization_fact = np.mean(combined_energy[central_reg])
    return kspace_data/normalization_fact , normalization_fact
