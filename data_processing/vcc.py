import torch
import torch.nn as nn

def virtual_coil_combination_2D(imgs, eps=1e-16):
    """
    Calculate the combination of all the coils using the virtual coil
    method for 2D images.

    Parameters
    ----------
    imgs: torch.Tensor
        The images reconstructed channel by channel [Nch, Nx, Ny]
    eps: float
        Small value to avoid division by zero

    Returns
    -------
    I: torch.Tensor
        The combination of all the channels in a complex valued [Nx, Ny]
    """
    # Ensure imgs is a complex tensor
    if not torch.is_complex(imgs):
        raise ValueError("Input tensor must be complex")

    # Compute the virtual coil
    nch, nx, ny = imgs.shape
    weights = torch.sum(torch.abs(imgs), dim=0, keepdim=True)
    weights = torch.clamp(weights, min=eps)

    phase_reference = torch.angle(torch.sum(imgs, dim=(1, 2), keepdim=True))
    reference = imgs / (weights * torch.exp(1j * phase_reference))
    virtual_coil = torch.sum(reference, dim=0)

    # Remove the background noise via low pass filtering
    hanning_2d = torch.outer(torch.hann_window(nx), torch.hann_window(ny))
    hanning_2d = torch.fft.fftshift(hanning_2d)[None, :, :]

    difference_original_vs_virtual = torch.conj(imgs) * virtual_coil
    difference_original_vs_virtual = torch.fft.ifft2(
        torch.fft.fft2(difference_original_vs_virtual, dim=(1, 2)) * hanning_2d, dim=(1, 2))

    combined_imgs = torch.sum(imgs * torch.exp(1j * torch.angle(difference_original_vs_virtual)), dim=0)
    return combined_imgs

def virtual_coil_combination_3D(imgs, eps=1e-16, preprocess=False):
    """
    Calculate the combination of all the coils using the virtual coil
    method for 2D images.

    Parameters
    ----------
    imgs: torch.Tensor
        The images reconstructed channel by channel [Nch, Nx, Ny]
    eps: float
        Small value to avoid division by zero

    Returns
    -------
    I: torch.Tensor
        The combination of all the channels in a complex valued [Nx, Ny]
    """
    # Ensure imgs is a complex tensor
    if not torch.is_complex(imgs):
        raise ValueError("Input tensor must be complex")

    # Compute the virtual coil
    nch, nx, ny, nz = imgs.shape
    weights = torch.sum(torch.abs(imgs), dim=0, keepdim=True)
    weights = torch.clamp(weights, min=eps)

    phase_reference = torch.angle(torch.sum(imgs, dim=(1, 2, 3), keepdim=True))
    reference = imgs / (weights * torch.exp(1j * phase_reference))
    virtual_coil = torch.sum(reference, dim=0)

    if preprocess:
        # Remove the background noise via low pass filtering
        hanning_3d = torch.outer(torch.hann_window(nx), torch.hann_window(ny), torch.hann_window(nz))
        filt = torch.fft.fftshift(hanning_3d)[None, :, :, :]
    else:
        filt = 1.0

    difference_original_vs_virtual = torch.conj(imgs) * virtual_coil
    difference_original_vs_virtual = torch.fft.ifftn(
        torch.fft.fftn(difference_original_vs_virtual, dim=(1, 2, 3)) * filt, dim=(1, 2, 3))

    combined_imgs = torch.sum(imgs * torch.exp(1j * torch.angle(difference_original_vs_virtual)), dim=0)
    return combined_imgs
