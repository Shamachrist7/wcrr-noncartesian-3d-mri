import torch
import numpy as np    
from ggrappa.grappaND import GRAPPA_Recon
from ggrappa.utils import get_cart_portion_sparkling, get_grappa_filled_data_and_loc
import warnings
import scipy as sp

def do_grappa_and_append_data(kspace_loc, kspace_data, traj_params, acs=None, af=(2, 2), caipi_delta=0):
    """Perform GRAPPA reconstruction and append the data.
    This is a multi-coil GRAPPA reconstruction function that processes k-space data
    and appends the reconstructed data to the existing k-space locations and data.

    Parameters:
        kspace_loc (np.ndarray): Locations in k-space.
        kspace_data (np.ndarray): K-space data.
        traj_params (dict): Trajectory parameters.
        grappa_maker (callable): Function to create GRAPPA reconstructor.
        acs (np.ndarray, optional): Autocalibration signal data. Defaults to None.
            
    Returns:
        tuple: Updated k-space locations and data after GRAPPA reconstruction.
    """
    kspace_shots = kspace_loc.reshape(traj_params['num_shots'], -1, traj_params['dimension'])
    gridded_center, new_kspace_data, new_kspace_loc = get_cart_portion_sparkling(kspace_shots, traj_params, kspace_data)
    if acs is not None:
        if acs.shape[1] != traj_params['img_size'][0]:
            warnings.warn("ACS size does not match the image size. Re-sampling")
            acs = sp.signal.resample(
                acs, traj_params['img_size'][0], axis=1
            )
    grappa_recon, grappa_kernel = GRAPPA_Recon(
        sig=torch.tensor(gridded_center).permute(0, 2, 3, 1),
        af=af,
        acs=torch.tensor(acs).permute(0, 2, 3, 1) if acs is not None else None,
        isGolfSparks=True,
        delta=caipi_delta,
    )
    grappa_recon = grappa_recon.permute(0, 3, 1, 2).numpy()
    extra_loc, extra_data = get_grappa_filled_data_and_loc(gridded_center, grappa_recon, traj_params)
    kspace_loc = np.concatenate([new_kspace_loc, extra_loc], axis=0)
    kspace_data = np.hstack([new_kspace_data, extra_data])
    return kspace_loc, kspace_data
