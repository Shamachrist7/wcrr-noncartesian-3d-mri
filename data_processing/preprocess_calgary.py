import os
import h5py 
import numpy as np
import glob
import torch
import argparse
from .vcc import virtual_coil_combination_3D

parser = argparse.ArgumentParser(description="Choosing the root directory")
parser.add_argument("--root", type=str, default='/LOCAL/mri_data')
inp = parser.parse_args()
root = inp.root + '/'

def _load_volumes(filename, sr = 0.85 ):
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

for train in [True, False]:

    if train:
        train_pth = root + 'Train/'
    else:
        train_pth = root + 'Val/'

    # Create important directories
    os.makedirs(train_pth + '_images/', exist_ok=True)
    os.makedirs(train_pth + '_images_vcc/', exist_ok=True)

    def from_file_to_multicoil_volumes(filename, sr=0.85):
        """
        Get multicoil raw kspace and images from h5 files
        """
        return  _load_volumes(filename, sr = sr )

    # loop through filenames in the folder
    def save2npy(train_pth, save_pth=None):
        for i, filename in enumerate(glob.glob(train_pth + '/*')):
            try:
                images = _load_volumes(filename)

                # save the images in numpy format with filename and np extension
                np.save(save_pth + filename.split('/')[-1] + '.npy', images)
            except:
                print('Error with file: ', filename)

    # If you want to save h5 to npy
    save_pth = train_pth + '_images/'
    save2npy(train_pth, save_pth)

    def vcc_and_save2npy(load_pth=None, save_pth=None):
        for i, filename in enumerate(glob.glob(load_pth + '/*')):
            images = np.load(filename)
            images = torch.from_numpy(images)  # shape (depth, height, width, channels)
            images = torch.moveaxis(images, -1, 0) # shape (channels, depth, height, width)
            images = virtual_coil_combination_3D(images)

            # save the images in numpy format with filename and np extension
            imname = filename.split('/')[-1]
            imname = imname.replace('.npy', '')
            imname = imname.replace('.h5', '')

            np.save(save_pth + imname + '.npy', images.cpu().numpy())

    load_pth_numpy = train_pth + '_images/'
    save_pth_vcc = train_pth + '_images_vcc/'

    vcc_and_save2npy(load_pth_numpy, save_pth_vcc)

