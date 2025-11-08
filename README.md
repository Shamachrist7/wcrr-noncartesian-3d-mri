# Data-Driven Weakly-Convex Ridge Regularizer for Accelerated 3D Non-Cartesian Parallel MRI Reconstruction

This `readme` file contains all the necessary informations to run this repository and is structured into

1. Installations & Preliminaries
2. Data Processing
3. Baseline reconstruction methods & Trainings
4. Hyperparameters tuning
5. Reconstructions
6. wandb routine to fetch the saved reconstruction results & visualize some reconstructions

## 1. Installations & Preliminaries

To get started, clone the repository.

### a. Installations
The code relies mainly on [DeepInverse](https://deepinv.github.io), [MRI-NUFFT](https://mind-inria.github.io/mri-nufft/) and [GGRAPPA](https://github.com/mind-inria/ggrappa). For tuning hyperparameters, the Bayesian Optimizer [Optuna](https://optuna.org/) is used. [Weights & Biases](https://docs.wandb.ai/models/quickstart#command-line) is used to monitor, visualize and save the results of the different runs. All the necessary dependencies can be installed at once with the following command:

```
pip install deepinv mri-nufft gpunufft cupy-cuda12x git+https://github.com/mind-inria/ggrappa optuna wandb
```
Note that a CUDA 12 machine is necessary. Previous versions might encounter some issues.

### b. Preliminaries
Before runing anything, sign up to Weights & Biases (If you don't have an account yet) through [wandb.ai/authorize](http://wandb.ai/signup). Once logged in into your account, go to *Settings → API keys*, then copy your **wandb API Key**. Finally in your Terminal (environment in which you are going to run everything), run ```wandb login```. You will be asked to provide your API Key; Paste it in there and validate. Your **wandb** automatic monitor of all the runs is all set.


## 2. Data Processing
The Calgary Campinas Train & Val data consist of 67 fully-sampled 12-coil k-space volumes saved as **.h5** files. First of all, organize them in a root directory that we will call **my_root_directory**, containing two folders named exactly **Train** (which contains the 47 .h5 training k-space volumes) and **Val** (which contains the 20 .h5 validation k-space volumes). For training purposes, we need the True image volumes (that we can compute by performing a 3D virtual coil combination (vcc) of the image domain versions of those kspaces). To get them, just run:
```
python data_processing/preprocess_calgary.py --root my_root_directory
```
It will create in each of the folders **Train** and **Val**, two sub-folders named **_images** (which contains the 12-coil image domain versions as .npy files; Each file actually keeps the name of its kspace version but ends with .h5.npy) and **_images_vcc** (which contains the estimated True MR image volumes via 3D virtual coil combination, as .npy files; Each file actually keeps the name of its kspace version but ends with .npy).


The Calgary Campinas Test data consist of 50 fully-sampled 12-coil k-space volumes and 50 fully-sampled 32-coil k-space volumes as **.h5** files. They just have to be organized as follows. In **my_root_directory**, create a folder named **Test** in which we have two sub-folders named **12coil** (which contains the  50 12-coil k-space .h5 files) and **32coil** (which contains the  50 32-coil k-space .h5 files). No further preprocessing of the Test data is required!


## 3. Baseline reconstruction methods & Trainings
We compare the *Weakly-Convex Ridge Regularizer (WCRR) + nmAPG (non-monotone Accelerated Proximal Gradient) solver* to the following baseline methods:
- Anisotropic Total Variation (TV) regularizer + ADMM (Alternating Direction Method of Multipliers) solver
- $l_1$-wavelets regularizer + ADMM solver
- Convex Ridge Regularizer (CRR) + nmAPG solver
- Plug-and-Play (PnP) DRUNet + ADMM solver
- (At the moment, only the above mentioned methods are implemented in this repository. We will add more baselines later on!)

Out of all of them, only WCRR, CRR and DRUNet require training. We can train each of them by runing the following commands:
- For WCRR: ```python training_wcrr.py --root my_root_directory```
- For CRR: ```python training_wcrr.py --root my_root_directory --regularizer_name "CRR"```
- For DRUNet: ```python training_drunet.py --root my_root_directory```


## 4. Hyperparameters tuning

