# Data-Driven Weakly-Convex Ridge Regularizer for Accelerated 3D Non-Cartesian Parallel MRI Reconstruction

This `readme` file contains all the necessary informations to run this code and is structured into

1. Installation instructions & Preliminaries
2. Data Processing
3. Training
4. An overview of the baseline regularizers and instructions on how to use them for reconstruction (Upcoming)
6. Instructions to reproduce the evaluation runs (Upcoming)
7. Instructions to reproduce the training runs (Upcoming)
8. Instructions to reproduce the automatic hyperparameters tuning (Upcoming)

## 1. Installations & Preliminaries

To get started, clone the repository.

### a. Installations
The code relies mainly on [DeepInverse](https://deepinv.github.io), [MRI-NUFFT](https://mind-inria.github.io/mri-nufft/) and [GGRAPPA](https://github.com/mind-inria/ggrappa). For tuning hyperparameters, the Bayesian Optimizer [Optuna](https://optuna.org/) is used. [Weights & Biases](https://docs.wandb.ai/models/quickstart#command-line) is used to monitor, visualize and save the results of the different runs. All the necessary dependencies can be installed at once with the following command:

```
pip install deepinv mri-nufft gpunufft cupy-cuda12x git+https://github.com/mind-inria/ggrappa optuna wandb
```
Note that a CUDA 12 machine is necessary. Previous versions might encounter some issues.

### b. Preliminary
Before runing anything, sign up to Weights & Biases (If you don't have an account yet) through [wandb.ai/authorize](http://wandb.ai/signup). Once logged in into your account, go to *Settings → API keys*, then copy your **wandb API Key**. Finally in your Terminal (environment in which you are going to run everything), run ```wandb login```. You will be asked to provide your API Key; Paste it in there and validate. Your **wandb** automatic monitor of all the runs is all set.


## 2. Data Processing
The Calgary Campinas Train & Val data consist of 67 fully-sampled 12-coil k-space volumes saved as **.h5** files. First of all, organize them in a root directory that we will call **my_root_directory**, containing two folders named exactly **Train** (which contains the 47 .h5 training k-space volumes) and **Val** (which contains the 20 .h5 validation k-space volumes). For training purposes, we need the True image volumes (that we can compute by performing a 3D virtual coil combination (vcc) of the image domain versions of those kspaces). To get them, just run:
```
python data_processing/preprocess_calgary.py --root my_root_directory
```
It will create in each of the folders **Train** and **Val**, two sub-folders named **_images** (which contains the 12-coil image domain versions as .npy files; Each file actually keeps the name of its kspace version but ends with .h5.npy) and **_images_vcc** (which contains the estimated True MR image volumes via 3D virtual coil combination, as .npy files; Each file actually keeps the name of its kspace version but ends with .npy).


The Calgary Campinas Test data consist of 50 fully-sampled 12-coil k-space volumes and 50 fully-sampled 32-coil k-space volumes as **.h5** files. They just have to be organized as follows. In **my_root_directory**, create a folder named **Test** in which we have two sub-folders named **12coil** (which contains the  50 12-coil k-space .h5 files) and **32coil** (which contains the  50 32-coil k-space .h5 files). No further preprocessing is required!


## 3. Training


