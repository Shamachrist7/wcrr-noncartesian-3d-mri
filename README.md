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

