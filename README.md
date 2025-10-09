# Data-Adaptive Ridge Regularizers for Accelerated 3D Non-Cartesian Parallel MRI Reconstruction

This `readme` file is structured into

1. Installation instructions of the dependencies
2. An overview of the implemented regularizer architectures (CRR and WCRR) and instructions on how to use them for reconstruction (Upcoming)
3. An overview of the baseline regularizers and instructions on how to use them for reconstruction (Upcoming)
4. Instructions to reproduce the training for Learnable regularizers (Upcoming)
5. Instructions to reproduce the evaluation runs (Upcoming)
6. Instructions to reproduce the training runs (Upcoming)
7. Instructions to reproduce the automatic hyperparameters tuning (Upcoming)

## 1. Installation

To get started using `conda`, clone the repository and run (may take a few minutes)
```
conda env create --file=environment.yaml
```

The code relies mainly on [DeepInverse](https://deepinv.github.io), [MRI-NUFFT](https://mind-inria.github.io/mri-nufft/) and [GGRAPPA](https://github.com/mind-inria/ggrappa). [Optuna](https://optuna.org/) is used for Bayesian hyperparameters optimization. All the necessary dependencies can be installed at once with the following command:

```
pip install deepinv mri-nufft gpunufft cupy-cuda12x git+https://github.com/mind-inria/ggrappa optuna
```
Note that a CUDA 12 machine is necessary. Previous versions might encounter some issues.


## 2. Overview of Regularizers and Reconstruction

