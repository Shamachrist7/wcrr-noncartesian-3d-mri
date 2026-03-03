import torch
import torch.nn as nn
from deepinv.models import TVDenoiser
from deepinv.loss.metric import PSNR
from tqdm import tqdm


class PDHG_TV(nn.Module):
    def __init__(
        self,
        lambda_reg,
        max_iter,
        lipschitz, # of the data fidelity gradient, to be estimated via power iteration
        data_fidelity,
        stopping_criterion=1e-5,
    ):
        super(PDHG_TV, self).__init__()
        self.lambd = lambda_reg
        self.steps = max_iter
        self.sigma = lipschitz / (2 * 12) # 12 becomes 8 for a 2D problem, and 6 for a 1D problem, as it corresponds to the operator norm of the TV gradient
        self.tau = 1.5 / lipschitz
        self.stopping_criterion = stopping_criterion
        self.df = data_fidelity
        self.psnr_metric = PSNR()

    def forward(self, y, physics, init, x_gt=None, compute_metrics=False):
        # reconstruction with Condat-Vu primal-dual hybrid gradient algorithm
        x = init
        p = torch.zeros_like(TVDenoiser.nabla(x))
        
        metrics = {"psnr": [], "residual": []}
        if compute_metrics and x_gt is not None:
            metrics["psnr"].append(self.psnr_metric(x, x_gt).item())

        for step in tqdm(range(self.steps)):
            x_old = x.clone()

            # primal update
            x = x - self.tau * (self.df.grad(x, y, physics) + TVDenoiser.nabla_adjoint(p))

            # Dual update
            p = p + self.sigma * TVDenoiser.nabla(2 * x - x_old)
            p = p / torch.max(torch.tensor(1.0), p.norm()/self.lambd) # projection onto l2 ball

            rel_err = torch.linalg.norm(
                x_old.flatten() - x.flatten()
            ) / torch.linalg.norm(x.flatten() + 1e-12)
            
            if compute_metrics and x_gt is not None:
                metrics["psnr"].append(self.psnr_metric(x, x_gt).item())
                metrics["residual"].append(rel_err.item())
                
            if (rel_err < self.stopping_criterion):
                print("Converged at iteration:", step+1)
                break
            
        if compute_metrics and x_gt is not None:
            return x, metrics

        return x
