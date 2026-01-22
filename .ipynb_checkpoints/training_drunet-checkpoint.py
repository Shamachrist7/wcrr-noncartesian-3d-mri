import torch
import torch.utils
import torchvision
import deepinv as dinv
import numpy as np
from deepinv.utils import plot
from argparse import ArgumentParser
import wandb
from pathlib import Path
from baselines.drunet.utils import ArtifactRemoval, rescale_img
from baselines.drunet.drunet_base import DRUNet
from data_processing import load_data

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

import matplotlib
#matplotlib.pyplot.close()

parser = ArgumentParser(description="Choosing the root directory")
parser.add_argument("--root", type=str, default='../../../../../../../../LOCAL/mri_data')
parser.add_argument('--grayscale', type=int, default=0)
parser.add_argument('--ckpt_resume', type=str, default='')
parser.add_argument('--model_name', type=str, default='drunet')
parser.add_argument('--wandb_resume_id', type=str, default='')
parser.add_argument('--lr_scheduler', type=str, default='multistep')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--sigma_min', type=float, default=0.01)
parser.add_argument('--sigma_max', type=float, default=0.1)
parser.add_argument('--train_batch_size', type=int, default=12) #12
parser.add_argument('--train_patch_size', type=int, default=64) #64
parser.add_argument('--jac_reg', type=bool, default=False) # Decides wether to perform Jacobian regularization or not (True or False)
parser.add_argument('--jac_reg_para', type=float, default=1e2) # The Jacobian regularization parameter if applicable (float)
args = parser.parse_args()
root = args.root # root directory for the data

WANDB_LOGS_PATH = "weights/drunet"
OUT_DIR = Path("weights/drunet")
CKPT_DIR = OUT_DIR  # path to store the checkpoints
PRETRAINED_PATH = None #"weights/drunet/drunet_3d_complex_denoise.pth"
WANDB_PROJ_NAME = 'DRUNet_MRI_denoiser'  # Name of the wandb project


class MyTrainer(dinv.training.Trainer):
    def __init__(self, *args, jac_reg=False, jac_reg_para=1e-4, **kwargs):
        super(MyTrainer, self).__init__(*args, **kwargs)
        self.jac_reg = jac_reg # Decides wether to perform Jacobian regularization or not (True or False)
        self.jac_reg_para = jac_reg_para # The Jacobian regularization parameter if applicable (float)

    def to_image(self, x):
        r"""
        Convert the tensor to an image. Necessary for complex images (2 channels)

        :param torch.Tensor x: input tensor
        :return: image
        """
        if x.shape[1] == 2:
            out = torch.moveaxis(x, 1, -1).contiguous()
            out = torch.view_as_complex(out).abs().unsqueeze(1)
        else:
            out = x
        if len(x.shape) == 5:
            out = out[:, :, 0, :, :]
        return out

    def prepare_images(self, physics_cur, x, y, x_net):
        r"""
        Prepare the images for plotting.

        It prepares the images for plotting by rescaling them and concatenating them in a grid.

        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Reconstruction network output.
        :returns: The images, the titles, the grid image, and the caption.
        """
        with torch.no_grad():
            if len(y.shape) == len(x.shape) and y.shape != x.shape:
                y_reshaped = torch.nn.functional.interpolate(y, size=x.shape[2])
                if hasattr(physics_cur, "A_adjoint"):
                    imgs = [y_reshaped, physics_cur.A_adjoint(y), x_net, x]
                    caption = (
                        "From top to bottom: input, backprojection, output, target"
                    )
                    titles = ["Input", "Backprojection", "Output", "Target"]
                else:
                    imgs = [y_reshaped, x_net, x]
                    titles = ["Input", "Output", "Target"]
                    caption = "From top to bottom: input, output, target"
            else:
                if hasattr(physics_cur, "A_adjoint"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        back = physics_cur.module.A_adjoint(y)
                    else:
                        back = physics_cur.A_adjoint(y)
                    imgs = [back, x_net, x]
                    titles = ["Backprojection", "Output", "Target"]
                    caption = "From top to bottom: backprojection, output, target"
                elif y.shape == x.shape:
                    imgs = [y, x_net, x]
                    titles = ["Measurement", "Output", "Target"]
                    caption = "From top to bottom: measurement, output, target"
                else:
                    imgs = [x_net, x]
                    caption = "From top to bottom: output, target"
                    titles = ["Output", "Target"]

            # Concatenate the images along the batch dimension
            for i in range(len(imgs)):
                imgs[i] = self.to_image(imgs[i])

            vis_array = torch.cat(imgs, dim=0)
            for i in range(len(vis_array)):
                vis_array[i] = rescale_img(vis_array[i], rescale_mode="min_max")
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])

        return imgs, titles, grid_image, caption

    def plot(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Plot the images.

        It plots the images at the end of each epoch.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        post_str = "Training" if train else "Eval"
        if self.plot_images and ((epoch + 1) % self.freq_plot == 0):
            imgs, titles, grid_image, caption = self.prepare_images(
                physics, x, y, x_net
            )

            # if MRI in class name, rescale = min-max
            if "MRI" in str(physics):
                rescale_mode = "min_max"
            else:
                rescale_mode = "clip"
            plot(
                imgs,
                titles=titles,
                show=self.plot_images,
                return_fig=True,
                rescale_mode=rescale_mode,
            )

            if self.wandb_vis:
                log_dict_post_epoch = {}
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch[post_str + " samples"] = images
                log_dict_post_epoch["step"] = epoch
                wandb.log(log_dict_post_epoch)
    
    def jac_pow_loss(self, x, physics, M=50, tol=1e-2, xi=0.05, logger=None):
        # initialize unit vector v (per sample)
        v = torch.randn_like(x)
        v = torch.nn.functional.normalize(v, dim=[-4, -3, -2, -1], out=v)
        
        f = lambda inp: self.model(inp, physics)
        
        v_old = v.clone()
        for i in range(M):
            # JVP: J v
            _, Jv = torch.func.jvp(f, (x,), (v,))
            # VJP factory at y
            _, vjp_f = torch.func.vjp(f, x)
            # JTJ v = J^T (J v)
            JTJv = vjp_f(Jv)[0]
            
            # normalize next v
            v = torch.nn.functional.normalize(JTJv, dim=[-4, -3, -2, -1])
            
            if torch.norm(v - v_old) / x.size(0) < tol:
                break
            v_old = v.clone()
            
            # Rayleigh quotient per sample
            norm_sq = torch.sum(v * JTJv, dim=[1, 2, 3, 4]) / torch.sum(v * v, dim=[1, 2, 3, 4])
            norm_sq = torch.sum(torch.clip(norm_sq, min=1 - xi, max=None)) / x.size(0)
            #print(f"{norm_sq:.6f}")
            return norm_sq
            
    
    def compute_loss(self, physics, x, y, train=True, epoch: int = None, step=False):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :param bool step: Whether to perform an optimization step when computing the loss.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        if train and step:
            self.optimizer.zero_grad()

        if train or self.compute_eval_losses:
            # Evaluate reconstruction network
            x_net = self.model_inference(y=y, physics=physics, x=x, train=True)

            # Compute the losses
            loss_total = 0
            for k, l in enumerate(self.losses):
                loss = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                    epoch=epoch,
                )
                loss_total += loss.mean()
                meters = (
                    self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                )
                meters.update(loss.detach().cpu().numpy())
                if len(self.losses) > 1 and self.verbose_individual_losses:
                    logs[l.__class__.__name__] = meters.avg

            meters = self.logs_total_loss_train if train else self.logs_total_loss_eval
            meters.update(loss_total.item())
            logs[f"l1_loss"] = meters.avg
            
            if self.jac_reg:
                weight = torch.rand((x_net.shape[0], 1, 1, 1, 1)).to(device)
                interpolation = weight * x_net + (1 - weight) * y
                jac_reg_cost = self.jac_pow_loss(interpolation, physics) # Cost of the jacobian regularizer
                logs["jac_reg_cost"] = jac_reg_cost.item()
                # Add it to the total loss (so that the updates take it into account)
                loss_total += self.jac_reg_para * jac_reg_cost
                logs["TotalLoss"] = loss_total.item() 
                
        else:
            loss_total = 0
            x_net = None

        if train:
            loss_total.backward()  # Backward the total loss

            norm = self.check_clip_grad()
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            if step:
                self.optimizer.step()  # Optimizer step

        return loss_total, x_net, logs




def train_denoiser(model_name='drunet',
                   ckpt_resume=None,
                   wandb_resume_id=None,
                   seed=0,
                   wandb_vis=True,
                   epochs=None,
                   train_batch_size=None,
                   train_patch_size=None,
                   sigma_min=0.01,
                   sigma_max=0.1,
                   jac_reg=False,
                   jac_reg_para=10**2):

    if train_patch_size is None:
        train_patch_size = 64

    if train_batch_size is None:
        train_batch_size = 12

    train_dataloader = [load_data(root, train_patch_size, train_batch_size, device=device, train=True, scale_fact=1e-6)]
    val_dataloader = [load_data(root, train_patch_size, train_batch_size, device=device, train=False, scale_fact=1e-6)]
    physics = [dinv.physics.DecomposablePhysics(device=device, noise_model=dinv.physics.noise.GaussianNoise(sigma=sigma_min))]
    physics_generator = [dinv.physics.generator.SigmaGenerator(sigma_min=sigma_min, sigma_max=sigma_max, device=device)]

    model = DRUNet(in_channels=2, out_channels=2, dim=3, pretrained=PRETRAINED_PATH, train=True, device=device).to(device)
    model = ArtifactRemoval(model).to(device)

    if ckpt_resume is not None:
        model.load_state_dict(torch.load(ckpt_resume, map_location=lambda storage, loc: storage)['state_dict'])
        print('Model loaded from', ckpt_resume)

    if epochs is None:
        epochs = 12000 #20000 # 1000 * train_batch_size

    # choose training losses
    losses = dinv.loss.SupLoss(metric=torch.nn.L1Loss(reduction="mean"))

    # choose optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=int(epochs / 8))

    wandb_setup = {'dir': WANDB_LOGS_PATH,
                   'mode': 'online',
                   'project': WANDB_PROJ_NAME}

    print('Start training on ', device, ' batch size = ', train_batch_size, ' patch size = ', train_patch_size)
     
    trainer = MyTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        epochs=epochs,
        scheduler=scheduler,
        losses=losses,
        physics=physics,
        physics_generator=physics_generator,
        optimizer=optimizer,
        device=device,
        save_path=CKPT_DIR, #str(CKPT_DIR / operation),
        verbose=True,
        wandb_vis=wandb_vis,
        wandb_setup=wandb_setup,
        plot_images=False,
        eval_interval=100,
        ckp_interval=2000,
        online_measurements=True,
        check_grad=True,
        ckpt_pretrained=ckpt_resume,
        freq_plot=1,
        show_progress_bar=True,
        jac_reg=jac_reg,
        jac_reg_para=jac_reg_para,
    )

    trainer.train()


grayscale = False if args.grayscale == 0 else True
ckpt_resume = None if args.ckpt_resume == '' else args.ckpt_resume
wanddb_resume_id = None if args.wandb_resume_id == '' else args.wandb_resume_id
lr_scheduler = None if args.lr_scheduler == '' else args.lr_scheduler
epochs = None if args.epochs == 0 else args.epochs
train_batch_size = None if args.train_batch_size == 0 else args.train_batch_size
train_patch_size = None if args.train_patch_size == 0 else args.train_patch_size
jac_reg = args.jac_reg # Decides wether to perform Jacobian regularization or not (True or False)
jac_reg_para = args.jac_reg_para # The Jacobian regularization parameter if applicable (float)

# check_dataloader(train=True)

train_denoiser(model_name=args.model_name, epochs=epochs, train_batch_size=train_batch_size,
           train_patch_size=train_patch_size, seed=args.seed, sigma_min=args.sigma_min,
           sigma_max=args.sigma_max, ckpt_resume=ckpt_resume, jac_reg=jac_reg, jac_reg_para=jac_reg_para)

