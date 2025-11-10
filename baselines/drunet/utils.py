import torch
import torch.nn as nn

class ArtifactRemoval(nn.Module):
    r"""
    Artifact removal architecture :math:`\phi(A^{\top}y)`.

    This differs from the dinv.models.ArtifactRemoval in that it allows to forward the physics.

    In the end we should not use this for unet !!
    """

    def __init__(self, backbone_net, pinv=False, ckpt_path=None, device=None, fm_mode=False):
        super(ArtifactRemoval, self).__init__()
        self.pinv = pinv
        self.backbone_net = backbone_net
        self.fm_mode = fm_mode

        if ckpt_path is not None:
            self.backbone_net.load_state_dict(torch.load(ckpt_path), strict=True)

    def forward_basic(self, y=None, physics=None, x_in=None, **kwargs):
        r"""
        Reconstructs a signal estimate from measurements y

        :param torch.tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        """
        x_in = physics.A_adjoint(y) if not self.pinv else physics.A_dagger(y)
        if hasattr(physics.noise_model, "sigma"):
            sigma = physics.noise_model.sigma
        else:
            sigma = 1e-3  # WARNING: this is a default value that we may not want to use?

        return self.backbone_net(x_in, sigma=sigma)

    def forward(self,  y=None, physics=None, x_in=None, **kwargs):
        return self.forward_basic(physics=physics, y=y, **kwargs)




def rescale_img(img, rescale_mode="min_max"):
    if rescale_mode == "min_max":
        if img.max() != img.min():
            img = img - img.min()
            img = img / img.max()
    elif rescale_mode == "clip":
        img = img.clamp(min=0.0, max=1.0)
    else:
        raise ValueError("rescale_mode has to be either 'min_max' or 'clip'.")
    return img
