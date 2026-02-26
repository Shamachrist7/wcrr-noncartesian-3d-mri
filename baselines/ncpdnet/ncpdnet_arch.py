import torch
from torch import nn
from . import build_model
from mrinufft.density.nufft_based import pipe

def measurements_residual(concatenated_kspace):
    batch_size = concatenated_kspace.shape[0] // 2
    current_kspace = concatenated_kspace[:batch_size, ...]
    original_kspace = concatenated_kspace[batch_size:, ...]
    return current_kspace - original_kspace

class CrossDomainNet(nn.Module):
    def __init__(
        self,
        nufft_op,
        image_net,
        kspace_net,
        n_coils=None,
        dim=3,
        i_buffer_size=1,
        k_buffer_size=1,
        i_buffer_mode=True,
        k_buffer_mode=False,
        domain_sequence="KIKI",
        complex_recon=False,
        normalize_input=False,
        **kwargs
    ):
        """
        Cross-Domain Network for iterative k-space and image-domain refinements.

        Parameters:
            image_net (nn.Module): Image-domain refinement network.
            kspace_net: k-space refinement module.
            kspace_loc (Tensor): Non-Cartesian k-space trajectory.
            n_coils (int, optional): Number of coils for test data.
            dim (int): Use 3D/2D processing.
            i_buffer_size (int): Image buffer size for recurrent refinements.
            k_buffer_size (int): k-space buffer size for recurrent refinements.
            domain_sequence (str): Alternating sequence of 'K' and 'I' refinements.
        """
        super(CrossDomainNet, self).__init__()
        self.kspace_net = kspace_net
        self.image_net = image_net
        self.domain_sequence = domain_sequence
        self.i_buffer_size = i_buffer_size
        self.k_buffer_size = k_buffer_size
        self.i_buffer_mode = i_buffer_mode
        self.k_buffer_mode = k_buffer_mode
        self.complex_recon = complex_recon
        self.dim = dim
        self.n_coils = n_coils
        self.normalize_input = normalize_input
        # Define Nufft operators
        self.nufft_op = nufft_op

    def apply_data_consistency(self, kspace, original_kspace):
        return torch.cat([kspace, original_kspace], 0)

    def forward_operator(self, image_data):
        return self.nufft_op.op(image_data[:, 0:1, ...])

    def backward_operator(self, kspace_data):
        return self.nufft_op.adj_op(kspace_data)

    def k_domain_correction(
        self, i_domain, image_buffer, kspace_buffer, original_kspace):
        forward_op_res = self.forward_operator(image_buffer)
        if self.k_buffer_mode:
            kspace_buffer = torch.cat(
                [
                    kspace_buffer,
                    forward_op_res,
                ],
                0,
            )
        else:
            kspace_buffer = forward_op_res
        kspace_buffer = self.apply_data_consistency(kspace_buffer, original_kspace)
        kspace_buffer = self.kspace_net[i_domain // 2](
            kspace_buffer
        )  
        return kspace_buffer  

    def i_domain_correction(self, i_domain, image_buffer, kspace_buffer):
        #print(f"first_k_buff:{kspace_buffer.shape}")
        backward_op_res = self.backward_operator(kspace_buffer)
        #print(f"backward_op_res:{backward_op_res.shape}")
        if self.i_buffer_mode:
            image_buffer = torch.cat(
                [
                    image_buffer,
                    backward_op_res,
                ],
                1,
            )
        else:
            image_buffer = backward_op_res
        image_buffer = self.image_net[i_domain // 2](image_buffer)
        return image_buffer

    def forward(self, original_kspace):
        # Load input data
        original_kspace = original_kspace.contiguous()
        #print(f"in_k:{original_kspace.shape}")
        # Compute x0
        image = self.backward_operator(original_kspace)
        #print(f"in_i:{image.shape}")
        if self.normalize_input:
            norm_fact = self._normalize_img(image)
            image = image / norm_fact
            original_kspace = original_kspace / norm_fact
        # Init buffers
        kspace_buffer = torch.cat([original_kspace] * self.k_buffer_size, dim=0)
        image_buffer = torch.cat([image] * self.i_buffer_size, dim=1)
        # Apply cross-domain refinement
        for i_domain, domain in enumerate(self.domain_sequence):
            if domain == "K":
                kspace_buffer = self.k_domain_correction(
                    i_domain,
                    image_buffer,
                    kspace_buffer,
                    original_kspace,
                )
            elif domain == "I":
                image_buffer = self.i_domain_correction(
                    i_domain, image_buffer, kspace_buffer
                )
        if self.normalize_input:
            image_buffer = image_buffer * norm_fact
        # Return the final image reconstruction
        if self.complex_recon: 
            recon = image_buffer[:, 0:1, ...] # coil-combined complex-valued image.
        else: 
            recon = torch.abs(image_buffer[:, 0:1, ...])  # magnitude image.
        
        return recon
    
    def _normalize_img(self,img):
        p = torch.quantile(img.abs(), 0.98)
        return p


class NCPDNET(nn.Module):
    def __init__(
        self,
        nufft_op,
        image_net_type="ImageNetCNN",
        base_filters=16,
        num_stages=3,
        n_primal=2,
        n_iter=6,
        activation="relu",
        dim=3,
        kspace_loc=None,
        n_coils=None,
        complex_recon=False,
        double_conv=False,
        **kwargs,
    ):
        super(NCPDNET, self).__init__()

        self.base_filters = base_filters
        self.num_stages = num_stages
        self.n_primal = n_primal
        self.n_iter = n_iter
        self.activation = activation
        self.dim = dim
        self.nufft_op = nufft_op

        # Build image and k-space networks
        image_net = nn.Sequential(
            *[
                build_model(
                    name=image_net_type,
                    custom_name=f"image_net_{i}",
                    num_stages=self.num_stages,
                    base_filters=self.base_filters,
                    out_channels=self.n_primal,
                    in_channels=self.n_primal + 1,
                    activation=self.activation,
                    dim=self.dim,
                    res=True,
                )
                for i in range(self.n_iter)
            ]
        )
        kspace_net = [measurements_residual for _ in range(self.n_iter)]

        crossDomain_net_args = dict(
            nufft_op=self.nufft_op,
            complex_recon=complex_recon,
            dim=self.dim,
            i_buffer_mode=True,
            k_buffer_mode=False,
            i_buffer_size=self.n_primal,
            k_buffer_size=1,
            domain_sequence="KI" * self.n_iter,
            **kwargs,
        )
        self.ncpdnet = CrossDomainNet(
                kspace_net=kspace_net,
                image_net=image_net,
                **crossDomain_net_args
            )
        
    def update_nufft_op(self, new_nufft_op):
        """
        Replace the NUFFT operator used in the network.
        """
        self.nufft_op = new_nufft_op
        if hasattr(self.ncpdnet, "nufft_op"):
            self.ncpdnet.nufft_op = new_nufft_op

    def forward(self, original_kspace):
        return self.ncpdnet(original_kspace)
    
    
