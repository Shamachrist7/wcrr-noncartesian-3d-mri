import torch
import torch.nn.functional as F
from torch import nn 


class ConvBlock(nn.Module):
    """
    A flexible convolutional block supporting 2D and 3D dims.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function name ("relu", "leaky_relu", etc.).
        double_conv (bool): If True, uses two conv layers per level; else uses one.
        batch_norm (bool): If True, adds BatchNorm.
        dim (int): Dimensionality of the conv block (2 or 3).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation="relu",
        double_conv=False,
        batch_norm=False,
        dim=3,
    ):
        super(ConvBlock, self).__init__()
        self.dim = dim
        self.double_conv = double_conv
        self.batch_norm = batch_norm
        self.activation = get_activation(activation)
        # Select layer types dynamically
        self.Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        self.conv_block = self._make_conv_block(in_channels, out_channels)
        

    def _make_conv_block(self, in_channels, out_channels):
        """Helper function to build the convolutional block."""
        layers = self._add_conv_layer(in_channels, out_channels)
        if self.double_conv:
            layers += self._add_conv_layer(out_channels, out_channels)
        return nn.Sequential(*layers)

    def _add_conv_layer(self, in_channels, out_channels):
        """Adds a single convolution layer with optional batch norm and activation."""
        if self.dim ==2 :
            layers = [self.Conv(in_channels, out_channels, kernel_size=2, padding=1, bias=False)]
        elif self.dim == 3 :
            layers = [self.Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
        else:
            raise NotImplementedError
        if self.batch_norm:
            layers.append(self.BatchNorm(out_channels))
        layers.append(self.activation)
        return layers

    def forward(self, x):
        return self.conv_block(x)
    
class CNNblock(nn.Module):
    def __init__(
        self,
        num_stages=3,
        base_filters=16,
        in_channels=1,
        out_channels=1,
        activation="relu",
        res=True,
        dim=3,
        double_conv=False,
        batch_norm=False,
        **kwargs
    ):
        super(CNNblock, self).__init__()
        self.num_stages = num_stages
        if self.num_stages < 2:
            raise ValueError("Choose at least two conv blocks for CNN")
        self.base_filters = base_filters
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.res = res
        self.dim = dim
        self.double_conv = double_conv
        self.batch_norm = batch_norm
        
        in_channels = in_channels * 2 
        self.out_channels = self.out_channels * 2 
        
        self.blocks = []
        for i in range(self.num_stages):
            self.blocks.append(
                ConvBlock(
                    in_channels if i==0 else base_filters,
                    self.out_channels if i==self.num_stages-1 else base_filters,
                    double_conv=double_conv,
                    batch_norm=batch_norm,
                    activation=activation,
                    dim=self.dim,
                )
            )
        self.cnn = nn.Sequential(*self.blocks)

    def forward(self, inputs):
        x = torch.cat((torch.real(inputs), torch.imag(inputs)), dim=1)
        x = self.cnn(x)
        x = to_complex(x, self.out_channels//2)
        if self.res:
            x = inputs[:, :self.out_channels//2, ...] + x
        return x
    
# Subclassing to add custom name to each image net network
class ImageNetCNN(CNNblock):
    def __init__(
        self, custom_name, **kwargs
    ):
        super(ImageNetCNN, self).__init__(**kwargs)
        self.custom_name = custom_name

    def __repr__(self):
        return f"{self.custom_name}({super().__repr__()})"
    
class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=3,
        base_filters=32,
        num_stages=4,
        activation="relu",
        double_conv=False,
        batch_norm=False,
        res=False,
        **kwargs
    ):
        """
        Parameters:
            in_channels (int): Number of input channels, for complex input.
            out_channels (int): Number of output channels, for complex input.
            dim (int): Dimensionality of the conv block (2 or 3).
            base_filters (int): Number of filters at the first level (doubles with each encoder stage).
            num_stages (int): Depth of the U-Net (number of encoder/decoder levels).
            activation (str): Activation function to use ("relu", "leaky_relu", etc.).
            double_conv (bool): If True, use two conv layers per block; else use one.
            batch_norm (bool): If True, apply Batch Normalization after conv layers.
            res (bool): If True, adds a residual connection from input to output.
        """ 
        super(Unet, self).__init__()
        in_channels *= 2 
        out_channels *= 2 
        self.dim = dim
        self.num_stages = num_stages
        self.activation = activation
        self.residual = res
        self.out_channels = out_channels
        self.MaxPool = nn.MaxPool3d if dim==3 else nn.MaxPool2d
        self.Upconv = nn.ConvTranspose3d if dim==3 else nn.ConvTranspose2d
        self.Fconv = nn.Conv3d if dim==3 else nn.Conv2d
        # Downsampling path (encoder)
        self.encoders = nn.ModuleList()
        filters = base_filters
        for i in range(num_stages):
            if i == 0:
                self.encoders.append(
                    ConvBlock(
                        in_channels,
                        filters,
                        double_conv=double_conv,
                        batch_norm=batch_norm,
                        activation=activation,
                        dim=self.dim
                    )
                )
            else:
                self.encoders.append(
                    ConvBlock(
                        filters // 2,
                        filters,
                        double_conv=double_conv,
                        batch_norm=batch_norm,
                        activation=activation,
                        dim=self.dim
                    )
                )
            filters *= 2  # Double the filters at each downscale

        # Bottleneck
        self.bottleneck = ConvBlock(
            filters // 2, filters, double_conv=True, batch_norm=batch_norm, dim=self.dim, activation=activation        )

        # Upsampling path (decoder)
        self.decoders = nn.ModuleList()
        filters = filters // 2
        for i in range(num_stages):
            self.decoders.append(self.Upconv(filters * 2, filters, kernel_size=2, stride=2))
            self.decoders.append(
                ConvBlock(filters * 2, filters, double_conv=double_conv, dim=self.dim, activation=activation)
            )
            filters //= 2

        # Final output conv
        self.out_conv = self.Fconv(filters * 2, out_channels, kernel_size=1)

    def forward(self, inputs):
        x = torch.cat((torch.real(inputs), torch.imag(inputs)), dim=1)
        # Padding
        x, padding = pad_for_pool(x, self.num_stages, dim=self.dim)
        # Encoder path (Down)
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x.clone())
            x = self.MaxPool(kernel_size=2, stride=2)(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (Up)
        for i in range(self.num_stages):
            x = self.decoders[i * 2](x)  # Transposed conv for upsampling
            encoder_output = encoder_outputs[-(i + 1)]

            # Concatenate encoder output with the upsampled output
            x = torch.cat([encoder_output, x], dim=1)
            x = self.decoders[i * 2 + 1](x)  # ConvBlock after concatenation

        # Final conv-layer
        x = self.out_conv(x)
        
        x = to_complex(unpad_tensor(x, padding, dim=self.dim), self.out_channels // 2)
        if self.residual:
            x = inputs[:, : self.out_channels // 2, ...] + x    
        return x
    
class ImageNetUNET(Unet):
    def __init__(
        self,
        custom_name,
        **unet_args
    ):
        super(ImageNetUNET, self).__init__(**unet_args)
        self.custom_name = custom_name

    def __repr__(self):
        return f"{self.custom_name}({super().__repr__()})"
    
    
##### Utils fcts #####
def to_complex(x, n):
    return torch.complex(
        x[:, :n, ...].type(torch.float32),
        x[:, n:, ...].type(torch.float32),
    )
    
def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "lrelu":
        return nn.LeakyReLU(negative_slope=0.02)
    elif activation == "silu": 
        return nn.SiLU()
    else:
        raise ValueError("Activation function is not supported yet")
    
def pad_for_pool_3d(input_tensor, n_pool_stages):
    """
    Pad the input tensor so that its spatial dimensions (height, width, depth)
    are divisible by 2^n_pool_stages for the U-Net.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, height, width, depth).
        n_pool_stages (int): Number of pooling stages (or downsampling layers).

    Returns:
        padded_tensor (torch.Tensor): Padded input tensor with new spatial dimensions.
        padding (tuple): The padding added for each dimension (depth, width, height).
    """
    _, _, height, width, depth = input_tensor.shape

    # Calculate the required size for each dimension to be divisible by 2^n_pool_stages
    required_height = int(
        torch.ceil(torch.tensor(height / (2**n_pool_stages))) * (2**n_pool_stages)
    )
    required_width = int(
        torch.ceil(torch.tensor(width / (2**n_pool_stages))) * (2**n_pool_stages)
    )
    required_depth = int(
        torch.ceil(torch.tensor(depth / (2**n_pool_stages))) * (2**n_pool_stages)
    )

    pad_height = required_height - height
    pad_width = required_width - width
    pad_depth = required_depth - depth

    p_depth = (
        pad_depth // 2,
        pad_depth - pad_depth // 2,
    )  # depth
    p_width = (
        pad_width // 2,
        pad_width - pad_width // 2,
    )  # width
    p_height = pad_height // 2, pad_height - pad_height // 2  # height

    padded_tensor = F.pad(
        input_tensor, pad=(*p_depth, *p_width, *p_height), mode="constant", value=0
    )

    return padded_tensor, (*p_height, *p_width, *p_depth)


def unpad_tensor_3d(padded_tensor, padding):
    """
    Remove the padding from the tensor, based on the padding values.

    Args:
        padded_tensor (torch.Tensor): The padded tensor.
        padding (tuple): The padding applied for each dimension (height,width,depth).

    Returns:
        unpadded_tensor (torch.Tensor): The tensor with the padding removed.
    """

    pad_height_before, pad_height_after = padding[0], padding[1]
    pad_width_before, pad_width_after = padding[2], padding[3]
    pad_depth_before, pad_depth_after = padding[4], padding[5]
    unpadded_tensor = padded_tensor[
        :,
        :,
        pad_height_before : padded_tensor.shape[2] - pad_height_after,
        pad_width_before : padded_tensor.shape[3] - pad_width_after,
        pad_depth_before : padded_tensor.shape[4] - pad_depth_after,
    ]

    return unpadded_tensor

def pad_for_pool_2d(input_tensor, n_pool_stages):
    """
    Pad the input tensor so that its height and width are divisible by 2^n_pool_stages.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        n_pool_stages (int): Number of pooling stages.

    Returns:
        padded_tensor (torch.Tensor): Padded tensor.
        padding (tuple): Padding applied (H_before, H_after, W_before, W_after).
    """
    _, _, height, width = input_tensor.shape

    # Calculate required sizes
    divisor = 2 ** n_pool_stages
    required_height = int(torch.ceil(torch.tensor(height / divisor)) * divisor)
    required_width = int(torch.ceil(torch.tensor(width / divisor)) * divisor)

    pad_height = required_height - height
    pad_width = required_width - width

    p_height = (pad_height // 2, pad_height - pad_height // 2)
    p_width = (pad_width // 2, pad_width - pad_width // 2)

    # Note: F.pad expects (W_left, W_right, H_top, H_bottom)
    padded_tensor = F.pad(input_tensor, (*p_width, *p_height), mode="constant", value=0)

    return padded_tensor, (*p_height, *p_width)


def unpad_tensor_2d(padded_tensor, padding):
    """
    Remove the padding from a 2D tensor.

    Args:
        padded_tensor (torch.Tensor): Tensor of shape (B, C, H, W).
        padding (tuple): (H_before, H_after, W_before, W_after)

    Returns:
        unpadded_tensor (torch.Tensor): Tensor with padding removed.
    """
    h_before, h_after, w_before, w_after = padding
    return padded_tensor[
        :,
        :,
        h_before : padded_tensor.shape[2] - h_after,
        w_before : padded_tensor.shape[3] - w_after,
    ]
    
def pad_for_pool(input_tensor, n_pool_stages, dim=3):
    if dim == 3:
        return pad_for_pool_3d(input_tensor, n_pool_stages)
    elif dim == 2:
        return pad_for_pool_2d(input_tensor, n_pool_stages)
    else:
        raise ValueError("Only 2D and 3D supported")

def unpad_tensor(padded_tensor, padding, dim=3):
    if dim == 3:
        return unpad_tensor_3d(padded_tensor, padding)
    elif dim == 2:
        return unpad_tensor_2d(padded_tensor, padding)
    else:
        raise ValueError("Only 2D and 3D supported")
    
