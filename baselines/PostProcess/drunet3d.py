import torch
import torch.nn as nn
import torch.nn.functional as F


def sequential(*modules):
    if len(modules) == 1:
        return modules[0]
    out = []
    for m in modules:
        if isinstance(m, nn.Sequential):
            out += list(m.children())
        elif isinstance(m, nn.Module):
            out.append(m)
    return nn.Sequential(*out)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
         bias=True, mode="CRC", negative_slope=0.2, dim=3):
    layers = []
    for t in mode:
        if t == "C":
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "T":
            layers.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif t == "R":
            layers.append(nn.ReLU(inplace=True))
        elif t == "r":
            layers.append(nn.ReLU(inplace=False))
        elif t == "L":
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        elif t == "l":
            layers.append(nn.LeakyReLU(negative_slope, inplace=False))
        elif t == "E":
            layers.append(nn.ELU(inplace=True))
        elif t == "S":
            layers.append(nn.Softplus())
        elif t == "U":
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        else:
            raise NotImplementedError(f"Undefined mode: {t}")
    return sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, mode="CRC", negative_slope=0.2, dim=3):
        super().__init__()
        assert in_channels == out_channels
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding,
                        bias, mode, negative_slope, dim=dim)

    def forward(self, x):
        return x + self.res(x)


def downsample_strideconv(in_channels, out_channels, bias=True, mode="2", dim=3):
    k = int(mode[0])
    return conv(in_channels, out_channels, kernel_size=k, stride=k,
                padding=0, bias=bias, mode="C", dim=dim)


def downsample_avgpool(in_channels, out_channels, bias=True, mode="2", dim=3):
    k = int(mode[0])
    return nn.Sequential(
        nn.AvgPool3d(k, stride=k),
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=bias),
    )


def downsample_maxpool(in_channels, out_channels, bias=True, mode="2", dim=3):
    k = int(mode[0])
    return nn.Sequential(
        nn.MaxPool3d(k, stride=k),
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=bias),
    )


def upsample_convtranspose(in_channels, out_channels, bias=True, mode="2", dim=3):
    k = int(mode[0])
    return conv(in_channels, out_channels, kernel_size=k, stride=k,
                padding=0, bias=bias, mode="T", dim=dim)


def upsample_upconv(in_channels, out_channels, bias=True, mode="2", dim=3):
    return nn.Sequential(
        nn.Upsample(scale_factor=int(mode[0]), mode="nearest"),
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=bias),
    )


def weights_init_drunet(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.orthogonal_(m.weight, gain=0.2)


def pad_to_multiple_3d(x, m=8):
    _, _, D, H, W = x.shape

    def p(n):
        r = n % m
        q = 0 if r == 0 else m - r
        return q // 2, q - q // 2

    pd0, pd1 = p(D)
    ph0, ph1 = p(H)
    pw0, pw1 = p(W)

    pad = (pw0, pw1, ph0, ph1, pd0, pd1)
    if any(pad):
        x = F.pad(x, pad, mode="reflect")
    return x, pad


def crop_from_pad_3d(x, pad):
    pw0, pw1, ph0, ph1, pd0, pd1 = pad
    if not any(pad):
        return x
    return x[..., pd0:x.shape[-3]-pd1,
                ph0:x.shape[-2]-ph1,
                pw0:x.shape[-1]-pw1]


def match_spatial(x, ref):
    d, h, w = ref.shape[-3:]
    x = x[..., :d, :h, :w]

    pd = d - x.shape[-3]
    ph = h - x.shape[-2]
    pw = w - x.shape[-1]

    if pd > 0 or ph > 0 or pw > 0:
        x = F.pad(x, (0, max(pw, 0), 0, max(ph, 0), 0, max(pd, 0)))
    return x


class DRUNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        pretrained=None,
        train=True,
        device=None,
        blind=True,
        dim=3,
    ):
        super().__init__()

        self.blind = blind
        self.dim = dim
        in_channels = in_channels + 1

        if downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        elif downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        else:
            raise NotImplementedError(downsample_mode)

        if upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        elif upsample_mode == "upconv":
            upsample_block = upsample_upconv
        else:
            raise NotImplementedError(upsample_mode)

        self.m_head = conv(in_channels, nc[0], bias=False, mode="C", dim=dim)

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C", dim=dim) for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode="2", dim=dim),
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C", dim=dim) for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode="2", dim=dim),
        )
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C", dim=dim) for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode="2", dim=dim),
        )

        self.m_body = sequential(
            *[ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C", dim=dim) for _ in range(nb)]
        )

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2", dim=dim),
            *[ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C", dim=dim) for _ in range(nb)],
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2", dim=dim),
            *[ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C", dim=dim) for _ in range(nb)],
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2", dim=dim),
            *[ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C", dim=dim) for _ in range(nb)],
        )

        self.m_tail = conv(nc[0], out_channels, bias=False, mode="C", dim=dim)

        if pretrained is None:
            self.apply(weights_init_drunet)
        else:
            ckpt = torch.load(pretrained, map_location=device)
            self.load_state_dict(ckpt, strict=True)

        if not train:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

        if device is not None:
            self.to(device)

    def forward_unet(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        x = self.m_body(x4)

        x = self.m_up3(x + x4)
        x = self.m_up2(match_spatial(x, x3) + x3)
        x = self.m_up1(match_spatial(x, x2) + x2)
        x = self.m_tail(match_spatial(x, x1) + x1)

        return x

    def forward(self, x, sigma=0.0):
        # x: [B,C,D,H,W]

        if isinstance(sigma, torch.Tensor):
            if sigma.ndim == 0:
                sigma = sigma.expand(x.size(0))
            s = sigma.view(x.size(0), 1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
            s = s.expand(-1, 1, x.size(2), x.size(3), x.size(4))
        else:
            s = torch.full(
                (x.size(0), 1, x.size(2), x.size(3), x.size(4)),
                float(sigma),
                device=x.device,
                dtype=x.dtype,
            )

        x = torch.cat([x, s], dim=1)

        x, pad = pad_to_multiple_3d(x, m=8)
        x = self.forward_unet(x)
        x = crop_from_pad_3d(x, pad)

        return x