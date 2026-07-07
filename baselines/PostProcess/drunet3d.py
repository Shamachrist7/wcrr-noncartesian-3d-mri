import torch
import torch.nn as nn
import torch.nn.functional as F


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
    x = F.pad(x, pad, mode="reflect") if any(pad) else x
    return x, pad


def crop_from_pad_3d(x, pad):
    pw0, pw1, ph0, ph1, pd0, pd1 = pad
    if not any(pad):
        return x
    return x[
        ...,
        pd0:x.shape[-3] - pd1,
        ph0:x.shape[-2] - ph1,
        pw0:x.shape[-1] - pw1,
    ]


class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


def block(c, nb):
    return nn.Sequential(*[ResBlock3D(c) for _ in range(nb)])


class DRUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, nc=(64, 128, 256, 512), nb=4, residual=True):
        super().__init__()
        self.residual = residual

        self.head = nn.Conv3d(in_ch, nc[0], 3, padding=1, bias=False)

        self.enc1 = nn.Sequential(block(nc[0], nb), nn.Conv3d(nc[0], nc[1], 2, stride=2, bias=False))
        self.enc2 = nn.Sequential(block(nc[1], nb), nn.Conv3d(nc[1], nc[2], 2, stride=2, bias=False))
        self.enc3 = nn.Sequential(block(nc[2], nb), nn.Conv3d(nc[2], nc[3], 2, stride=2, bias=False))

        self.body = block(nc[3], nb)

        self.up3 = nn.Sequential(nn.ConvTranspose3d(nc[3], nc[2], 2, stride=2, bias=False), block(nc[2], nb))
        self.up2 = nn.Sequential(nn.ConvTranspose3d(nc[2], nc[1], 2, stride=2, bias=False), block(nc[1], nb))
        self.up1 = nn.Sequential(nn.ConvTranspose3d(nc[1], nc[0], 2, stride=2, bias=False), block(nc[0], nb))

        self.tail = nn.Conv3d(nc[0], out_ch, 3, padding=1, bias=False)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.orthogonal_(m.weight, gain=0.2)

    def forward_unet(self, x):
        x1 = self.head(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)

        y = self.body(x4)
        y = self.up3(y + x4)
        y = self.up2(y + x3)
        y = self.up1(y + x2)
        y = self.tail(y + x1)
        return y

    def forward(self, x):
        # x: [B,1,D,H,W]
        x_pad, pad = pad_to_multiple_3d(x, m=8)
        y_pad = self.forward_unet(x_pad)
        y = crop_from_pad_3d(y_pad, pad)

        if self.residual:
            y = x + y

        return y.clamp_min(0.0)