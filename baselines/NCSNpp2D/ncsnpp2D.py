
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 256, scale: float = 16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1).float()
        x = sigma[:, None] * self.W[None, :] * 2.0 * math.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, 2 * out_ch)
        self.norm2 = group_norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = group_norm(ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        q = self.q(y).reshape(b, c, h * w).permute(0, 2, 1)
        k = self.k(y).reshape(b, c, h * w)
        v = self.v(y).reshape(b, c, h * w).permute(0, 2, 1)
        a = torch.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        y = torch.bmm(a, v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(y)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode='nearest'))


class NCSNpp2D(nn.Module):
    """Compact NCSN++-style VE score model for [B,2,H,W] complex MRI slices."""
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        image_size=320,
        base_channels=64,
        channel_mult: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions: Sequence[int] = (20,),
        dropout=0.0,
        embedding_size=256,
        fourier_scale=16.0,
        scale_by_sigma=True,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.scale_by_sigma = scale_by_sigma
        self.num_res_blocks = int(num_res_blocks)
        self.channel_mult = tuple(channel_mult)
        self.attn_resolutions = set(int(r) for r in attn_resolutions)

        emb_ch = base_channels * 4
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embedding_size, fourier_scale),
            nn.Linear(embedding_size, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = base_channels
        res = self.image_size
        self.skip_ch = [ch]
        for li, mult in enumerate(self.channel_mult):
            out_ch = base_channels * mult
            level = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                level.append(nn.ModuleList([
                    ResBlock(ch, out_ch, emb_ch, dropout),
                    AttnBlock(out_ch) if res in self.attn_resolutions else nn.Identity()
                ]))
                ch = out_ch
                self.skip_ch.append(ch)
            self.down.append(level)
            if li != len(self.channel_mult) - 1:
                self.downsample.append(Down(ch))
                self.skip_ch.append(ch)
                res //= 2
            else:
                self.downsample.append(nn.Identity())

        self.mid1 = ResBlock(ch, ch, emb_ch, dropout)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, emb_ch, dropout)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        skip_ch = list(self.skip_ch)
        for li, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch = base_channels * mult
            level = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                sc = skip_ch.pop()
                level.append(nn.ModuleList([
                    ResBlock(ch + sc, out_ch, emb_ch, dropout),
                    AttnBlock(out_ch) if res in self.attn_resolutions else nn.Identity()
                ]))
                ch = out_ch
            self.up.append(level)
            if li != 0:
                self.upsample.append(Up(ch))
                res *= 2
            else:
                self.upsample.append(nn.Identity())

        self.out_norm = group_norm(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, sigma):
        sigma = sigma.reshape(x.shape[0]).to(x.device).float()
        emb = self.time_embed(torch.log(sigma.clamp_min(1e-12))) # emb = self.time_embed(sigma)
        h = self.in_conv(x)
        skips = [h]
        for level, downsample in zip(self.down, self.downsample):
            for block, attn in level:
                h = attn(block(h, emb))
                skips.append(h)
            h = downsample(h)
            if not isinstance(downsample, nn.Identity):
                skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for level, upsample in zip(self.up, self.upsample):
            for block, attn in level:
                s = skips.pop()
                if h.shape[-2:] != s.shape[-2:]:
                    h = F.interpolate(h, size=s.shape[-2:], mode='nearest')
                h = attn(block(torch.cat([h, s], dim=1), emb))
            h = upsample(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        if self.scale_by_sigma:
            h = h / sigma[:, None, None, None].clamp_min(1e-12)
        return h


def build_model_from_config(cfg):
    m = cfg['model']
    return NCSNpp2D(
        in_channels=m.get('in_channels', 2),
        out_channels=m.get('out_channels', 2),
        image_size=m.get('image_size', cfg['data'].get('crop_size', 320)),
        base_channels=m.get('base_channels', 96),
        channel_mult=tuple(m.get('channel_mult', [1, 2, 2, 4])),
        num_res_blocks=m.get('num_res_blocks', 2),
        attn_resolutions=tuple(m.get('attn_resolutions', [32])),
        dropout=m.get('dropout', 0.0),
        embedding_size=m.get('embedding_size', 256),
        fourier_scale=m.get('fourier_scale', 16.0),
        scale_by_sigma=m.get('scale_by_sigma', True),
    )
