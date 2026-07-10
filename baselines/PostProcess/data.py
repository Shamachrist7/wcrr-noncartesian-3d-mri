import os, warnings, random, math
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cupy as cp

from mrinufft import get_operator
from mrinufft.io import read_trajectory
from mrinufft.extras.smaps import get_smaps
from utils import MRINUFFTPhysicsRI, complex_to_ri, _load_volumes


def ensure_ri_batched(x):
    # returns [1,2,H,W,D]
    if x.ndim == 4:      # [2,H,W,D]
        x = x[None]
    assert x.ndim == 5 and x.shape[1] == 2, f"Expected [1,2,H,W,D], got {x.shape}"
    return x


def phase_rotate_ri(x, phi):
    # x: [2,H,W,D]
    c, s = math.cos(phi), math.sin(phi)
    xr, xi = x[0], x[1]
    return torch.stack([c * xr - s * xi, s * xr + c * xi], dim=0)


def smooth_bias(shape, strength=0.10):
    # shape: [H,W,D]
    b = torch.randn(1, 1, 4, 4, 4)
    b = F.interpolate(b, size=shape, mode="trilinear", align_corners=False)[0, 0]
    b = (b - b.mean()) / (b.std() + 1e-8)
    return torch.exp(strength * b).clamp(0.85, 1.20)


def augment_pair_ri(zf, gt, input_noise=0.0):
    # zf, gt: [2,H,W,D]

    for dim in (1, 2, 3):
        if random.random() < 0.5:
            zf, gt = torch.flip(zf, [dim]), torch.flip(gt, [dim])

    if random.random() < 0.5:
        shifts = [random.randint(-3, 3) for _ in range(3)]
        zf, gt = torch.roll(zf, shifts, dims=(1, 2, 3)), torch.roll(gt, shifts, dims=(1, 2, 3))

    a = random.uniform(0.9, 1.1)
    zf, gt = a * zf, a * gt

    if random.random() < 0.5:
        phi = random.uniform(-math.pi, math.pi)
        zf, gt = phase_rotate_ri(zf, phi), phase_rotate_ri(gt, phi)

    if random.random() < 0.3:
        b = smooth_bias(zf.shape[-3:]).to(zf.device, zf.dtype)
        zf, gt = zf * b[None], gt * b[None]

    if input_noise > 0:
        zf = zf + random.uniform(0.0, input_noise) * torch.randn_like(zf)

    return zf.contiguous(), gt.contiguous()


class ZFToGTDataset(Dataset):
    def __init__(
        self,
        roots=("/LOCAL/mri_data/Train", "/LOCAL/mri_data/Val"),
        traj_name="gs.bin",
        scaler=1e-6,
        noise_level=2e-3,
        backend="cufinufft",
        augment=True,
        cache=True,
    ):
        self.files = []
        for r in roots:
            self.files += sorted(Path(r).rglob("*.h5"))
        assert self.files, "No .h5 files found."

        traj, p = read_trajectory(traj_name, dwell_time=0.01 / 2)
        traj = np.clip(traj.copy(), -0.5, 0.5)
        self.kspace_loc = traj.reshape(-1, p["dimension"])

        self.scaler = scaler
        self.noise_level = noise_level
        self.backend = backend
        self.augment = augment
        self.cache = {} if cache else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Found {len(self.files)} volumes. Cache: {cache}. Augment: {augment}.")

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def _make_pair(self, path):
        print(f"Computing ZF/GT: {path}")

        x = self.scaler * np.moveaxis(_load_volumes(path), -1, 0)  # [C,H,W,D]
        x = torch.from_numpy(x).to(torch.complex64)
        coils = x.shape[0]

        F_raw = get_operator(self.backend)(
            self.kspace_loc,
            x.shape[1:],
            n_coils=coils,
            density=True,
            squeeze_dims=True,
        )

        y = F_raw.op(x)
        y = y + self.noise_level * torch.randn_like(y)

        Smaps = get_smaps("espirit")(
            self.kspace_loc,
            x.shape[1:],
            kspace_data=cp.asarray(y),
            density=F_raw.density,
            backend=self.backend,
            decim=4,
        )

        E = get_operator(self.backend)(
            self.kspace_loc,
            x.shape[1:],
            n_coils=coils,
            smaps=Smaps.get(),
            squeeze_dims=True,
        )

        physics = MRINUFFTPhysicsRI(E)

        zf = physics.A_adjoint(y.to(self.device))          # [1,2,H,W,D]
        zf = ensure_ri_batched(zf).detach().cpu().float()

        smaps = torch.as_tensor(np.asarray(E.smaps)).to(torch.complex64)
        gt = torch.sum(torch.conj(smaps) * x, dim=0)       # [H,W,D], complex
        gt = ensure_ri_batched(complex_to_ri(gt)).detach().cpu().float()

        cp.get_default_memory_pool().free_all_blocks()
        return zf, gt

    def __getitem__(self, i):
        path = str(self.files[i])

        if self.cache is not None and path in self.cache:
            zf, gt = self.cache[path]
        else:
            zf, gt = self._make_pair(path)
            if self.cache is not None:
                self.cache[path] = (zf, gt)

        zf, gt = zf[0].clone(), gt[0].clone()  # [2,H,W,D]

        if self.augment:
            zf, gt = augment_pair_ri(zf, gt, input_noise=1e-4)

        return zf, gt


def build_loader(batch_size=1, shuffle=True, cache=True, augment=False):
    ds = ZFToGTDataset(cache=cache, augment=augment)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return ds, loader