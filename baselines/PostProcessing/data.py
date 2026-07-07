import os, warnings, random
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
from utils import MRINUFFTPhysicsRI, ri_to_complex, _load_volumes


def smooth_bias(shape, strength=0.15):
    b = torch.randn(1, 1, 4, 4, 4)
    b = F.interpolate(b, size=shape, mode="trilinear", align_corners=False)[0]
    b = (b - b.mean()) / (b.std() + 1e-8)
    return torch.exp(strength * b).clamp(0.8, 1.25)  # [1,H,W,D]


def augment_pair(zf, gt, input_noise=0.01):
    # zf, gt: [1,D,H,W]

    for dim in [-1, -2]:  # W, H flips
        if random.random() < 0.5:
            zf, gt = torch.flip(zf, [dim]), torch.flip(gt, [dim])

    if random.random() < 0.3:  # optional z flip
        zf, gt = torch.flip(zf, [-3]), torch.flip(gt, [-3])

    a = random.uniform(0.9, 1.1)
    zf, gt = a * zf, a * gt

    if random.random() < 0.5:
        b = smooth_bias(zf.shape[-3:])
        zf, gt = zf * b, gt * b

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

        assert len(self.files), "No .h5 files found."

        traj, p = read_trajectory(traj_name, dwell_time=0.01 / 2)
        traj = np.clip(traj.copy(), -0.5, 0.5)
        self.kspace_loc = traj.reshape(-1, p["dimension"])

        self.scaler = scaler
        self.noise_level = noise_level
        self.backend = backend
        self.augment = augment
        self.cache = {} if cache else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Found {len(self.files)} volumes.")

    def __len__(self):
        return len(self.files)

    @torch.no_grad()
    def _make_pair(self, path):
        x = self.scaler * np.moveaxis(_load_volumes(str(path)), -1, 0)  # [C,H,W,D]
        x = torch.from_numpy(x).to(torch.complex64)
        coils = x.shape[0]

        F_raw = get_operator(self.backend)(
            self.kspace_loc, x.shape[1:], n_coils=coils,
            density=True, squeeze_dims=True
        )

        y = F_raw.op(x)
        y = y + self.noise_level * torch.randn_like(y)

        Smaps = get_smaps("espirit")(
            self.kspace_loc, x.shape[1:], kspace_data=cp.asarray(y),
            density=F_raw.density, backend=self.backend, decim=4
        )

        E = get_operator(self.backend)(
            self.kspace_loc, x.shape[1:], n_coils=coils,
            smaps=Smaps.get(), squeeze_dims=True
        )

        physics = MRINUFFTPhysicsRI(E)

        zf = ri_to_complex(physics.A_adjoint(y.to(self.device))).abs().cpu()
        zf = zf[None]  # [1,H,W,D]

        smaps = torch.as_tensor(np.asarray(E.smaps)).to(torch.complex64)
        gt = torch.sum(torch.conj(smaps) * x, dim=0).abs()
        gt = gt[None]  # [1,H,W,D]

        cp.get_default_memory_pool().free_all_blocks()
        return zf, gt

    def __getitem__(self, i):
        path = self.files[i]

        if self.cache is not None and path in self.cache:
            zf, gt = self.cache[path]
        else:
            zf, gt = self._make_pair(path)
            if self.cache is not None:
                self.cache[path] = (zf, gt)

        zf, gt = zf.clone(), gt.clone()

        if self.augment:
            zf, gt = augment_pair(zf, gt)

        return zf, gt


def build_loader(batch_size=1, shuffle=True, cache=True):
    ds = ZFToGTDataset(cache=cache, augment=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      # keep 0 because MRINUFFT/CuPy/GPU inside dataset
        pin_memory=True,
        drop_last=False,
    )
    return ds, loader