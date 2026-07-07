import math, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ComplexSliceDataset(Dataset):
    def __init__(
        self,
        roots=(
            "/LOCAL/mri_data/Train/_images_vcc",
            "/LOCAL/mri_data/Val/_images_vcc",
        ),
        drop_slices=5,
        fixed_scale=1e-6,
        crop_size=None,
        augment=True,
    ):
        self.files = []
        for root in roots:
            self.files += sorted(Path(root).glob("*.npy"))

        assert len(self.files) > 0, "No .npy files found."

        self.drop_slices = drop_slices
        self.fixed_scale = fixed_scale
        self.crop_size = crop_size
        self.augment = augment

        self.index = []

        for i, f in enumerate(self.files):
            vol = np.load(f, mmap_mode="r")
            assert vol.ndim == 3, f"{f} must have shape [H,W,D], got {vol.shape}"
            H, W, D = vol.shape

            assert np.iscomplexobj(vol), f"{f} is not complex."

            for z in range(drop_slices, D - drop_slices):
                self.index.append((i, z))

        print(f"Found {len(self.files)} volumes.")
        print(f"Using {len(self.index)} axial slices.")

    def __len__(self):
        return len(self.index)

    def _crop(self, x):
        if self.crop_size is None:
            return x

        if isinstance(self.crop_size, int):
            ch = cw = self.crop_size
        else:
            ch, cw = self.crop_size

        _, H, W = x.shape

        if H < ch or W < cw:
            pad_h = max(0, ch - H)
            pad_w = max(0, cw - W)
            x = torch.nn.functional.pad(
                x,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                value=0.0,
            )
            _, H, W = x.shape

        if self.augment:
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
        else:
            top = (H - ch) // 2
            left = (W - cw) // 2

        return x[:, top:top + ch, left:left + cw]

    def _augment(self, x):
        # x: [2,H,W], real/imag

        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])  # left-right flip

        # small integer translation
        if random.random() < 0.5:
            max_shift = 8
            sh = random.randint(-max_shift, max_shift)
            sw = random.randint(-max_shift, max_shift)
            x = torch.roll(x, shifts=(sh, sw), dims=(-2, -1))

        # complex-consistent amplitude + global phase
        amp = random.uniform(0.9, 1.1)
        phi = random.uniform(-math.pi, math.pi)
        c, s = math.cos(phi), math.sin(phi)

        xr, xi = x[0], x[1]
        yr = amp * (c * xr - s * xi)
        yi = amp * (s * xr + c * xi)

        x = torch.stack([yr, yi], dim=0)

        return x

    def __getitem__(self, idx):
        file_idx, z = self.index[idx]
        path = self.files[file_idx]

        vol = np.load(path, mmap_mode="r")          # [H,W,D], complex
        sl = vol[:, :, z] * self.fixed_scale        # [H,W], complex

        x = torch.stack([
            torch.from_numpy(sl.real.copy()),
            torch.from_numpy(sl.imag.copy()),
        ], dim=0).float()                           # [2,H,W]

        x = self._crop(x)

        if self.augment:
            x = self._augment(x)

        return x


def build_loader(
    batch_size=16,
    crop_size=None,
    num_workers=4,
    seed=0,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ds = ComplexSliceDataset(
        roots=(
            "/LOCAL/mri_data/Train/_images_vcc",
            "/LOCAL/mri_data/Val/_images_vcc",
        ),
        drop_slices=5,
        fixed_scale=1e-6,
        crop_size=crop_size,
        augment=True,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    return ds, loader