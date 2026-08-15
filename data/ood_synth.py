"""
Infinite Synthetic OOD Texture Generator
========================================
Procedurally generates ground-truth textures whose statistics differ from the
training distribution (fBm fractals, particle fields, gratings, circuit-like
geometry, speckle fields), then runs them through the physics degradation
pipeline. Because every sample is regenerated from fresh random parameters,
the generator effectively never repeats - an unbounded OOD curriculum that the
problem statement explicitly encourages ("synthetic data generation strategies
to improve out-of-distribution performance").

Used via `SyntheticPairsDataset` or mixed into real training data through
`create_dataloaders(synth_ratio=...)`.
"""
import random
from typing import Tuple

import cv2
import numpy as np
import torch

from .degrade import synthetic_degrade


def _rand_contrast(img: np.ndarray) -> np.ndarray:
    """Stretch a [0,1]-ish float image to a random contrast/brightness window."""
    lo, hi = np.percentile(img, [1, 99])
    if hi - lo < 1e-6:
        img = np.full_like(img, 0.5)
    else:
        img = (img - lo) / (hi - lo)
    img = img * random.uniform(0.4, 1.0) + random.uniform(0.0, 0.6)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def gen_fbm(size: int = 256, octaves: int = None) -> np.ndarray:
    """Fractal Brownian motion texture (multi-scale value noise)."""
    if octaves is None:
        octaves = random.randint(3, 6)
    out = np.zeros((size, size), dtype=np.float32)
    amp_total = 0.0
    for o in range(octaves):
        res = max(2, 4 * (2 ** o))  # coarse to fine
        small = np.random.rand(res, res).astype(np.float32)
        layer = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        amp = 1.0 / (2 ** o)
        out += amp * layer
        amp_total += amp
    return _rand_contrast(out / amp_total)


def gen_particles(size: int = 256) -> np.ndarray:
    """Scattered particles/blobs - SIMS/SEM-like cluster fields."""
    img = np.zeros((size, size), dtype=np.float32)
    n_particles = random.randint(80, 600)
    bg = random.uniform(0.0, 0.3)
    img[:] = bg
    for _ in range(n_particles):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.uniform(1.5, size / 20)
        intensity = np.random.uniform(0.3, 1.0)
        cv2.circle(img, (int(cx), int(cy)), int(r), intensity, -1)
    if random.random() < 0.5:  # sometimes blur clusters into grains
        img = cv2.GaussianBlur(img, (5, 5), random.uniform(0.5, 2.0))
    return _rand_contrast(img)


def gen_grating(size: int = 256) -> np.ndarray:
    """Sinusoidal grating with random orientation, frequency, harmonics."""
    freq = random.uniform(2, 24)  # cycles across the image
    theta = random.uniform(0, np.pi)
    xs = np.linspace(0, 1, size, dtype=np.float32)
    grid = np.outer(np.sin(theta) * xs, np.ones(size)) + np.outer(np.ones(size), np.cos(theta) * xs)
    phase = random.uniform(0, 2 * np.pi)
    img = np.sin(2 * np.pi * freq * grid + phase)
    if random.random() < 0.5:  # square-ish waves via harmonic clipping
        img = np.tanh(img * random.uniform(1.5, 4.0))
    return _rand_contrast((img + 1) / 2)


def gen_circuit(size: int = 256) -> np.ndarray:
    """PCB/interconnect-like random rectangles and wire segments."""
    img = np.zeros((size, size), dtype=np.float32)
    img[:] = random.uniform(0.0, 0.2)
    for _ in range(random.randint(10, 60)):
        if random.random() < 0.5:
            x, y = np.random.randint(0, size, 2)
            w, h = np.random.randint(size // 16, size // 3, 2)
            cv2.rectangle(img, (x, y), (min(x + w, size - 1), min(y + h, size - 1)),
                          np.random.uniform(0.3, 1.0), -1)
        else:
            p1 = tuple(np.random.randint(0, size, 2))
            p2 = tuple(np.random.randint(0, size, 2))
            cv2.line(img, p1, p2, np.random.uniform(0.3, 1.0),
                     random.randint(1, max(2, size // 64)))
    img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 1.0))
    return _rand_contrast(img)


def gen_speckle_field(size: int = 256) -> np.ndarray:
    """Laser-speckle-like field: correlated multiplicative noise."""
    raw = np.random.rand(size, size).astype(np.float32)
    k = random.choice([3, 5, 7, 9])
    img = cv2.GaussianBlur(raw, (k, k), random.uniform(0.5, 2.5))
    img = img * np.random.rand(size, size).astype(np.float32)  # re-modulate
    return _rand_contrast(img)


GENERATORS = {
    'fbm': gen_fbm,
    'particles': gen_particles,
    'grating': gen_grating,
    'circuit': gen_circuit,
    'speckle_field': gen_speckle_field,
}


def generate_texture(size: int = 256, kind: str = None) -> np.ndarray:
    """Generate one random clean texture in [0, 1], shape (size, size)."""
    if kind is None:
        kind = random.choice(list(GENERATORS.keys()))
    return GENERATORS[kind](size)


def generate_pair(
    size: int = 256,
    scale: int = 2,
    kind: str = None,
    degradation_severity: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one (degraded_LR, GT) pair as float32 arrays.

    degradation_severity: None -> random per-sample severity curriculum
    ('low' | 'medium' | 'high' | 'extreme' passthrough to synthetic_degrade).
    """
    gt = generate_texture(size, kind)
    if degradation_severity is None:
        degradation_severity = random.choice(['low', 'medium', 'medium', 'high', 'extreme'])
    # Randomize the degrade knobs within the chosen severity band
    bands = {
        'low':     dict(speckle_sigma=(0.03, 0.10), gaussian_sigma=(4 / 255, 10 / 255), blur_sigma=(0.3, 0.7)),
        'medium':  dict(speckle_sigma=(0.08, 0.20), gaussian_sigma=(8 / 255, 20 / 255), blur_sigma=(0.5, 1.0)),
        'high':    dict(speckle_sigma=(0.15, 0.30), gaussian_sigma=(15 / 255, 30 / 255), blur_sigma=(0.8, 1.4)),
        'extreme': dict(speckle_sigma=(0.25, 0.45), gaussian_sigma=(25 / 255, 45 / 255), blur_sigma=(1.0, 1.8)),
    }[degradation_severity]
    lr = synthetic_degrade(
        gt,
        scale=scale,
        speckle_sigma=random.uniform(*bands['speckle_sigma']),
        gaussian_sigma=random.uniform(*bands['gaussian_sigma']),
        blur_sigma=random.uniform(*bands['blur_sigma']),
        downsample_method=random.choice(['bicubic', 'bilinear', 'area']),
        jpeg_quality=random.choice([None, None, 90, 80, 70]),
        clip_range=(-1.0, 2.0),  # speckle may exceed [0,1]; don't hide it
    )
    return lr, gt


class SyntheticPairsDataset:
    """Duck-typed dataset matching SemiconductorDataset's item format.

    `length` virtual items; each __getitem__ generates a brand-new random pair.
    """
    def __init__(self, length: int = 4000, size: int = 256, scale: int = 2, normalize: bool = True):
        self.length = length
        self.size = size
        self.scale = scale
        self.normalize = normalize
        self.augment = False  # generation is already fully random

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict:
        lr, gt = generate_pair(size=self.size, scale=self.scale)
        lr_t = torch.from_numpy(lr[..., np.newaxis].transpose(2, 0, 1)).float()
        gt_t = torch.from_numpy(gt[..., np.newaxis].transpose(2, 0, 1)).float()
        if self.normalize:
            lr_t = lr_t * 2 - 1
            gt_t = gt_t * 2 - 1
        return {
            'degraded': lr_t,
            'ground_truth': gt_t,
            'deg_path': f'synth://{idx}',
            'gt_path': f'synth://{idx}',
            'scale': self.scale,
        }


if __name__ == '__main__':
    for kind in GENERATORS:
        lr, gt = generate_pair(kind=kind)
        print(f"{kind:14s} GT {gt.shape} [{gt.min():.2f},{gt.max():.2f}] | "
              f"LR {lr.shape} [{lr.min():.2f},{lr.max():.2f}]")
