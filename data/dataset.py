"""
Dataset for Semiconductor Image Restoration
Supports paired degraded/ground-truth .npy files with comprehensive augmentations.
Handles 128->256 (2x) super-resolution with speckle+Gaussian noise.
"""
import os
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SemiconductorDataset(Dataset):
    """
    Paired dataset for semiconductor image restoration.
    Expects directory structure:
        root/
            GT/         # High-res clean images (256x256) - .npy files
            NoisyLR/    # Low-res noisy images (128x128) - .npy files
    Filenames must match between GT and NoisyLR.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        patch_size: int = 256,
        scale: int = 2,
        augment: bool = True,
        normalize: bool = True,
        cache: bool = False,
        gt_subdir: str = 'GT',
        lr_subdir: str = 'NoisyLR'
    ):
        """
        Args:
            root_dir: Root directory containing GT/ and NoisyLR/ subdirectories
            split: 'train', 'val', or 'test'
            patch_size: Size of HR patches to extract (for training)
            scale: Upscale factor (2 for 128->256)
            augment: Whether to apply augmentations
            normalize: Whether to normalize to [-1, 1]
            cache: Whether to cache images in memory
            gt_subdir: Ground truth subdirectory name
            lr_subdir: Low-res degraded subdirectory name
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        self.cache = cache

        self.gt_dir = self.root_dir / gt_subdir
        self.lr_dir = self.root_dir / lr_subdir

        # Find paired files
        self.pairs = self._find_pairs()
        print(f"[{split}] Found {len(self.pairs)} pairs")

        # Cache
        self._cache = {} if cache else None

        # Build geometric augmentations (flip, rotate - these work on different sizes)
        self.geo_transform = self._build_geo_transforms()

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching LR/GT pairs (.npy files)"""
        pairs = []
        if not self.lr_dir.exists() or not self.gt_dir.exists():
            print(f"Warning: Directories not found: {self.lr_dir}, {self.gt_dir}")
            return pairs

        for lr_file in sorted(self.lr_dir.glob('*.npy')):
            gt_file = self.gt_dir / lr_file.name
            if gt_file.exists():
                pairs.append((lr_file, gt_file))
            else:
                # Try without extension
                for ext in ['.npy', '.npz']:
                    gt_file = self.gt_dir / (lr_file.stem + ext)
                    if gt_file.exists():
                        pairs.append((lr_file, gt_file))
                        break

        return pairs

    def _build_geo_transforms(self) -> Optional[A.Compose]:
        """Build geometric augmentations that work on different sized images"""
        if not self.augment:
            return None

        # These augmentations don't require same image sizes
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),
        ], additional_targets={'lr': 'image'}, is_check_shapes=False)

    def _random_crop_paired(self, lr: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random crop on GT (HR), compute corresponding crop on LR.
        GT: [H, W, 1] or [H, W], LR: [H//scale, W//scale, 1] or [H//scale, W//scale]
        """
        h_gt, w_gt = gt.shape[:2]
        h_lr, w_lr = lr.shape[:2]

        # Random crop coordinates for GT
        if h_gt <= self.patch_size or w_gt <= self.patch_size:
            # No crop needed
            return lr, gt

        y_gt = random.randint(0, h_gt - self.patch_size)
        x_gt = random.randint(0, w_gt - self.patch_size)

        # Corresponding LR coordinates
        y_lr = y_gt // self.scale
        x_lr = x_gt // self.scale
        patch_lr = self.patch_size // self.scale

        # Crop both
        gt_crop = gt[y_gt:y_gt + self.patch_size, x_gt:x_gt + self.patch_size]
        lr_crop = lr[y_lr:y_lr + patch_lr, x_lr:x_lr + patch_lr]

        return lr_crop, gt_crop

    def _apply_photometric_aug(self, lr: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply photometric augmentations to LR (degraded) only"""
        # Speckle noise (multiplicative) - realistic for semiconductor
        if random.random() < 0.5:
            speckle = np.random.randn(*lr.shape[:2]) * random.uniform(0.05, 0.2)
            if lr.ndim == 3:
                speckle = speckle[..., np.newaxis]
            lr = lr * (1 + speckle)

        # Gaussian noise (additive)
        if random.random() < 0.5:
            noise = np.random.randn(*lr.shape[:2]) * random.uniform(0, 25/255)
            if lr.ndim == 3:
                noise = noise[..., np.newaxis]
            lr = lr + noise

        # Intensity scaling and shift (handles range exceedance from speckle)
        # NOTE: We don't clip here - speckle can push values beyond [0,1], model must handle this
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            shift = random.uniform(-0.04, 0.04)
            lr = lr * scale + shift

        # JPEG compression artifacts (simulate sensor/transmission)
        if random.random() < 0.3:
            quality = random.randint(70, 95)
            # Normalize to 0-255 for JPEG, then back
            lr_norm = np.clip(lr, 0, 1)  # Clip for encoding only
            if lr_norm.ndim == 3:
                lr_norm = lr_norm[..., 0]
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode('.jpg', (lr_norm * 255).astype(np.uint8), encode_param)
            lr_decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if lr.ndim == 3:
                lr_decoded = lr_decoded[..., np.newaxis]
            lr = lr_decoded

        # Blur (simulate optical degradation)
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            if lr.ndim == 3:
                lr = cv2.GaussianBlur(lr[..., 0], (ksize, ksize), random.uniform(0.5, 1.5))[..., np.newaxis]
            else:
                lr = cv2.GaussianBlur(lr, (ksize, ksize), random.uniform(0.5, 1.5))

        return lr, gt

    def _load_npy(self, path: Path) -> np.ndarray:
        """Load .npy file as float32"""
        if self._cache is not None and str(path) in self._cache:
            return self._cache[str(path)]

        arr = np.load(str(path))
        # Ensure 2D
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)

        if self._cache is not None:
            self._cache[str(path)] = arr

        return arr

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lr_path, gt_path = self.pairs[idx]

        lr = self._load_npy(lr_path)   # [128, 128]
        gt = self._load_npy(gt_path)   # [256, 256]

        # Add channel dimension: [H, W] -> [H, W, 1]
        lr = lr[..., np.newaxis]
        gt = gt[..., np.newaxis]

        # Apply geometric augmentations (both) - flip, rotate, transpose
        if self.geo_transform:
            augmented = self.geo_transform(image=gt, lr=lr)
            gt = augmented['image']
            lr = augmented['lr']

        # Apply random crop (paired) - GT gets patch_size, LR gets patch_size/scale
        if self.augment:
            lr, gt = self._random_crop_paired(lr, gt)

        # Apply photometric augmentations to LR only
        if self.augment:
            lr, gt = self._apply_photometric_aug(lr, gt)

        # Ensure correct scale relationship: GT should be scale x LR
        h, w = lr.shape[:2]
        target_h, target_w = h * self.scale, w * self.scale

        if gt.shape[0] != target_h or gt.shape[1] != target_w:
            gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Convert to tensor [C, H, W]
        lr_tensor = torch.from_numpy(lr.transpose(2, 0, 1)).float()
        gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).float()

        # Normalize to [-1, 1] for better training
        # NOTE: LR values may exceed [-1,1] due to speckle - this is expected!
        if self.normalize:
            lr_tensor = lr_tensor * 2 - 1
            gt_tensor = gt_tensor * 2 - 1

        return {
            'degraded': lr_tensor,
            'ground_truth': gt_tensor,
            'deg_path': str(lr_path),
            'gt_path': str(gt_path),
            'scale': self.scale
        }


class TestDataset(Dataset):
    """
    Test dataset for inference (degraded .npy images only, no ground truth)
    """
    def __init__(
        self,
        input_dir: str,
        normalize: bool = True,
        subdir: str = 'NoisyLR'
    ):
        self.input_dir = Path(input_dir)
        self.normalize = normalize

        # Check subdirectory
        subdir_path = self.input_dir / subdir
        if subdir_path.exists():
            self.files = sorted(subdir_path.glob('*.npy'))
        else:
            self.files = sorted(self.input_dir.glob('*.npy'))

        print(f"[test] Found {len(self.files)} .npy files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        arr = np.load(str(path))
        # Ensure 2D
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        arr = arr[..., np.newaxis]  # [H, W, 1]

        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()

        if self.normalize:
            tensor = tensor * 2 - 1

        return {
            'degraded': tensor,
            'path': str(path),
            'filename': path.name
        }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    patch_size: int = 256,
    scale: int = 2,
    val_split: float = 0.1,
    cache: bool = False,
    gt_subdir: str = 'GT',
    lr_subdir: str = 'NoisyLR',
    synth_ratio: float = 0.0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for .npy data.

    synth_ratio: fraction of *additional* synthetic OOD pairs mixed into the
    training set (e.g. 0.25 -> 1 synthetic pair per 4 real pairs). Validation
    always stays real-only.
    """
    # Two independent dataset instances so augment settings don't leak between
    # splits (a shared dataset can't have augment on for train and off for val)
    train_dataset = SemiconductorDataset(
        root_dir=data_root,
        split='train',
        patch_size=patch_size,
        scale=scale,
        augment=True,
        cache=cache,
        gt_subdir=gt_subdir,
        lr_subdir=lr_subdir
    )
    val_dataset = SemiconductorDataset(
        root_dir=data_root,
        split='val',
        patch_size=patch_size,
        scale=scale,
        augment=False,
        cache=cache,
        gt_subdir=gt_subdir,
        lr_subdir=lr_subdir
    )

    # Deterministic shuffled split of the file pairs
    all_pairs = list(train_dataset.pairs)
    random.Random(42).shuffle(all_pairs)
    val_size = int(len(all_pairs) * val_split)

    val_dataset.pairs = all_pairs[:val_size]
    train_dataset.pairs = all_pairs[val_size:]
    print(f"Split: {len(train_dataset.pairs)} train, {len(val_dataset.pairs)} val")

    # Mix in the infinite synthetic OOD curriculum (train only)
    if synth_ratio > 0:
        from torch.utils.data import ConcatDataset
        from .ood_synth import SyntheticPairsDataset
        n_synth = int(len(train_dataset.pairs) * synth_ratio)
        synth_dataset = SyntheticPairsDataset(length=n_synth, size=patch_size, scale=scale)
        train_dataset = ConcatDataset([train_dataset, synth_dataset])
        print(f"Synthetic OOD curriculum: +{n_synth} procedural pairs "
              f"(synth_ratio={synth_ratio})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_test_loader(
    input_dir: str,
    batch_size: int = 1,
    num_workers: int = 2,
    subdir: str = 'NoisyLR'
) -> DataLoader:
    """Create test dataloader for inference"""
    dataset = TestDataset(input_dir, subdir=subdir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test with actual data
    data_root = r"C:\Users\vinna\Downloads\train\train"
    dataset = SemiconductorDataset(data_root, patch_size=256, scale=2)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Degraded: {sample['degraded'].shape}, range=[{sample['degraded'].min():.3f}, {sample['degraded'].max():.3f}]")
    print(f"Ground Truth: {sample['ground_truth'].shape}, range=[{sample['ground_truth'].min():.3f}, {sample['ground_truth'].max():.3f}]")

    train_loader, val_loader = create_dataloaders(data_root, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print(f"\nBatch degraded: {batch['degraded'].shape}")
    print(f"Batch ground_truth: {batch['ground_truth'].shape}")