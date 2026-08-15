#!/usr/bin/env python3
"""
Evaluation Script for KLA Semiconductor Image Restoration
==========================================================
Standalone inference script - THE MOST IMPORTANT FILE for benchmarking.

Usage:
    python scripts/evaluate.py \
        --input_dir /path/to/test_degraded \
        --output_dir /path/to/output_restored \
        --weights weights/best.pt \
        --device cuda

Requirements:
- No manual edits needed
- Loads model, runs inference on all images, saves outputs
- Handles both 256→512 (2x) and 128→256 (4x) automatically
- Progress bar, timing stats
- Tiled inference for large images
- FP16 support for speed
"""
import os
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model


class TiledInference:
    """
    Tiled inference for large images to avoid OOM.
    Processes image in overlapping tiles and blends results.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tile_size: int = 256,
        overlap: int = 32,
        device: str = 'cuda',
        scale: int = 2
    ):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.scale = scale
        self.stride = tile_size - overlap
        # Match model parameter dtype (model may have been converted via .half())
        self.dtype = next(model.parameters()).dtype

    def __call__(self, img_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_lr: [1, 1, H, W] low-res input tensor (normalized to [-1, 1])
        Returns:
            [1, 1, H*scale, W*scale] high-res output tensor (normalized to [-1, 1])
        """
        _, _, h, w = img_lr.shape
        h_hr, w_hr = h * self.scale, w * self.scale

        # Output accumulator and weight map for blending
        output = torch.zeros(1, 1, h_hr, w_hr, device=self.device, dtype=self.dtype)
        weight_map = torch.zeros(1, 1, h_hr, w_hr, device=self.device, dtype=self.dtype)

        # Gaussian weight for blending (feather edges)
        weight_kernel = self._gaussian_weight(self.tile_size * self.scale, self.overlap * self.scale)

        # Process tiles
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                # Tile boundaries in LR space
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)

                # Extract tile
                tile_lr = img_lr[:, :, y_start:y_end, x_start:x_end]

                # Pad if needed
                pad_h = self.tile_size - tile_lr.shape[2]
                pad_w = self.tile_size - tile_lr.shape[3]
                if pad_h > 0 or pad_w > 0:
                    tile_lr = F.pad(tile_lr, (0, pad_w, 0, pad_h), mode='reflect')

                # Inference
                tile_lr = tile_lr.to(self.dtype)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    tile_hr = self.model(tile_lr)

                # Remove padding from HR output
                tile_hr = tile_hr[:, :, :self.scale*(y_end-y_start), :self.scale*(x_end-x_start)]

                # Corresponding HR coordinates
                y_hr_start = y_start * self.scale
                x_hr_start = x_start * self.scale
                y_hr_end = y_hr_start + tile_hr.shape[2]
                x_hr_end = x_hr_start + tile_hr.shape[3]

                # Accumulate with blending weight
                # (named wgt: a bare `w` would shadow the image width unpacked above)
                wgt = weight_kernel[:tile_hr.shape[2], :tile_hr.shape[3]].to(tile_hr.dtype)
                output[:, :, y_hr_start:y_hr_end, x_hr_start:x_hr_end] += tile_hr * wgt
                weight_map[:, :, y_hr_start:y_hr_end, x_hr_start:x_hr_end] += wgt

        # Normalize by weight map
        output = output / (weight_map + 1e-8)
        return output

    def _gaussian_weight(self, size: int, overlap: int) -> torch.Tensor:
        """Create 2D Gaussian weight kernel for blending"""
        sigma = size / 6.0
        x = torch.arange(size, device=self.device, dtype=torch.float32) - size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        return gauss_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]


def load_model(weights_path: str, device: str, model_type: str = 'nafnet') -> torch.nn.Module:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(weights_path, map_location=device)

    # Extract config from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = create_model(
            model_type=config.get('model_type', model_type),
            scale=config.get('scale', 2),
            width=config.get('width', 48),
            enc_blks=config.get('enc_blks', [2, 2, 4, 8]),
            middle_blks=config.get('middle_blks', 12),
            dec_blks=config.get('dec_blks', [2, 2, 2, 2])
        )
    else:
        # Default config
        model = create_model(model_type=model_type, scale=2)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path: str, device: str) -> torch.Tensor:
    """Load and preprocess image for inference (supports .npy and image formats)"""
    if img_path.endswith('.npy') or img_path.endswith('.npz'):
        arr = np.load(img_path)
        # Ensure 2D
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        img = arr.astype(np.float32)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = img.astype(np.float32) / 255.0

    # Normalize to [-1, 1]
    img = img * 2 - 1

    # Add batch and channel dimensions: [1, 1, H, W]
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def postprocess_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert output tensor to uint8 image"""
    # [-1, 1] -> [0, 1] -> [0, 255]
    img = tensor.squeeze().float().cpu().numpy()
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def infer_image(
    model: torch.nn.Module,
    img_path: str,
    output_path: str,
    device: str,
    use_tiling: bool = False,
    tile_size: int = 256,
    overlap: int = 32,
    scale: int = 2
) -> float:
    """Run inference on a single image"""
    # Load and preprocess
    img_lr = preprocess_image(img_path, device)

    # Determine scale from input size if not provided
    _, _, h, w = img_lr.shape
    if h == 128 and w == 128:
        scale = 2  # 128 -> 256 (2x)
    elif h == 256 and w == 256:
        scale = 2  # 256 -> 512 (2x)

    # Inference
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.perf_counter()

    if use_tiling and (h > tile_size or w > tile_size):
        tiled_infer = TiledInference(model, tile_size, overlap, device, scale)
        img_hr = tiled_infer(img_lr)
    else:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == 'cuda'):
            img_hr = model(img_lr)

    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.perf_counter() - start_time

    # Postprocess and save
    img_out = postprocess_tensor(img_hr)
    cv2.imwrite(output_path, img_out)

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description='KLA Semiconductor Image Restoration - Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python evaluate.py --input_dir test_degraded --output_dir outputs --weights weights/best.pt

  # With tiling for large images
  python evaluate.py --input_dir test_degraded --output_dir outputs --weights weights/best.pt --tile_size 256 --overlap 32

  # CPU inference
  python evaluate.py --input_dir test_degraded --output_dir outputs --weights weights/best.pt --device cpu
        """
    )
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing degraded test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save restored images')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Inference device')
    parser.add_argument('--model_type', type=str, default='nafnet', choices=['nafnet', 'nafnet_local'], help='Model architecture')
    parser.add_argument('--tile_size', type=int, default=256, help='Tile size for tiled inference (0 to disable)')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between tiles')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 inference (faster on GPU)')
    parser.add_argument('--ext', type=str, default='png', help='Output image extension')
    parser.add_argument('--recursive', action='store_true', help='Search input_dir recursively')

    args = parser.parse_args()

    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    weights_path = Path(args.weights)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {weights_path}...")
    model = load_model(str(weights_path), device, args.model_type)
    print(f"Model loaded: {args.model_type}")

    # Enable FP16 if requested
    if args.fp16 and device == 'cuda':
        model.half()
        print("FP16 inference enabled")

    # Find input images (default to .npy for KLA dataset, but support other formats)
    exts = [args.ext] if args.ext != 'npy' else ['npy', 'npz']
    # Also add image formats
    for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp']:
        if ext not in exts:
            exts.append(ext)

    if args.recursive:
        image_files = []
        for ext in exts:
            image_files.extend(list(input_dir.rglob(f'*.{ext}')))
    else:
        image_files = []
        for ext in exts:
            image_files.extend(list(input_dir.glob(f'*.{ext}')))

    # Also check NoisyLR subdirectory (KLA test structure)
    noisy_lr_dir = input_dir / 'NoisyLR'
    if noisy_lr_dir.exists():
        for ext in exts:
            image_files.extend(list(noisy_lr_dir.glob(f'*.{ext}')))

    image_files = sorted(set(image_files))  # Remove duplicates
    print(f"Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print("No images found! Check input directory and extension.")
        return

    # Determine tiling
    use_tiling = args.tile_size > 0

    # Process images
    print("\nStarting inference...")
    times = []

    for img_path in tqdm(image_files, desc="Processing"):
        # Maintain relative path structure for output
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix(f'.{args.ext}')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            elapsed = infer_image(
                model=model,
                img_path=str(img_path),
                output_path=str(out_path),
                device=device,
                use_tiling=use_tiling,
                tile_size=args.tile_size,
                overlap=args.overlap
            )
            times.append(elapsed)
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

    # Statistics
    if times:
        total_time = sum(times)
        avg_time = total_time / len(times)
        fps = len(times) / total_time

        print(f"\n{'='*50}")
        print(f"Inference Complete!")
        print(f"Processed: {len(times)} images")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {avg_time*1000:.1f}ms/image")
        print(f"Throughput: {fps:.2f} FPS")
        print(f"Min time: {min(times)*1000:.1f}ms")
        print(f"Max time: {max(times)*1000:.1f}ms")
        print(f"Output saved to: {output_dir}")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()