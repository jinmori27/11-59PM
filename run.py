#!/usr/bin/env python3
"""KLA Hackathon entry script.

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy file in <input-dir> (degraded low-res images), restores it,
and writes a float32 .npy of the same filename into <output-dir> with values
clipped to [0, 1]. Target resolution is 2x the input (128->256, 256->512).

Runs on CPU or NVIDIA GPU (auto-detected). No internet access, API keys,
or downloads required at runtime; weights ship inside this repository.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / 'weights' / 'best.pt'
CHUNK_DIR = ROOT / 'weights' / 'chunks'


def reassemble_weights() -> Path:
    """Join sharded weights into weights/best.pt (first run only)."""
    if WEIGHTS.exists():
        return WEIGHTS
    parts = sorted(CHUNK_DIR.glob('best.pt.part-*'))
    if not parts:
        raise FileNotFoundError(
            f"No weights found: neither {WEIGHTS} nor chunks in {CHUNK_DIR}")
    data = b''.join(p.read_bytes() for p in parts)
    WEIGHTS.write_bytes(data)
    return WEIGHTS


def load_model(device: torch.device) -> torch.nn.Module:
    ck = torch.load(reassemble_weights(), map_location='cpu', weights_only=False)
    config = ck.get('config', {})
    # Import inside function to keep startup cheap and avoid heavy deps
    from models.diag_nafnet import DiagNAFNet
    model = DiagNAFNet(
        width=config.get('width', 48),
        enc_blks=config.get('enc_blks', [2, 2, 4, 8]),
        middle_blks=config.get('middle_blks', 12),
        dec_blks=config.get('dec_blks', [2, 2, 2, 2]),
        scale=config.get('scale', 2),
        use_film=config.get('use_film', True),
        use_uncertainty=config.get('use_uncertainty', True),
    )
    model.load_state_dict(ck['model_state_dict'])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('*.npy'))
    if not files:
        print(f"No .npy files found in {input_dir}")
        return 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Files: {len(files)}")
    model = load_model(device)

    # Group by shape so batches are uniform
    by_shape = {}
    for f in files:
        arr = np.load(str(f), mmap_mode='r')
        by_shape.setdefault(arr.shape, []).append(f)

    n_done, t0 = 0, time.time()
    with torch.no_grad():
        for shape, group in sorted(by_shape.items()):
            for i in range(0, len(group), args.batch_size):
                batch = group[i:i + args.batch_size]
                # Normalize to [-1, 1] exactly as during training (no clipping;
                # speckle may push raw values outside [0, 1])
                x = np.stack([np.array(np.load(str(f)), dtype=np.float32) for f in batch])
                x = torch.from_numpy(x).unsqueeze(1).to(device) * 2.0 - 1.0
                y = model(x)
                # [-1, 1] -> [0, 1], grayscale (H, W), clean float32
                out = ((y.squeeze(1).float().cpu().numpy() + 1.0) / 2.0)
                out = np.clip(out, 0.0, 1.0).astype(np.float32)
                for f, img in zip(batch, out):
                    np.save(str(output_dir / f.name), img)
                n_done += len(batch)
                print(f"\r{n_done}/{len(files)}", end='', flush=True)
    print(f"\nDone: {n_done} files in {time.time() - t0:.1f}s -> {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
