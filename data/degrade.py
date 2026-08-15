"""
Synthetic Degradation Pipeline for Semiconductor Images
Generates realistic degraded images from clean ground truth for development/testing.
"""
import cv2
import numpy as np
import random
from typing import Tuple, Optional


def add_speckle_noise(
    image: np.ndarray,
    sigma: float = 0.1,
    mean: float = 0.0
) -> np.ndarray:
    """
    Add multiplicative speckle noise.
    Speckle pushes pixel values beyond original range (realistic for semiconductor).
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image * (1 + noise)
    return noisy.astype(np.float32)


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 25.0 / 255.0
) -> np.ndarray:
    """Add additive Gaussian noise"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return noisy.astype(np.float32)


def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Add Poisson (shot) noise"""
    # Scale to photon counts, add Poisson, scale back
    photons = image * scale
    noisy_photons = np.random.poisson(photons)
    return (noisy_photons / scale).astype(np.float32)


def downsample(
    image: np.ndarray,
    scale: int,
    method: str = 'bicubic'
) -> np.ndarray:
    """
    Downsample image by scale factor.
    Methods: bicubic, bilinear, nearest, area
    """
    h, w = image.shape[:2]
    new_h, new_w = h // scale, w // scale

    interp_map = {
        'bicubic': cv2.INTER_CUBIC,
        'bilinear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
        'area': cv2.INTER_AREA
    }
    interp = interp_map.get(method, cv2.INTER_CUBIC)

    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def upsample(
    image: np.ndarray,
    scale: int,
    method: str = 'bicubic'
) -> np.ndarray:
    """Upsample image by scale factor"""
    h, w = image.shape[:2]
    new_h, new_w = h * scale, w * scale

    interp_map = {
        'bicubic': cv2.INTER_CUBIC,
        'bilinear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST
    }
    interp = interp_map.get(method, cv2.INTER_CUBIC)

    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def apply_blur(
    image: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0
) -> np.ndarray:
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def add_jpeg_artifacts(image: np.ndarray, quality: int = 85) -> np.ndarray:
    """Add JPEG compression artifacts"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', (image * 255).astype(np.uint8), encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def synthetic_degrade(
    gt: np.ndarray,
    scale: int = 2,
    speckle_sigma: float = 0.15,
    gaussian_sigma: float = 15.0 / 255.0,
    blur_sigma: float = 0.8,
    downsample_method: str = 'bicubic',
    jpeg_quality: Optional[int] = None,
    intensity_scale: Tuple[float, float] = (0.9, 1.1),
    intensity_shift: Tuple[float, float] = (-0.02, 0.02),
    clip_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Full degradation pipeline: GT -> degraded
    Simulates realistic semiconductor inspection degradation.
    """
    img = gt.copy().astype(np.float32)

    # 1. Optical blur (before downsampling)
    if blur_sigma > 0:
        ksize = max(3, int(blur_sigma * 3) // 2 * 2 + 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    # 2. Downsample
    img = downsample(img, scale, downsample_method)

    # 3. Speckle noise (multiplicative) - can push beyond range
    if speckle_sigma > 0:
        img = add_speckle_noise(img, speckle_sigma)

    # 4. Gaussian noise (additive)
    if gaussian_sigma > 0:
        img = add_gaussian_noise(img, gaussian_sigma)

    # 5. Intensity scaling/shift (simulates sensor gain/offset variations)
    scale_factor = random.uniform(*intensity_scale)
    shift = random.uniform(*intensity_shift)
    img = img * scale_factor + shift

    # 6. JPEG artifacts (optional)
    if jpeg_quality is not None:
        img = add_jpeg_artifacts(img, jpeg_quality)

    # 7. Final clipping
    img = np.clip(img, *clip_range)

    return img.astype(np.float32)


def create_degradation_pipeline(
    scale: int = 2,
    noise_level: str = 'medium'
) -> callable:
    """
    Create a degradation function with preset noise levels.
    noise_level: 'low', 'medium', 'high', 'extreme'
    """
    presets = {
        'low': {
            'speckle_sigma': 0.05,
            'gaussian_sigma': 5.0 / 255.0,
            'blur_sigma': 0.5,
        },
        'medium': {
            'speckle_sigma': 0.15,
            'gaussian_sigma': 15.0 / 255.0,
            'blur_sigma': 0.8,
        },
        'high': {
            'speckle_sigma': 0.25,
            'gaussian_sigma': 25.0 / 255.0,
            'blur_sigma': 1.2,
        },
        'extreme': {
            'speckle_sigma': 0.4,
            'gaussian_sigma': 40.0 / 255.0,
            'blur_sigma': 1.5,
        }
    }

    params = presets.get(noise_level, presets['medium'])
    params['scale'] = scale

    def degrade_fn(gt):
        return synthetic_degrade(gt, **params)

    return degrade_fn


def batch_degrade(
    gt_batch: np.ndarray,
    scale: int = 2,
    noise_level: str = 'medium'
) -> np.ndarray:
    """Apply degradation to batch of images"""
    degrade_fn = create_degradation_pipeline(scale, noise_level)
    return np.stack([degrade_fn(img) for img in gt_batch])


if __name__ == "__main__":
    # Test degradation pipeline
    import matplotlib.pyplot as plt

    # Create synthetic ground truth (semiconductor-like pattern)
    gt = np.zeros((512, 512), dtype=np.float32)
    # Add some lines and shapes
    cv2.line(gt, (0, 256), (512, 256), 1.0, 2)
    cv2.line(gt, (256, 0), (256, 512), 1.0, 2)
    cv2.rectangle(gt, (100, 100), (200, 200), 0.8, -1)
    cv2.circle(gt, (400, 400), 50, 0.6, -1)
    gt = cv2.GaussianBlur(gt, (5, 5), 1.0)

    # Test different scales
    for scale in [2, 4]:
        for noise in ['low', 'medium', 'high', 'extreme']:
            deg = create_degradation_pipeline(scale, noise)(gt)
            print(f"Scale {scale}x, {noise}: GT_range=[{gt.min():.3f}, {gt.max():.3f}], "
                  f"Deg_range=[{deg.min():.3f}, {deg.max():.3f}], "
                  f"Deg_shape={deg.shape}")