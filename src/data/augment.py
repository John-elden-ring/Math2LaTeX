"""Image augmentation pipeline for Math2LaTeX.

Transforms clean synthetic/vector strokes into realistic photo-like handwritten note crops:
- Subtle rotation (up to 3 degrees) and perspective tilt
- Line thickness variation (dilation / erosion)
- Paper textures and subtle notebook ruled lines
- Uneven lighting and phone camera shadow gradients
- Blue ballpoint and dark ink simulation
- Sensor noise, camera blur, and JPEG compression artifacts

Note: Augmentations are applied during training only.
Validation and testing use clean, unaltered images.
"""

import io
import math
import random
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def random_rotation(img: np.ndarray, max_angle: float = 3.0) -> np.ndarray:
    """Rotate image by a small random angle between -max_angle and +max_angle degrees."""
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    # Border value is white (255)
    border_val = (255, 255, 255) if img.ndim == 3 else 255
    rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return rotated


def random_stroke_morphology(img_gray: np.ndarray, prob: float = 0.4) -> np.ndarray:
    """Thicken or thin pen strokes via mathematical morphology (dilation / erosion)."""
    if random.random() > prob:
        return img_gray

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # In an image where ink is black (0) and paper is white (255):
    # Erode shrinks white regions -> thickens dark ink
    # Dilate grows white regions -> thins dark ink
    if random.random() < 0.5:
        # Thicker strokes
        return cv2.erode(img_gray, kernel, iterations=1)
    else:
        # Thinner strokes
        return cv2.dilate(img_gray, kernel, iterations=1)


def add_notebook_lines(img_rgb: np.ndarray, prob: float = 0.3) -> np.ndarray:
    """Add faint ruled horizontal notebook lines behind handwriting."""
    if random.random() > prob:
        return img_rgb

    h, w, _ = img_rgb.shape
    line_spacing = random.randint(28, 42)
    offset = random.randint(5, line_spacing)

    # Line color: faint pale blue or light graphite
    if random.random() < 0.6:
        line_color = np.array([210, 215, 235], dtype=np.float32)  # faint blue
    else:
        line_color = np.array([220, 220, 220], dtype=np.float32)  # faint gray

    out = img_rgb.astype(np.float32).copy()
    for y in range(offset, h, line_spacing):
        # Blend line onto paper background (where pixel is light)
        mask = (out[y, :, 0] > 180) & (out[y, :, 1] > 180) & (out[y, :, 2] > 180)
        out[y, mask, :] = 0.75 * out[y, mask, :] + 0.25 * line_color

    return np.clip(out, 0, 255).astype(np.uint8)


def add_uneven_lighting(img_rgb: np.ndarray, prob: float = 0.4) -> np.ndarray:
    """Simulate uneven phone camera lighting and cast shadows."""
    if random.random() > prob:
        return img_rgb

    h, w, _ = img_rgb.shape
    # Create smooth 2D linear gradient
    theta = random.uniform(0, 2 * math.pi)
    x_coords = np.linspace(-1, 1, w)
    y_coords = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Directional gradient across the image
    gradient = xx * math.cos(theta) + yy * math.sin(theta)
    # Normalize between 0.75 (shadow) and 1.05 (highlight)
    light_factor = 0.88 + 0.17 * gradient
    light_factor = np.clip(light_factor, 0.70, 1.10)[:, :, np.newaxis]

    out = (img_rgb.astype(np.float32) * light_factor)
    return np.clip(out, 0, 255).astype(np.uint8)


def simulate_ink_and_paper(img_gray: np.ndarray, blue_ink_prob: float = 0.35) -> np.ndarray:
    """Convert binary black-on-white image into realistic paper and ink RGB colors."""
    h, w = img_gray.shape
    normalized_ink = img_gray.astype(np.float32) / 255.0  # 0.0 is ink, 1.0 is paper

    # Paper color: off-white / light cream / slight notebook tint
    paper_r = random.uniform(240, 253)
    paper_g = random.uniform(238, 252)
    paper_b = random.uniform(230, 248)
    paper_color = np.array([paper_r, paper_g, paper_b], dtype=np.float32)

    # Ink color: either blue ballpoint or dark charcoal/black
    if random.random() < blue_ink_prob:
        # Blue ballpoint ink
        ink_r = random.uniform(20, 50)
        ink_g = random.uniform(35, 75)
        ink_b = random.uniform(110, 170)
    else:
        # Black / graphite ink
        val = random.uniform(20, 55)
        ink_r, ink_g, ink_b = val, val, val
    ink_color = np.array([ink_r, ink_g, ink_b], dtype=np.float32)

    # Linear interpolation based on normalized_ink
    # pixel = ink * (1 - normalized_ink) + paper * normalized_ink
    t = normalized_ink[:, :, np.newaxis]
    rgb = (1.0 - t) * ink_color + t * paper_color

    return np.clip(rgb, 0, 255).astype(np.uint8)


def add_blur_and_noise(img_rgb: np.ndarray, blur_prob: float = 0.3, noise_prob: float = 0.3) -> np.ndarray:
    """Simulate camera lens blur and sensor noise."""
    out = img_rgb.copy()

    # Gaussian blur
    if random.random() < blur_prob:
        ksize = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=random.uniform(0.5, 1.2))

    # Sensor noise
    if random.random() < noise_prob:
        sigma = random.uniform(3.0, 8.0)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def simulate_jpeg_compression(img_rgb: np.ndarray, prob: float = 0.35) -> np.ndarray:
    """Simulate phone JPEG compression artifacts."""
    if random.random() > prob:
        return img_rgb

    quality = random.randint(45, 80)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decompressed = Image.open(buffer)
    return np.array(decompressed)


def augment_image(
    img_gray: np.ndarray,
    is_training: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Full end-to-end image processing & augmentation pipeline.
    
    Args:
        img_gray: Grayscale image (H, W) with pixel values 0-255.
        is_training: If True, apply probabilistic photographic & geometric augmentations.
                     If False, cleanly convert to 3-channel RGB without noise.
        config: Optional configuration dictionary.
        
    Returns:
        RGB image of shape (H, W, 3), dtype uint8.
    """
    if not is_training:
        # Deterministic conversion: 3 identical channels
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    cfg = config or {}
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # 1. Stroke morphology on grayscale ink
    out_gray = random_stroke_morphology(img_gray, prob=0.4)

    # 2. Geometric rotation
    rot_deg = float(aug_cfg.get("rotation_degrees", 3.0))
    out_gray = random_rotation(out_gray, max_angle=rot_deg)

    # 3. Simulate realistic paper texture & ink color (creates RGB)
    photo_cfg = aug_cfg.get("photo_effects", {})
    blue_prob = float(photo_cfg.get("blue_ink_prob", 0.35))
    out_rgb = simulate_ink_and_paper(out_gray, blue_ink_prob=blue_prob)

    # 4. Add subtle notebook ruled lines
    lines_prob = float(photo_cfg.get("ruled_lines_prob", 0.3))
    out_rgb = add_notebook_lines(out_rgb, prob=lines_prob)

    # 5. Add uneven lighting / phone camera shadows
    shadow_prob = float(photo_cfg.get("shadow_prob", 0.4))
    out_rgb = add_uneven_lighting(out_rgb, prob=shadow_prob)

    # 6. Camera blur and sensor noise
    blur_prob = float(aug_cfg.get("blur_prob", 0.3))
    noise_prob = float(aug_cfg.get("noise_prob", 0.3))
    out_rgb = add_blur_and_noise(out_rgb, blur_prob=blur_prob, noise_prob=noise_prob)

    # 7. Phone camera JPEG compression
    out_rgb = simulate_jpeg_compression(out_rgb, prob=0.3)

    return out_rgb
