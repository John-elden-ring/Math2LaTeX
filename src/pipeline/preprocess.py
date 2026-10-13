"""Page photo preprocessing for Math2LaTeX.

Straightens tilted page photos, removes shadows using adaptive thresholding,
and produces clean binarized images for line segmentation.
"""

from typing import Tuple
import cv2
import numpy as np


def deskew_image(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Detect tilt angle and straighten page photo.
    
    Why:
        Photos taken with phones often have a slight tilt. Straightening lines
        ensures horizontal projection profile cleanly separates equations.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    
    # Otsu thresholding to find prominent ink structures
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find coordinates of all foreground pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return img  # Not enough text to determine angle safely
        
    # Minimum area bounding box around text
    angle = cv2.minAreaRect(coords)[-1]
    
    # Correct OpenCV angle convention
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Limit deskew to reasonable rotation (-max_angle to +max_angle)
    if abs(angle) > max_angle or abs(angle) < 0.2:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


def clean_and_binarize(
    img: np.ndarray,
    block_size: int = 25,
    c: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean shadows and binarize page photo using OpenCV adaptive thresholding.
    
    Args:
        img: Input BGR or Grayscale image.
        block_size: Size of pixel neighborhood for local thresholding.
        c: Constant subtracted from weighted mean.
        
    Returns:
        binary_inv: Inverted binary image where ink is 255 and paper is 0 (for projection).
        cleaned_gray: Normalized grayscale image where shadows have been flattened.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Mild Gaussian blur to suppress camera sensor grain
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive thresholding: handles lighting gradients across paper smoothly
    binary_inv = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )

    # Estimate background illumination using morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    cleaned_gray = cv2.divide(gray, background, scale=255)

    return binary_inv, cleaned_gray


def preprocess_page(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: deskew, flatten lighting, and binarize.
    
    Returns:
        straight_rgb: Straightened color image.
        binary_inv: Inverted binary image for line segmentation.
        cleaned_gray: Flattened grayscale image for line cropping.
    """
    straight_rgb = deskew_image(img)
    binary_inv, cleaned_gray = clean_and_binarize(straight_rgb)
    return straight_rgb, binary_inv, cleaned_gray
