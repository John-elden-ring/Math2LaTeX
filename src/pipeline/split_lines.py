"""Horizontal projection line segmentation for handwritten math notes.

Finds and crops mathematical equation lines from page photos:
- Horizontal projection profile (ink count per row)
- Dynamic gap detection and line bounding
- Aspect-ratio preserving crop extraction (height 128px)
"""

from typing import List, NamedTuple, Optional, Tuple
import cv2
import numpy as np


class LineCrop(NamedTuple):
    """Represents an extracted equation line from a page."""
    index: int
    image: np.ndarray             # Cropped image (fixed height 128px)
    bbox: Tuple[int, int, int, int]  # (y1, y2, x1, x2) in original image


def smooth_projection(profile: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Smooth 1D projection profile with a moving average kernel."""
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(profile, kernel, mode="same")


def find_line_intervals(
    binary_inv: np.ndarray,
    min_line_height: int = 20,
    min_line_gap: int = 15,
    ink_threshold: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """Find vertical (y1, y2) bounds of lines using horizontal projection profile."""
    h, w = binary_inv.shape[:2]
    # Count number of ink pixels (non-zero) per row
    raw_profile = np.sum(binary_inv > 0, axis=1).astype(np.float32)
    smoothed = smooth_projection(raw_profile, kernel_size=9)

    # Adaptive threshold: average ink per row * factor, or small baseline
    thresh = ink_threshold if ink_threshold is not None else max(float(np.mean(smoothed)) * 0.15, 5.0)

    is_ink = smoothed > thresh

    # Extract contiguous bands
    intervals: List[Tuple[int, int]] = []
    in_line = False
    start_y = 0

    for y in range(h):
        if is_ink[y] and not in_line:
            in_line = True
            start_y = y
        elif not is_ink[y] and in_line:
            in_line = False
            intervals.append((start_y, y))

    if in_line:
        intervals.append((start_y, h - 1))

    # Merge intervals that are closer than min_line_gap
    merged: List[Tuple[int, int]] = []
    for cur in intervals:
        if not merged:
            merged.append(cur)
            continue
        prev_y1, prev_y2 = merged[-1]
        cur_y1, cur_y2 = cur
        if cur_y1 - prev_y2 < min_line_gap:
            merged[-1] = (prev_y1, cur_y2)
        else:
            merged.append(cur)

    # Filter out intervals shorter than min_line_height
    valid = [(y1, y2) for (y1, y2) in merged if (y2 - y1) >= min_line_height]
    return valid


def find_horizontal_bounds(
    binary_slice: np.ndarray,
    padding: int = 15,
) -> Tuple[int, int]:
    """Find horizontal (x1, x2) bounds of ink inside a line slice."""
    w = binary_slice.shape[1]
    col_ink = np.sum(binary_slice > 0, axis=0)
    non_zero = np.where(col_ink > 2)[0]

    if len(non_zero) == 0:
        return 0, w

    x1 = max(0, int(non_zero[0]) - padding)
    x2 = min(w, int(non_zero[-1]) + padding)
    return x1, x2


def resize_to_standard_height(img: np.ndarray, target_height: int = 128) -> np.ndarray:
    """Resize image to fixed height 128px while preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_height, 64), 255, dtype=np.uint8)
    scale = target_height / float(h)
    new_w = max(int(round(w * scale)), 32)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def split_page_into_lines(
    page_img: np.ndarray,
    binary_inv: Optional[np.ndarray] = None,
    min_line_height: int = 20,
    min_line_gap: int = 15,
    target_height: int = 128,
) -> List[LineCrop]:
    """Segment full page image into an ordered list of equation line crops.
    
    Args:
        page_img: Cleaned grayscale or RGB page image.
        binary_inv: Inverted binary image (ink=255, paper=0).
        min_line_height: Minimum pixel height to qualify as an equation line.
        min_line_gap: Gaps smaller than this are merged into one line.
        target_height: Normalized output height (128px per design doc).
        
    Returns:
        List of LineCrop objects ordered from top of page to bottom.
    """
    if binary_inv is None:
        gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY) if page_img.ndim == 3 else page_img
        binary_inv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )

    intervals = find_line_intervals(
        binary_inv, min_line_height=min_line_height, min_line_gap=min_line_gap
    )

    h_page, w_page = page_img.shape[:2]
    crops: List[LineCrop] = []

    for idx, (y1, y2) in enumerate(intervals):
        # Vertical padding
        pad_y = 6
        pad_y1 = max(0, y1 - pad_y)
        pad_y2 = min(h_page, y2 + pad_y)

        # Horizontal bounds
        slice_bin = binary_inv[pad_y1:pad_y2, :]
        x1, x2 = find_horizontal_bounds(slice_bin, padding=15)

        # Crop from page image
        raw_crop = page_img[pad_y1:pad_y2, x1:x2]
        standard_crop = resize_to_standard_height(raw_crop, target_height=target_height)

        crops.append(LineCrop(
            index=idx,
            image=standard_crop,
            bbox=(pad_y1, pad_y2, x1, x2),
        ))

    return crops
