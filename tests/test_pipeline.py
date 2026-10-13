"""Unit tests for Stage 6, 7, and 8: Pipeline, Metrics, Beam Search, and Segmentation."""

import numpy as np
import pytest
import cv2

from src.model.evaluate import (
    levenshtein_distance,
    compute_normalized_edit_distance,
    is_exact_match,
    check_latex_compilable,
)
from src.pipeline.preprocess import deskew_image, clean_and_binarize
from src.pipeline.split_lines import split_page_into_lines, find_line_intervals
from src.pipeline.page_to_tex import format_latex_document


def test_edit_distance_metrics():
    # Exact same
    assert levenshtein_distance("x+y=z", "x+y=z") == 0
    assert compute_normalized_edit_distance("x+y=z", "x+y=z") == 0.0

    # 1 insertion / deletion / substitution
    assert levenshtein_distance("kitten", "sitting") == 3
    assert compute_normalized_edit_distance("cat", "cut") == 1.0 / 3.0

    # Empty string
    assert levenshtein_distance("", "abc") == 3
    assert compute_normalized_edit_distance("", "abc") == 1.0


def test_latex_compilable_check():
    # Valid syntax
    assert check_latex_compilable(r"\frac{1}{2}") is True
    assert check_latex_compilable(r"\sum_{i=1}^{n} x_i") is True
    assert check_latex_compilable(r"\left( a + b \right)") is True

    # Invalid syntax (unbalanced braces, unclosed left)
    assert check_latex_compilable(r"\frac{1}{2") is False
    assert check_latex_compilable(r"\left( a + b") is False
    assert check_latex_compilable("x + \\") is False


def test_deskew_and_preprocess():
    # Create simple synthetic test image
    img = np.full((200, 300), 255, dtype=np.uint8)
    # Draw horizontal line
    img[100:105, 50:250] = 0

    deskewed = deskew_image(img)
    assert deskewed.shape == img.shape

    bin_inv, cleaned = clean_and_binarize(img)
    assert bin_inv.shape == img.shape
    assert cleaned.shape == img.shape
    assert bin_inv.max() == 255
    assert bin_inv.min() == 0


def test_line_splitting():
    # Create synthetic page with 2 equation lines separated by blank space
    page = np.full((300, 400), 255, dtype=np.uint8)

    # Line 1: y from 40 to 80
    page[40:80, 50:350] = 0

    # Line 2: y from 160 to 200
    page[160:200, 50:350] = 0

    crops = split_page_into_lines(page, min_line_height=15, min_line_gap=15)
    assert len(crops) == 2, f"Expected 2 lines, got {len(crops)}"

    for crop in crops:
        assert crop.image.shape[0] == 128  # Must be normalized to 128px height
        assert crop.image.shape[1] > 32


def test_format_latex_document():
    line_data = [
        {"index": 0, "latex": "a^2+b^2=c^2", "confidence": 0.95, "compiles": True},
        {"index": 1, "latex": r"\int x dx = \frac{x^2}{2}", "confidence": 0.90, "compiles": True},
    ]
    doc = format_latex_document(line_data)
    assert r"\documentclass{article}" in doc
    assert r"\begin{align*}" in doc
    assert r"a^2+b^2=c^2" in doc
    assert r"\end{align*}" in doc
    assert r"\end{document}" in doc
