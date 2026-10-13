"""Convert online handwriting pen strokes (InkML) to raster images.

Renders vector traces into grayscale images of fixed height 128px with proportional width.
Adds subtle stroke-width jitter, anti-aliased drawing, and produces dataset manifests.
"""

import argparse
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm

from src.data.tokenizer import normalize_latex


def parse_inkml(file_path: Path) -> Tuple[List[List[Tuple[float, float]]], str]:
    """Parse an InkML file and extract stroke traces and ground truth LaTeX.
    
    Args:
        file_path: Path to .inkml file.
        
    Returns:
        traces: List of strokes, where each stroke is a list of (x, y) coordinates.
        label: Ground truth LaTeX string (normalized if available).
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Check namespace if present
    ns = {"ink": "http://www.w3.org/2003/InkML"}
    
    # Extract normalizedLabel or label
    label = ""
    # Try with namespace
    for ann in root.findall("ink:annotation", ns):
        if ann.attrib.get("type") == "normalizedLabel" and ann.text:
            label = ann.text.strip()
            break
    if not label:
        for ann in root.findall("ink:annotation", ns):
            if ann.attrib.get("type") == "label" and ann.text:
                label = ann.text.strip()
                break
    # Fallback without namespace
    if not label:
        for ann in root.findall(".//annotation"):
            if ann.attrib.get("type") == "normalizedLabel" and ann.text:
                label = ann.text.strip()
                break
            elif ann.attrib.get("type") == "label" and ann.text:
                label = ann.text.strip()

    # Extract traces
    traces: List[List[Tuple[float, float]]] = []
    trace_elements = root.findall(".//{http://www.w3.org/2003/InkML}trace")
    if not trace_elements:
        trace_elements = root.findall(".//trace")
        
    for trace in trace_elements:
        if not trace.text:
            continue
        coords: List[Tuple[float, float]] = []
        for point_str in trace.text.strip().split(","):
            parts = point_str.strip().split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append((x, y))
                except ValueError:
                    continue
        if coords:
            traces.append(coords)

    return traces, normalize_latex(label)


def render_traces_to_image(
    traces: List[List[Tuple[float, float]]],
    target_height: int = 128,
    margin: int = 12,
    base_stroke_width: float = 2.5,
    stroke_jitter: float = 0.5,
    max_width: int = 1200,
) -> np.ndarray:
    """Render vector ink traces onto a white canvas with black ink.
    
    Args:
        traces: List of strokes [(x, y), ...].
        target_height: Desired fixed height in pixels (128px per design doc).
        margin: Vertical padding margin inside the canvas.
        base_stroke_width: Baseline line thickness.
        stroke_jitter: Random variation applied to thickness.
        max_width: Maximum allowed canvas width.
        
    Returns:
        Grayscale image (np.ndarray) with shape (target_height, width), uint8 (0-255).
    """
    # Fallback for empty traces: return blank white strip
    if not traces or all(len(t) == 0 for t in traces):
        return np.full((target_height, 64), 255, dtype=np.uint8)

    # Calculate global bounding box
    all_points = [pt for stroke in traces for pt in stroke]
    xs = [pt[0] for pt in all_points]
    ys = [pt[1] for pt in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    raw_w = max(max_x - min_x, 1e-3)
    raw_h = max(max_y - min_y, 1e-3)

    # Scaling factor to match target_height while preserving aspect ratio
    usable_h = target_height - 2 * margin
    scale = usable_h / raw_h
    out_w = int(raw_w * scale) + 2 * margin
    out_w = max(min(out_w, max_width), 32)

    # Initialize white canvas
    canvas = np.full((target_height, out_w), 255, dtype=np.uint8)

    # Randomized stroke width per sample
    pen_w = max(1, int(round(base_stroke_width + random.uniform(-stroke_jitter, stroke_jitter))))

    # Draw strokes
    for stroke in traces:
        if not stroke:
            continue
        # Transform points into canvas coordinates
        pts = [
            (
                int(round((pt[0] - min_x) * scale + margin)),
                int(round((pt[1] - min_y) * scale + margin)),
            )
            for pt in stroke
        ]

        if len(pts) == 1:
            # Single dot
            cv2.circle(canvas, pts[0], max(1, pen_w // 2), color=0, thickness=-1)
        else:
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], color=0, thickness=pen_w, lineType=cv2.LINE_AA)

    return canvas


def process_dataset_split(
    split_dir: Path,
    output_dir: Path,
    split_name: str,
    target_height: int = 128,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Convert an InkML directory split to PNG images and create a manifest.
    
    Returns:
        manifest: List of dicts {"image_path": str, "label": str, "id": str}
        sample_ids: List of sample IDs in this split (used to verify no leakage)
    """
    inkml_files = sorted(list(split_dir.glob("*.inkml")))
    if max_samples and max_samples > 0:
        inkml_files = inkml_files[:max_samples]

    img_out_dir = output_dir / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, str]] = []
    sample_ids: List[str] = []

    print(f"Processing split '{split_name}': {len(inkml_files)} samples...")
    for f in tqdm(inkml_files, desc=f"Rendering {split_name}"):
        sample_id = f.stem
        try:
            traces, label = parse_inkml(f)
            if not label:
                continue

            img = render_traces_to_image(traces, target_height=target_height)
            img_rel_path = f"{split_name}/{sample_id}.png"
            img_full_path = output_dir / img_rel_path

            # Save PNG image
            cv2.imwrite(str(img_full_path), img)

            manifest.append({
                "id": sample_id,
                "image_path": img_rel_path,
                "label": label,
            })
            sample_ids.append(sample_id)
        except Exception as e:
            print(f"[Warning] Failed rendering {f.name}: {e}")
            continue

    manifest_path = output_dir / f"{split_name}_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as out_f:
        for item in manifest:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(manifest)} samples to {manifest_path}")
    return manifest, sample_ids


def main():
    parser = argparse.ArgumentParser(description="Render InkML handwriting strokes into PNG images")
    parser.add_argument("--raw_dir", default="data/raw/mathwriting", help="Directory containing raw InkML files")
    parser.add_argument("--output_dir", default="data/processed", help="Directory to save rendered images")
    parser.add_argument("--target_height", type=int, default=128, help="Target image height in pixels")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per split (useful for small tests)")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)

    # Locate directories (support both excerpt folder structure and direct folders)
    candidates = list(raw_path.glob("**/train"))
    if not candidates:
        print(f"[ERROR] No train directory found in {raw_path}")
        return
    base_split_dir = candidates[0].parent

    train_dir = base_split_dir / "train"
    valid_dir = base_split_dir / "valid"
    test_dir = base_split_dir / "test"

    train_manifest, train_ids = process_dataset_split(train_dir, out_path, "train", args.target_height, args.max_samples)
    val_manifest, val_ids = process_dataset_split(valid_dir, out_path, "valid", args.target_height, args.max_samples)
    test_manifest, test_ids = process_dataset_split(test_dir, out_path, "test", args.target_height, args.max_samples)

    # Verify no split leakage (Section 14 requirement)
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    leak_train_val = train_set.intersection(val_set)
    leak_train_test = train_set.intersection(test_set)
    leak_val_test = val_set.intersection(test_set)

    assert not leak_train_val, f"Split leakage detected between train and val: {leak_train_val}"
    assert not leak_train_test, f"Split leakage detected between train and test: {leak_train_test}"
    assert not leak_val_test, f"Split leakage detected between val and test: {leak_val_test}"
    print("Verification passed: Zero overlap between train, val, and test splits!")


if __name__ == "__main__":
    main()
