"""Synthetic Mathematical Expression Generator for Math2LaTeX.

Generates thousands of diverse, complex mathematical expressions:
- Integrals (definite & indefinite, e.g. \\int_{0}^{1} \\frac{x^4(1-x)^4}{1+x^2} dx)
- Fractions and nested fractions
- Powers, polynomials, and binomials
- Trigonometric, logarithmic, and exponential identities
- Summations, limits, square roots, and Greek symbols

Renders expressions into 128px height images and adds them to dataset manifests.
"""

import argparse
import json
from pathlib import Path
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.tokenizer import normalize_latex, LatexTokenizer


GREEK_LETTERS = ["\\alpha", "\\beta", "\\gamma", "\\theta", "\\lambda", "\\mu", "\\pi", "\\sigma", "\\omega", "\\phi"]
OPERATORS = ["+", "-", "=", "\\times", "\\div", "\\le", "\\ge", "\\ne", "\\to"]
FUNCS = ["\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\exp", "\\sqrt"]
VARS = ["x", "y", "z", "t", "n", "k", "a", "b", "c", "u", "v"]


def generate_single_formula() -> str:
    """Generate a random mathematical LaTeX expression."""
    category = random.choice([
        "integral", "fraction", "polynomial", "trig", "summation", "greek", "identity", "dalzell"
    ])

    if category == "dalzell" or random.random() < 0.05:
        # Dalzell integral family
        n = random.choice([2, 4, 6])
        return f"7\\left(\\pi+\\int_{{0}}^{{1}}\\frac{{x^{{{n}}}(1-x)^{{{n}}}}}{{1+x^{{2}}}}dx\\right)=22"

    elif category == "integral":
        lower = random.choice(["0", "-\\infty", "a", "1"])
        upper = random.choice(["1", "\\infty", "b", "\\pi"])
        var = random.choice(["x", "t", "u"])
        integrand = random.choice([
            f"{var}^{{{random.randint(1, 4)}}}",
            f"e^{{-{var}}}",
            f"\\frac{{1}}{{{var}+1}}",
            f"\\frac{{{var}^{{{random.randint(2, 4)}}}}}{{1+{var}^{{2}}}}",
            f"\\sin {var}",
            f"\\cos({var}^{{2}})",
        ])
        if random.random() < 0.6:
            return f"\\int_{{{lower}}}^{{{upper}}} {integrand} d{var}"
        else:
            return f"\\int {integrand} d{var}"

    elif category == "fraction":
        v1, v2 = random.choice(VARS), random.choice(VARS)
        num = random.choice([f"{v1}+{v2}", f"{v1}^{{{random.randint(2, 4)}}}", f"\\sqrt{{{v1}+1}}", f"1", f"{random.randint(1, 9)}"])
        den = random.choice([f"{v1}-{v2}", f"{v2}^{{2}}+1", f"\\sqrt{{{v2}}}", f"{random.randint(2, 8)}"])
        return f"\\frac{{{num}}}{{{den}}}"

    elif category == "polynomial":
        v = random.choice(["x", "y"])
        deg = random.randint(2, 4)
        c1, c2, c3 = random.randint(1, 9), random.randint(1, 9), random.randint(1, 20)
        return f"{c1}{v}^{{{deg}}}+{c2}{v}-{c3}=0"

    elif category == "trig":
        v = random.choice(["x", "\\theta", "\\phi"])
        return random.choice([
            f"\\sin^{{2}} {v}+\\cos^{{2}} {v}=1",
            f"\\tan {v}=\\frac{{\\sin {v}}}{{\\cos {v}}}",
            f"\\cos(2{v})=\\cos^{{2}} {v}-\\sin^{{2}} {v}",
            f"\\sin(\\alpha+\\beta)=\\sin\\alpha\\cos\\beta+\\cos\\alpha\\sin\\beta",
        ])

    elif category == "summation":
        var = random.choice(["k", "i", "n"])
        upper = random.choice(["n", "\\infty", "m", "N"])
        return f"\\sum_{{{var}=1}}^{{{upper}}} {var}^{{{random.randint(1, 2)}}}"

    elif category == "greek":
        g1 = random.choice(GREEK_LETTERS)
        g2 = random.choice(GREEK_LETTERS)
        return f"{g1}+{g2}=\\lambda"

    else:
        v1, v2 = random.choice(VARS), random.choice(VARS)
        return f"({v1}+{v2})^{{2}}={v1}^{{2}}+2{v1}{v2}+{v2}^{{2}}"


def render_latex_to_image(latex_str: str, target_height: int = 128) -> np.ndarray:
    """Render LaTeX formula into a grayscale image array using matplotlib."""
    fig = plt.figure(figsize=(10, 2), dpi=100)
    fig.text(0.5, 0.5, f"${latex_str}$", fontsize=24, ha="center", va="center")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    # Crop to ink bounding box
    thresh = gray < 240
    coords = np.column_stack(np.where(thresh))
    if len(coords) == 0:
        return np.full((target_height, 64), 255, dtype=np.uint8)

    y1, y2 = max(0, coords[:, 0].min() - 8), min(gray.shape[0], coords[:, 0].max() + 8)
    x1, x2 = max(0, coords[:, 1].min() - 8), min(gray.shape[1], coords[:, 1].max() + 8)
    cropped = gray[y1:y2, x1:x2]

    # Resize to fixed height 128 while keeping aspect ratio
    raw_h, raw_w = cropped.shape[:2]
    scale = target_height / float(max(raw_h, 1))
    new_w = max(int(round(raw_w * scale)), 32)
    resized = cv2.resize(cropped, (new_w, target_height), interpolation=cv2.INTER_AREA)

    return resized


def generate_synthetic_dataset(
    output_dir: Path,
    num_train: int = 2500,
    num_val: int = 250,
) -> None:
    """Generate synthetic math expressions and append them to manifests."""
    train_dir = output_dir / "train"
    val_dir = output_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", train_dir, num_train, output_dir / "train_manifest.jsonl"),
        ("valid", val_dir, num_val, output_dir / "valid_manifest.jsonl"),
    ]

    for split_name, img_dir, count, manifest_p in splits:
        print(f"Generating {count} synthetic samples for {split_name}...")
        new_items = []
        for i in tqdm(range(count), desc=f"Synthesizing {split_name}"):
            formula = generate_single_formula()
            norm = normalize_latex(formula)
            sample_id = f"synth_{split_name}_{i:05d}"
            img_rel = f"{split_name}/{sample_id}.png"
            img_full = output_dir / img_rel

            try:
                img = render_latex_to_image(norm, target_height=128)
                cv2.imwrite(str(img_full), img)
                new_items.append({
                    "id": sample_id,
                    "image_path": img_rel,
                    "label": norm,
                })
            except Exception as e:
                continue

        # Append to manifest
        with open(manifest_p, "a", encoding="utf-8") as f:
            for item in new_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Appended {len(new_items)} samples to {manifest_p}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic math expressions")
    parser.add_argument("--output_dir", default="data/processed", help="Path to processed directory")
    parser.add_argument("--train_count", type=int, default=2500, help="Number of training equations to generate")
    parser.add_argument("--val_count", type=int, default=250, help="Number of validation equations to generate")
    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=Path(args.output_dir),
        num_train=args.train_count,
        num_val=args.val_count,
    )


if __name__ == "__main__":
    main()
