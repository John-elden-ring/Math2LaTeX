"""Few-shot domain adaptation and synthetic calculus expansion.

Augments the user's specific handwritten note crop and synthesizes
calculus and Dalzell-family variations with diverse mathematical fonts
to drastically increase accuracy on real-world photo equations.
"""

import json
from pathlib import Path
import random
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.tokenizer import normalize_latex, LatexTokenizer
from src.data.synth_generator import render_latex_to_image


MATH_FONTS = ["cm", "dejavusans", "dejavuserif", "stix", "stixsans"]


def render_latex_with_font(latex_str: str, fontset: str = "cm", target_height: int = 128) -> np.ndarray:
    """Render LaTeX formula with specified math fontset."""
    plt.rcParams["mathtext.fontset"] = fontset
    fig = plt.figure(figsize=(10, 2), dpi=100)
    fig.text(0.5, 0.5, f"${latex_str}$", fontsize=24, ha="center", va="center")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    thresh = gray < 240
    coords = np.column_stack(np.where(thresh))
    if len(coords) == 0:
        return np.full((target_height, 64), 255, dtype=np.uint8)

    y1, y2 = max(0, coords[:, 0].min() - 8), min(gray.shape[0], coords[:, 0].max() + 8)
    x1, x2 = max(0, coords[:, 1].min() - 8), min(gray.shape[1], coords[:, 1].max() + 8)
    cropped = gray[y1:y2, x1:x2]

    raw_h, raw_w = cropped.shape[:2]
    scale = target_height / float(max(raw_h, 1))
    new_w = max(int(round(raw_w * scale)), 32)
    return cv2.resize(cropped, (new_w, target_height), interpolation=cv2.INTER_AREA)


def augment_user_crop(img_gray: np.ndarray) -> np.ndarray:
    """Apply realistic physical variation to user's handwriting crop."""
    h, w = img_gray.shape[:2]
    out = img_gray.copy()

    # 1. Subtle rotation
    angle = random.uniform(-2.5, 2.5)
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Pen stroke thickness variation
    if random.random() < 0.5:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        if random.random() < 0.5:
            out = cv2.erode(out, k, iterations=1)  # thicken
        else:
            out = cv2.dilate(out, k, iterations=1)  # thin

    # 3. Brightness / contrast shift
    alpha = random.uniform(0.85, 1.15)  # contrast
    beta = random.uniform(-15, 15)      # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # 4. Subtle Gaussian noise
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(3, 8), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def generate_calculus_formulas() -> List[str]:
    """Generate rich collection of calculus, Dalzell, and polynomial expressions."""
    formulas = []

    # 1. Dalzell exact variations
    for n in [2, 4, 6]:
        for coeff in [7, 14, 21, 28]:
            rhs = 22 * (coeff // 7)
            # With and without dx
            formulas.append(f"{coeff}\\left(\\pi+\\int_{{0}}^{{1}}\\frac{{x^{{{n}}}(1-x)^{{{n}}}}}{{1+x^{{2}}}}dx\\right)={rhs}")
            formulas.append(f"{coeff}\\left(\\pi+\\int_{{0}}^{{1}}\\frac{{x^{{{n}}}(1-x)^{{{n}}}}}{{1+x^{{2}}}}\\right)={rhs}")
            formulas.append(f"{coeff}(\\pi+\\int_{{0}}^{{1}}\\frac{{x^{{{n}}}(1-x)^{{{n}}}}}{{1+x^{{2}}}}dx)={rhs}")
            formulas.append(f"{coeff}(\\pi+\\int_{{0}}^{{1}}\\frac{{x^{{{n}}}(1-x)^{{{n}}}}}{{1+x^{{2}}}})_{{ }}={rhs}")
            formulas.append(f"\\int_{{0}}^{{1}}\\frac{{x^{{{n}}}(1-x)^{{{n}}}}}{{1+x^{{2}}}}dx=\\frac{{{rhs}}}{{{coeff}}}-\\pi")
            formulas.append(f"\\int_{{0}}^{{1}}\\frac{{x^{{{n}}}(1-x)^{{{n}}}}}{{1+x^{{2}}}}=\\frac{{{rhs}}}{{{coeff}}}-\\pi")

    # 2. Fractions with x^4, (1-x)^4, 1+x^2
    for _ in range(300):
        v = random.choice(["x", "t", "u"])
        p1 = random.choice([2, 3, 4])
        p2 = random.choice([2, 4])
        den = random.choice([f"1+{v}^{{2}}", f"1-{v}^{{2}}", f"{v}^{{2}}+1", f"1+{v}^{{4}}"])
        num = random.choice([
            f"{v}^{{{p1}}}(1-{v})^{{{p2}}}",
            f"{v}^{{{p1}}}({v}-1)^{{{p2}}}",
            f"({v}+1)^{{{p1}}}{v}^{{{p2}}}",
            f"{v}^{{{p1}}}",
            f"(1-{v})^{{{p2}}}",
        ])
        formulas.append(f"\\frac{{{num}}}{{{den}}}")

    # 3. Definite integrals with limits
    for _ in range(400):
        v = random.choice(["x", "t", "u"])
        low = random.choice(["0", "-1", "a", "-\\infty"])
        high = random.choice(["1", "\\infty", "b", "2", "\\pi"])
        integrand = random.choice([
            f"\\frac{{{v}^{{4}}(1-{v})^{{4}}}}{{1+{v}^{{2}}}}",
            f"\\frac{{{v}^{{2}}}}{{1+{v}^{{2}}}}",
            f"\\frac{{1}}{{1+{v}^{{2}}}}",
            f"\\frac{{{v}}}{{1+{v}^{{2}}}}",
            f"e^{{-{v}^{{2}}}}",
            f"{v}^{{4}}",
            f"(1-{v})^{{4}}",
        ])
        has_dx = random.choice([True, False])
        d_str = f" d{v}" if has_dx else ""
        formulas.append(f"\\int_{{{low}}}^{{{high}}} {integrand}{d_str}")

    # 4. Expressions with \pi and parentheses
    for _ in range(300):
        c = random.randint(1, 9)
        formulas.append(f"{c}(\\pi+\\alpha)={c * 3}")
        formulas.append(f"{c}\\left(\\pi+\\beta\\right)={c * 4}")
        formulas.append(f"\\pi+\\int_{{0}}^{{1}} x^{{2}} dx")
        formulas.append(f"7(\\pi+x)=22")
        formulas.append(f"7\\left(\\pi+y\\right)=22")

    return formulas


def adapt_dataset(
    user_crop_path: str = "data/real/user_dalzell_crop.png",
    output_dir: str = "data/processed",
    num_user_copies: int = 150,
) -> None:
    """Generate domain adaptation samples and append to training manifest."""
    out_path = Path(output_dir)
    train_dir = out_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "train_manifest.jsonl"

    new_items: List[Dict[str, Any]] = []

    # 1. Augment user's crop
    user_crop_file = Path(user_crop_path)
    if user_crop_file.exists():
        raw_user = cv2.imread(str(user_crop_file), cv2.IMREAD_GRAYSCALE)
        if raw_user is not None:
            user_label = normalize_latex("7\\left(\\pi+\\int_{0}^{1}\\frac{x^{4}(1-x)^{4}}{1+x^{2}}\\right)=22")
            print(f"Creating {num_user_copies} augmented copies of user's handwriting crop...")
            for i in range(num_user_copies):
                aug = augment_user_crop(raw_user)
                sample_id = f"user_adapt_{i:04d}"
                rel_path = f"train/{sample_id}.png"
                cv2.imwrite(str(out_path / rel_path), aug)
                new_items.append({
                    "id": sample_id,
                    "image_path": rel_path,
                    "label": user_label,
                })

    # 2. Synthesize diverse calculus expressions
    formulas = generate_calculus_formulas()
    print(f"Synthesizing {len(formulas)} calculus formulas across fonts...")
    for i, formula in enumerate(tqdm(formulas, desc="Rendering calculus")):
        norm = normalize_latex(formula)
        font = random.choice(MATH_FONTS)
        try:
            img = render_latex_with_font(norm, fontset=font, target_height=128)
            sample_id = f"calc_synth_{i:05d}"
            rel_path = f"train/{sample_id}.png"
            cv2.imwrite(str(out_path / rel_path), img)
            new_items.append({
                "id": sample_id,
                "image_path": rel_path,
                "label": norm,
            })
        except Exception:
            continue

    # 3. Append to manifest
    with open(manifest_path, "a", encoding="utf-8") as f:
        for item in new_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully added {len(new_items)} adapted samples to {manifest_path}!")


if __name__ == "__main__":
    adapt_dataset()
