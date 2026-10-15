"""Evaluation metrics and error reporting for Math2LaTeX.

Computes:
- Exact Match rate (%)
- Normalized Levenshtein Edit Distance
- LaTeX Compile Rate (%)
- Saves the top 50 worst prediction mistakes for error analysis
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.config import load_config
from src.data.tokenizer import LatexTokenizer, normalize_latex
from src.data.dataset import create_dataloader
from src.model.img2latex import Img2Latex


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance using dynamic programming."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m

    # Use two rows to save memory
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i
        c1 = s1[i - 1]
        for j in range(1, n + 1):
            c2 = s2[j - 1]
            cost = 0 if c1 == c2 else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,       # insertion
                prev_row[j] + 1,           # deletion
                prev_row[j - 1] + cost,    # substitution
            )
        prev_row, curr_row = curr_row, [0] * (n + 1)

    return prev_row[n]


def compute_normalized_edit_distance(pred: str, ref: str) -> float:
    """Normalized edit distance in range [0.0, 1.0]. Lower is better."""
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 0.0
    dist = levenshtein_distance(pred, ref)
    return dist / max_len


def is_exact_match(pred: str, ref: str) -> bool:
    """Check if normalized prediction equals normalized ground truth."""
    return normalize_latex(pred) == normalize_latex(ref)


def check_latex_compilable(latex_str: str) -> bool:
    """Check if a LaTeX string is syntactically valid and compilable.
    
    Uses pdflatex if available; otherwise performs strict structural validation.
    """
    cleaned = latex_str.strip()
    if not cleaned:
        return True

    # 1. Structural checks
    # Check brace balance
    depth = 0
    for ch in cleaned:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False

    # Check trailing illegal tokens
    if cleaned.endswith(("\\", "^", "_")):
        return False

    # Check matching \left and \right delimiters
    lefts = len(re.findall(r"\\left(?![a-zA-Z])", cleaned))
    rights = len(re.findall(r"\\right(?![a-zA-Z])", cleaned))
    if lefts != rights:
        return False

    # Check that \frac has two braced arguments: \frac{...}{...}
    pos = 0
    while True:
        idx = cleaned.find(r"\frac", pos)
        if idx == -1:
            break
        pos = idx + 5
        first_open = cleaned.find("{", pos)
        if first_open == -1:
            return False
        # find matching close brace
        b_depth = 1
        j = first_open + 1
        while j < len(cleaned) and b_depth > 0:
            if cleaned[j] == "{":
                b_depth += 1
            elif cleaned[j] == "}":
                b_depth -= 1
            j += 1
        if b_depth != 0:
            return False
        # find second group
        second_part = cleaned[j:].lstrip()
        if not second_part.startswith("{"):
            return False
        pos = j

    # 2. If pdflatex exists on the machine, test compilation
    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin:
        tex_source = (
            "\\documentclass{article}\n"
            "\\usepackage{amsmath,amssymb}\n"
            "\\begin{document}\n"
            f"\\[ {cleaned} \\]\n"
            "\\end{document}\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test.tex"
            tmp_path.write_text(tex_source, encoding="utf-8")
            try:
                res = subprocess.run(
                    [pdflatex_bin, "-interaction=nonstopmode", "-halt-on-error", "test.tex"],
                    cwd=tmp_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                return res.returncode == 0
            except Exception:
                return False

    return True


@torch.no_grad()
def evaluate_model(
    model: Img2Latex,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: LatexTokenizer,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run full evaluation on a dataset split.
    
    Returns:
        metrics: Dict with exact_match, mean_edit_distance, compile_rate.
        sample_results: List of per-sample predictions, targets, and edit distances.
    """
    model.eval()
    model.to(device)

    total_samples = 0
    exact_matches = 0
    total_edit_distance = 0.0
    compilable_count = 0
    sample_results: List[Dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["images"].to(device)
        ground_truths = batch["labels"]
        sample_ids = batch["ids"]

        # Greedy autoregressive decode
        pred_token_ids = model.greedy_decode(images)

        for i, token_ids in enumerate(pred_token_ids):
            pred_latex = tokenizer.decode(token_ids)
            ref_latex = normalize_latex(ground_truths[i])

            exact = is_exact_match(pred_latex, ref_latex)
            norm_dist = compute_normalized_edit_distance(pred_latex, ref_latex)
            compiles = check_latex_compilable(pred_latex)

            if exact:
                exact_matches += 1
            total_edit_distance += norm_dist
            if compiles:
                compilable_count += 1

            sample_results.append({
                "id": sample_ids[i],
                "ground_truth": ref_latex,
                "prediction": pred_latex,
                "exact_match": exact,
                "edit_distance": norm_dist,
                "compilable": compiles,
            })

            total_samples += 1
            if max_samples and total_samples >= max_samples:
                break
        if max_samples and total_samples >= max_samples:
            break

    n = max(total_samples, 1)
    metrics = {
        "total_samples": total_samples,
        "exact_match": (exact_matches / n) * 100.0,
        "mean_edit_distance": total_edit_distance / n,
        "compile_rate": (compilable_count / n) * 100.0,
    }

    return metrics, sample_results


def save_worst_mistakes(
    sample_results: List[Dict[str, Any]],
    output_path: str = "reports/worst_50_mistakes.json",
    top_k: int = 50,
) -> None:
    """Save the top K worst prediction mistakes sorted by edit distance descending."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort descending by edit distance
    sorted_mistakes = sorted(sample_results, key=lambda x: x["edit_distance"], reverse=True)
    worst_k = sorted_mistakes[:top_k]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(worst_k, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(worst_k)} worst mistakes to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Math2LaTeX model checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to checkpoint file")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--split", choices=["test", "valid", "train"], default="test", help="Split to evaluate")
    parser.add_argument("--manifest", default=None, help="Custom manifest path")
    parser.add_argument("--report", default="reports/evaluation_results.json", help="Path to save report")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = config["data"]["vocab_path"]
    tokenizer = LatexTokenizer(vocab_path)

    manifest_path = args.manifest or f"data/processed/{args.split}_manifest.jsonl"
    loader = create_dataloader(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        config=config,
        split=args.split,
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        shuffle=False,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = Img2Latex(
        vocab_size=tokenizer.vocab_size,
        model_dim=config["model"]["model_dim"],
        decoder_layers=config["model"]["decoder_layers"],
        heads=config["model"]["heads"],
        feedforward_dim=config["model"]["feedforward_dim"],
        dropout=config["model"]["dropout"],
        max_tokens=config["model"]["max_tokens"],
        pad_id=tokenizer.pad_id,
        start_id=tokenizer.start_id,
        end_id=tokenizer.end_id,
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics, results = evaluate_model(model, loader, tokenizer, device)
    print("\n" + "=" * 40)
    print(f"EVALUATION RESULTS ({args.split.upper()} SET)")
    print("=" * 40)
    print(f"Total Samples:       {metrics['total_samples']}")
    print(f"Exact Match:         {metrics['exact_match']:.2f}%")
    print(f"Mean Edit Distance:  {metrics['mean_edit_distance']:.4f}")
    print(f"Compile Rate:        {metrics['compile_rate']:.2f}%")
    print("=" * 40)

    # Save report and worst mistakes
    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_worst_mistakes(results, output_path="reports/worst_50_mistakes.json")


if __name__ == "__main__":
    main()
