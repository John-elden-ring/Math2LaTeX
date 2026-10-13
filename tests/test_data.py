"""Unit tests for Stage 2: Tokenizer, Dataset, Augmentation, and Data Pipeline."""

import json
from pathlib import Path
import pytest
import torch

from src.data.tokenizer import LatexTokenizer, normalize_latex, tokenize_latex
from src.data.strokes_to_image import parse_inkml, render_traces_to_image
from src.data.dataset import MathDataset, math_collate_fn
from src.data.augment import augment_image


def test_tokenizer_normalization():
    # Test whitespace trimming and canonical forms
    assert normalize_latex("  x  +   y  ") == "x+y"
    assert normalize_latex(r"\le") == r"\leq"
    assert normalize_latex(r"\to") == r"\rightarrow"
    assert normalize_latex(r"\frac { 1 } { 2 }") == r"\frac{1}{2}"


def test_tokenizer_roundtrip():
    samples = [
        r"x^{2}+y^{2}=z^{2}",
        r"\frac{a+b}{c-d}",
        r"\sum_{i=1}^{n} x_{i}",
        r"\int_{0}^{\infty} e^{-x} dx",
        r"\alpha+\beta=\gamma",
        r"\sqrt{x^{2}+1}",
        r"\log x+\sin y",
    ]
    tokenizer = LatexTokenizer()
    tokenizer.build_vocab(samples)
    
    for s in samples:
        norm = normalize_latex(s)
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        assert decoded == norm, f"Roundtrip failed for {s}: expected {norm}, got {decoded}"


def test_tokenizer_save_load(tmp_path):
    tokenizer = LatexTokenizer()
    tokenizer.build_vocab([r"\frac{1}{2}", r"x+y=z"])
    vocab_file = tmp_path / "vocab.json"
    tokenizer.save(vocab_file)
    
    loaded = LatexTokenizer(vocab_file)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.token2id == tokenizer.token2id


def test_strokes_rendering():
    # Synthetic traces: two strokes forming an 'X'
    traces = [
        [(10.0, 10.0), (50.0, 50.0)],
        [(50.0, 10.0), (10.0, 50.0)],
    ]
    img = render_traces_to_image(traces, target_height=128)
    assert img.shape[0] == 128
    assert img.shape[1] >= 32
    assert img.dtype.name == "uint8"
    assert img.min() == 0  # Black ink present
    assert img.max() == 255  # White paper present


def test_augmentation_shapes():
    import numpy as np
    canvas = np.full((128, 200), 255, dtype=np.uint8)
    # Draw simple line
    canvas[50:70, 50:150] = 0

    aug_train = augment_image(canvas, is_training=True)
    assert aug_train.shape == (128, 200, 3)

    aug_val = augment_image(canvas, is_training=False)
    assert aug_val.shape == (128, 200, 3)


def test_dataset_and_collate():
    manifest_path = Path("data/processed/train_manifest.jsonl")
    vocab_path = Path("data/processed/vocab.json")
    if not manifest_path.exists() or not vocab_path.exists():
        pytest.skip("Processed dataset not yet generated.")

    tokenizer = LatexTokenizer(vocab_path)
    dataset = MathDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        is_training=False,
        max_samples=10,
    )
    assert len(dataset) == 10

    sample = dataset[0]
    assert sample["image"].shape[0] == 3
    assert sample["image"].shape[1] == 128
    assert len(sample["token_ids"]) > 0

    # Test collate function with batch size 3
    batch = [dataset[0], dataset[1], dataset[2]]
    collated = math_collate_fn(batch, pad_id=tokenizer.pad_id)

    images = collated["images"]
    targets = collated["targets"]
    assert images.shape[0] == 3
    assert images.shape[1] == 3
    assert images.shape[2] == 128
    assert targets.shape[0] == 3
    assert collated["target_lengths"].shape[0] == 3


def test_split_non_overlap():
    for split in ["train", "valid", "test"]:
        p = Path(f"data/processed/{split}_manifest.jsonl")
        if not p.exists():
            pytest.skip("Manifest files not yet generated.")

    def load_ids(split):
        ids = set()
        with open(f"data/processed/{split}_manifest.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                ids.add(json.loads(line)["id"])
        return ids

    train_ids = load_ids("train")
    val_ids = load_ids("valid")
    test_ids = load_ids("test")

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap!"
