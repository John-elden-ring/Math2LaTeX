"""Training loop for Math2LaTeX.

Implements:
- 100-sample Overfit Sanity Check (--overfit flag)
- AdamW optimizer with warmup and cosine decay
- Gradient clipping (max_norm 1.0)
- Mixed precision (fp16) acceleration when GPU is available
- Checkpoint persistence and resume capability
- Validation tracking by Normalized Edit Distance
"""

import argparse
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.config import load_config
from src.data.tokenizer import LatexTokenizer
from src.data.dataset import create_dataloader
from src.model.img2latex import Img2Latex
from src.model.evaluate import evaluate_model


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    """Linear warmup followed by cosine annealing decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: Img2Latex,
    dataloader,
    optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    clip_norm: float = 1.0,
    log_interval: int = 25,
) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}")
    for step, batch in enumerate(pbar):
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Mixed precision if GPU scaler is available
        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss, _ = model(images, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _ = model(images, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.2e}"})

    return total_loss / max(1, num_batches)


def run_training(
    config_path: str = "configs/base.yaml",
    is_overfit: bool = False,
    resume_path: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    max_samples_override: Optional[int] = None,
) -> None:
    """Main training routine."""
    config = load_config(config_path)
    set_seed(config["data"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # Load tokenizer
    vocab_path = config["data"]["vocab_path"]
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Run strokes_to_image / tokenizer first.")
    tokenizer = LatexTokenizer(vocab_path)
    print(f"Loaded vocabulary with {tokenizer.vocab_size} tokens.")

    # Configure sample count & epochs
    if is_overfit:
        max_samples = 100
        epochs = epochs_override or 40
        batch_size = 16
        print("=" * 60)
        print("RUNNING 100-SAMPLE OVERFIT SANITY CHECK (MILESTONE STAGE 4)")
        print("Goal: Drive training loss close to zero on 100 samples.")
        print("=" * 60)
    else:
        max_samples = max_samples_override or config["data"].get("max_samples", None)
        epochs = epochs_override or config["training"].get("epochs", 30)
        batch_size = batch_size_override or config["training"].get("batch_size", 32)

    # Build DataLoaders
    train_manifest = f"{config['data']['processed_dir']}/train_manifest.jsonl"
    val_manifest = f"{config['data']['processed_dir']}/valid_manifest.jsonl"

    # For overfit check, disable data augmentation so network memorizes clean strokes directly
    train_cfg = dict(config)
    if is_overfit:
        train_cfg["augmentation"] = {"enabled": False}

    train_loader = create_dataloader(
        manifest_path=train_manifest,
        tokenizer=tokenizer,
        config=train_cfg,
        split="train",
        batch_size=batch_size,
        num_workers=0 if device.type == "cpu" else config["training"].get("num_workers", 0),
        shuffle=True,
        max_samples=max_samples,
    )

    val_loader = create_dataloader(
        manifest_path=val_manifest,
        tokenizer=tokenizer,
        config=config,
        split="valid",
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        max_samples=100 if is_overfit else None,
    )

    # Initialize model
    model = Img2Latex(
        vocab_size=tokenizer.vocab_size,
        model_dim=config["model"]["model_dim"],
        decoder_layers=config["model"]["decoder_layers"],
        heads=config["model"]["heads"],
        feedforward_dim=config["model"]["feedforward_dim"],
        dropout=0.0 if is_overfit else config["model"]["dropout"],
        max_tokens=config["model"]["max_tokens"],
        pad_id=tokenizer.pad_id,
        start_id=tokenizer.start_id,
        end_id=tokenizer.end_id,
        pretrained_backbone=True,
    )
    model.to(device)

    # Optimizer & LR Scheduler
    lr = lr_override or config["training"]["lr"]
    # If overfitting, a slightly higher lr like 3e-4 speeds up convergence to zero
    if is_overfit:
        lr = 3e-4

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0 if is_overfit else config["training"].get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(len(train_loader) * (1 if is_overfit else config["training"].get("warmup_epochs", 2)))
    scheduler = build_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # Mixed precision scaler (CUDA only)
    use_mp = config["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_mp else None

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_dist = float("inf")
    ckpt_dir = Path(config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_path:
        print(f"Resuming training from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if not lr_override:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_override
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dist = float("inf")  # Reset to track improvements during new run
        print(f"Resumed at epoch {start_epoch} (lr: {lr_override or lr:.2e})")

    print(f"Starting training for {epochs} epochs ({len(train_loader)} batches/epoch)...")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            clip_norm=config["training"].get("clip_grad_norm", 1.0),
        )
        duration = time.time() - epoch_start
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Time: {duration:.1f}s")

        # Save latest checkpoint
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_loss": train_loss,
            "best_val_dist": best_val_dist,
        }
        torch.save(latest_ckpt, ckpt_dir / "latest_model.pt")

        # Validation evaluation
        if is_overfit:
            # Overfit milestone condition: loss < 0.08 indicates near zero convergence
            if train_loss < 0.08:
                print("\n" + "=" * 60)
                print(f"[MILESTONE SUCCESS] Model successfully overfit 100 samples! Loss = {train_loss:.4f} < 0.08")
                print("=" * 60)
                torch.save(latest_ckpt, ckpt_dir / "overfit_100_model.pt")
                break
        else:
            # Fast validation pass (25 samples)
            val_metrics, _ = evaluate_model(model, val_loader, tokenizer, device, max_samples=25)
            val_dist = val_metrics["mean_edit_distance"]
            print(f"Validation -> Edit Dist: {val_dist:.4f} | Exact Match: {val_metrics['exact_match']:.2f}% | Compiles: {val_metrics['compile_rate']:.2f}%")

            if val_dist < best_val_dist:
                best_val_dist = val_dist
                print(f"[*] New best validation score! Saving to {ckpt_dir / 'best_model.pt'}")
                torch.save(latest_ckpt, ckpt_dir / "best_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Math2LaTeX model")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base.yaml")
    parser.add_argument("--overfit", action="store_true", help="Run 100-sample overfit sanity check")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Override training sample count")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        is_overfit=args.overfit,
        resume_path=args.resume,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        max_samples_override=args.max_samples,
    )


if __name__ == "__main__":
    main()
