"""PyTorch Dataset and DataLoader for Math2LaTeX.

Loads equation images and LaTeX labels from JSONL manifests.
Provides dynamic batch collation with proportional width padding and token sequence padding.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.data.augment import augment_image
from src.data.tokenizer import LatexTokenizer


# Standard ImageNet normalization for ResNet-18 backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MathDataset(Dataset):
    """PyTorch Dataset for handwritten math expression recognition."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        tokenizer: LatexTokenizer,
        image_root: Optional[Union[str, Path]] = None,
        is_training: bool = False,
        config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else self.manifest_path.parent
        self.is_training = is_training
        self.config = config or {}
        
        # Load samples from JSONL manifest
        self.samples: List[Dict[str, str]] = []
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Limit samples if requested
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        # PyTorch image tensor normalization transform
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_rel_path = item["image_path"]
        label_text = item["label"]
        sample_id = item.get("id", str(idx))

        full_img_path = self.image_root / img_rel_path
        img_gray = cv2.imread(str(full_img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            # Handle corrupt/missing image gracefully with blank canvas
            img_gray = np.full((128, 64), 255, dtype=np.uint8)

        # Apply augmentation (train only) or convert to clean RGB
        img_rgb = augment_image(img_gray, is_training=self.is_training, config=self.config)

        # Convert to Tensor (3, H, W) in [0.0, 1.0] and normalize
        tensor_img = self.to_tensor(img_rgb)
        tensor_img = self.normalize(tensor_img)

        # Tokenize target LaTeX
        token_ids = self.tokenizer.encode(label_text, add_special_tokens=True)

        return {
            "image": tensor_img,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": label_text,
            "id": sample_id,
            "width": tensor_img.shape[2],
        }


def math_collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    """Dynamically pad images and token sequences to the longest within the batch.
    
    Why:
        Preserves natural aspect ratios of mathematical equations of varying length
        without stretching or squishing characters.
    """
    batch_size = len(batch)
    channels = batch[0]["image"].shape[0]
    height = batch[0]["image"].shape[1]

    # Find maximum image width and token sequence length in this batch
    max_w = max(item["width"] for item in batch)
    max_token_len = max(len(item["token_ids"]) for item in batch)

    # Standard ImageNet normalized white background value
    # (1.0 - mean) / std
    white_vals = torch.tensor([(1.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]).view(channels, 1, 1)

    # Initialize padded batch tensors
    padded_images = white_vals.repeat(batch_size, 1, height, max_w)
    padded_targets = torch.full((batch_size, max_token_len), pad_id, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    widths = torch.zeros(batch_size, dtype=torch.long)

    labels: List[str] = []
    ids: List[str] = []

    for i, item in enumerate(batch):
        img = item["image"]
        tokens = item["token_ids"]
        w = item["width"]
        t_len = len(tokens)

        padded_images[i, :, :, :w] = img
        padded_targets[i, :t_len] = tokens
        target_lengths[i] = t_len
        widths[i] = w
        labels.append(item["label"])
        ids.append(item["id"])

    return {
        "images": padded_images,
        "targets": padded_targets,
        "target_lengths": target_lengths,
        "widths": widths,
        "labels": labels,
        "ids": ids,
    }


class MathCollator:
    """Pickleable batch collation callable for multi-worker DataLoader."""
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return math_collate_fn(batch, pad_id=self.pad_id)


def create_dataloader(
    manifest_path: Union[str, Path],
    tokenizer: LatexTokenizer,
    config: Optional[Dict[str, Any]] = None,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build PyTorch DataLoader with math_collate_fn."""
    is_training = (split == "train")
    do_shuffle = is_training if shuffle is None else shuffle

    dataset = MathDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        is_training=is_training,
        config=config,
        max_samples=max_samples,
    )

    pad_id = tokenizer.pad_id

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        collate_fn=MathCollator(pad_id=pad_id),
        pin_memory=torch.cuda.is_available(),
    )
