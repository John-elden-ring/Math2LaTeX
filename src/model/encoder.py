"""ResNet-18 Image Encoder with 2D Positional Embeddings for Math2LaTeX.

Encodes equation images into a spatial sequence of 256-dimensional feature vectors.
Adds 2D sinusoidal positional encodings so the Transformer decoder knows the spatial
origin (row and column) of every visual feature.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def create_2d_sinusoidal_embeddings(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate 2D sinusoidal positional encodings for an H x W feature grid.
    
    Args:
        height: Feature grid height.
        width: Feature grid width.
        dim: Embedding dimension (must be divisible by 4).
        device: Device to place the tensor on.
        
    Returns:
        Tensor of shape (1, height * width, dim).
    """
    assert dim % 4 == 0, f"Embedding dimension {dim} must be divisible by 4 for 2D encoding."
    half_dim = dim // 2
    quarter_dim = dim // 4

    # 1D row coordinates (0 to H-1) and col coordinates (0 to W-1)
    y_pos = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1)
    x_pos = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(quarter_dim, dtype=torch.float32, device=device) * (-math.log(10000.0) / quarter_dim)
    )

    # Row positional encoding: shape (H, quarter_dim * 2) = (H, half_dim)
    y_sin = torch.sin(y_pos * div_term)
    y_cos = torch.cos(y_pos * div_term)
    y_emb = torch.cat([y_sin, y_cos], dim=1)  # (H, half_dim)

    # Column positional encoding: shape (W, quarter_dim * 2) = (W, half_dim)
    x_sin = torch.sin(x_pos * div_term)
    x_cos = torch.cos(x_pos * div_term)
    x_emb = torch.cat([x_sin, x_cos], dim=1)  # (W, half_dim)

    # Expand into 2D grid: (H, W, half_dim)
    y_emb_grid = y_emb.unsqueeze(1).repeat(1, width, 1)
    x_emb_grid = x_emb.unsqueeze(0).repeat(height, 1, 1)

    # Concatenate row and column embeddings: (H, W, dim)
    pos_2d = torch.cat([y_emb_grid, x_emb_grid], dim=2)
    # Flatten spatial grid into sequence: (1, H * W, dim)
    return pos_2d.view(1, height * width, dim)


class ResNetEncoder(nn.Module):
    """ResNet-18 visual feature extractor for mathematical handwriting."""

    def __init__(
        self,
        model_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim

        # Load ResNet-18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Truncate final avgpool and fc classifier layers (keep conv1 through layer4)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # 1x1 convolution to project 512 backbone channels to model_dim (256)
        self.channel_proj = nn.Conv2d(512, model_dim, kernel_size=1)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Forward pass.
        
        Args:
            images: Tensor of shape (B, 3, H, W).
            
        Returns:
            features: Sequence tensor of shape (B, S_enc, model_dim) where S_enc = H_feat * W_feat.
            spatial_shape: (H_feat, W_feat).
        """
        batch_size = images.shape[0]

        # Extract features: (B, 512, H_feat, W_feat)
        feat_map = self.backbone(images)

        # Project 512 channels -> model_dim (256): (B, model_dim, H_feat, W_feat)
        feat_map = self.channel_proj(feat_map)

        _, channels, feat_h, feat_w = feat_map.shape

        # Compute 2D positional embeddings: (1, feat_h * feat_w, model_dim)
        pos_embed = create_2d_sinusoidal_embeddings(feat_h, feat_w, self.model_dim, images.device)

        # Permute feature map to (B, feat_h * feat_w, model_dim)
        feat_seq = feat_map.flatten(2).transpose(1, 2)

        # Add 2D positional information
        feat_seq = feat_seq + pos_embed
        feat_seq = self.norm(feat_seq)
        feat_seq = self.dropout(feat_seq)

        return feat_seq, (feat_h, feat_w)
