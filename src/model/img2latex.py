"""Full Sequence-to-Sequence Image-to-LaTeX Model.

Combines ResNetEncoder (visual feature extractor with 2D positional encoding)
and TransformerDecoder (cross-attention sequence generator).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import ResNetEncoder
from src.model.decoder import TransformerDecoder


class Img2Latex(nn.Module):
    """End-to-End Handwritten Mathematical Expression Recognition Model."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 256,
        decoder_layers: int = 4,
        heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_tokens: int = 256,
        pad_id: int = 0,
        start_id: int = 1,
        end_id: int = 2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_tokens = max_tokens
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id

        # Visual encoder
        self.encoder = ResNetEncoder(
            model_dim=model_dim,
            pretrained=pretrained_backbone,
            dropout=dropout,
        )

        # Sequence decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_layers=decoder_layers,
            num_heads=heads,
            feedforward_dim=feedforward_dim,
            max_tokens=max_tokens,
            dropout=dropout,
            pad_id=pad_id,
        )

        # Loss function ignoring <pad> tokens
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract visual feature sequence from input images."""
        return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for teacher-forced training.
        
        Args:
            images: Batch of images (B, 3, H, W).
            targets: Batch of token IDs (B, L) including <start> and <end>.
            
        Returns:
            loss: Cross-entropy loss scalar.
            logits: Predicted token logits (B, L - 1, vocab_size).
        """
        # Encoder: (B, S_enc, model_dim)
        memory, _ = self.encoder(images)

        # Teacher forcing: feed tokens 0 to L-2 into decoder
        dec_input = targets[:, :-1]
        # Ground truth labels to predict are tokens 1 to L-1
        dec_target = targets[:, 1:]

        # Decoder forward pass
        logits = self.decoder(dec_input, memory)

        # Flatten for cross-entropy loss: (B * (L - 1), vocab_size) vs (B * (L - 1))
        loss = self.criterion(logits.reshape(-1, self.vocab_size), dec_target.reshape(-1))

        return loss, logits

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """Greedy autoregressive decoding for fast evaluation."""
        self.eval()
        batch_size = images.shape[0]
        device = images.device
        max_len = max_length or self.max_tokens

        # Encode images once
        memory, _ = self.encoder(images)

        # Initialize sequence with <start> token for all batch elements
        generated = torch.full((batch_size, 1), self.start_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            logits = self.decoder(generated, memory)
            # Pick highest probability token at the most recent step
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

            # If finished, keep emitting pad
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, self.pad_id), next_token)
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == self.end_id)
            if finished.all():
                break

        # Convert to list of token IDs per batch item
        result = []
        for i in range(batch_size):
            tokens = generated[i].tolist()
            result.append(tokens)

        return result
