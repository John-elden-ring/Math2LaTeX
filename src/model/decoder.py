"""Transformer Decoder for Math2LaTeX.

Autoregressively decodes mathematical LaTeX tokens conditioned on visual encoder features.
Implements:
- Token embedding + learnable 1D position embedding
- Causal triangular self-attention mask (prevents looking into future tokens)
- Cross-attention over ResNet encoder visual feature memory
- Final vocabulary classification projection
"""

from typing import Optional
import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    """Autoregressive Transformer Decoder for generating LaTeX sequences."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        max_tokens: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.max_tokens = max_tokens
        self.pad_id = pad_id

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_id)

        # Learnable 1D positional embedding
        self.pos_embedding = nn.Embedding(max_tokens, model_dim)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

        # Final vocabulary projection: model_dim -> vocab_size
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal mask to prevent attending to future tokens."""
        # True values will be masked out in PyTorch Transformer
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for teacher-forced training.
        
        Args:
            tgt_tokens: Tensor of shape (B, L_tgt) containing input token IDs.
            memory: Tensor of shape (B, S_enc, model_dim) from visual encoder.
            tgt_mask: Optional causal mask of shape (L_tgt, L_tgt).
            tgt_key_padding_mask: Optional bool mask (B, L_tgt) where True indicates <pad>.
            
        Returns:
            logits: Tensor of shape (B, L_tgt, vocab_size).
        """
        batch_size, seq_len = tgt_tokens.shape
        device = tgt_tokens.device

        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(seq_len, device)

        # Create padding mask if not provided
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt_tokens == self.pad_id)

        # Embed tokens and scale by sqrt(d_model)
        tok_emb = self.token_embedding(tgt_tokens) * (self.model_dim ** 0.5)

        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)

        x = self.dropout(tok_emb + pos_emb)

        # Transformer decoding
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        out = self.norm(out)
        logits = self.fc_out(out)
        return logits
