"""Unit tests for Stage 3: ResNetEncoder, TransformerDecoder, and Img2Latex full model."""

import pytest
import torch

from src.model.encoder import ResNetEncoder, create_2d_sinusoidal_embeddings
from src.model.decoder import TransformerDecoder
from src.model.img2latex import Img2Latex


def test_2d_sinusoidal_embeddings():
    H, W, dim = 4, 10, 256
    device = torch.device("cpu")
    pos_emb = create_2d_sinusoidal_embeddings(H, W, dim, device)
    assert pos_emb.shape == (1, H * W, dim)
    assert not torch.isnan(pos_emb).any()


def test_encoder_forward():
    batch_size = 2
    height, width = 128, 256
    model_dim = 256

    encoder = ResNetEncoder(model_dim=model_dim, pretrained=False)
    dummy_images = torch.randn(batch_size, 3, height, width)

    feat_seq, (feat_h, feat_w) = encoder(dummy_images)

    # ResNet-18 downsamples by 32: 128 // 32 = 4, 256 // 32 = 8
    assert feat_h == 4
    assert feat_w == 8
    assert feat_seq.shape == (batch_size, 4 * 8, model_dim)


def test_decoder_forward():
    batch_size = 2
    seq_len = 15
    memory_len = 32
    model_dim = 256
    vocab_size = 100

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_layers=2,
        num_heads=4,
        feedforward_dim=512,
    )

    tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    memory = torch.randn(batch_size, memory_len, model_dim)

    logits = decoder(tgt_tokens, memory)
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_img2latex_full_model_forward():
    batch_size = 2
    vocab_size = 80
    model_dim = 256
    seq_len = 12

    model = Img2Latex(
        vocab_size=vocab_size,
        model_dim=model_dim,
        decoder_layers=2,
        heads=4,
        feedforward_dim=512,
        pretrained_backbone=False,
    )

    dummy_images = torch.randn(batch_size, 3, 128, 192)
    dummy_targets = torch.randint(1, vocab_size, (batch_size, seq_len))
    # Ensure start and end tokens
    dummy_targets[:, 0] = model.start_id
    dummy_targets[:, -1] = model.end_id

    loss, logits = model(dummy_images, dummy_targets)

    assert loss.dim() == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert logits.shape == (batch_size, seq_len - 1, vocab_size)


def test_img2latex_greedy_decode():
    vocab_size = 50
    model = Img2Latex(
        vocab_size=vocab_size,
        model_dim=128,
        decoder_layers=1,
        heads=4,
        feedforward_dim=256,
        pretrained_backbone=False,
    )

    dummy_images = torch.randn(2, 3, 128, 128)
    preds = model.greedy_decode(dummy_images, max_length=10)

    assert len(preds) == 2
    for p in preds:
        assert p[0] == model.start_id
        assert len(p) <= 10
