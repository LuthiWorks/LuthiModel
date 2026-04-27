"""Tests for the vision encoder.

Verifies that:
1. RGB images are converted to d_model token sequences
2. Token count matches the patch grid for the configured image_size / patch_size
3. Gradient flows through trainable components (patch_embed, proj, pos_embed, ln)
4. Edge cases — single sample, larger batches, sub-default image_size
5. No NaN / Inf in output
6. Encoder is deterministic in eval mode
7. CPU-only operation works (DirectML / CUDA-free path)
"""

import torch
import pytest

from luthi.vision_encoder import VisionEncoder


@pytest.fixture
def encoder():
    """Standard 224×224 / patch_size=16 encoder, d_model=64."""
    return VisionEncoder(d_model=64, image_size=224, patch_size=16)


@pytest.fixture
def encoder_small():
    """Small encoder for fast tests — 32×32, patch_size=8 → 16 tokens."""
    return VisionEncoder(
        d_model=16, image_size=32, patch_size=8, max_vision_tokens=64
    )


# --- Output shape ---

def test_output_shape(encoder):
    """Output has shape [batch, n_patches, d_model]."""
    images = torch.randn(2, 3, 224, 224)
    tokens = encoder(images)
    assert tokens.dim() == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[1] == encoder.n_patches  # 196 for 224/16
    assert tokens.shape[2] == 64


def test_output_shape_single_sample(encoder_small):
    """Works with batch size 1."""
    images = torch.randn(1, 3, 32, 32)
    tokens = encoder_small(images)
    assert tokens.shape[0] == 1
    assert tokens.shape[2] == 16


def test_output_shape_large_batch(encoder_small):
    """Works with larger batches."""
    images = torch.randn(8, 3, 32, 32)
    tokens = encoder_small(images)
    assert tokens.shape[0] == 8
    assert tokens.shape[1] == 16  # (32/8)^2


# --- Token count ---

def test_token_count_matches_patch_grid(encoder):
    """n_patches = (image_size / patch_size)^2."""
    assert encoder.n_patches == (224 // 16) ** 2  # 196
    assert encoder.n_tokens_for_image() == 196


def test_token_count_small(encoder_small):
    """Small encoder: 32/8 = 4 → 4×4 = 16 tokens."""
    assert encoder_small.n_patches == 16
    images = torch.randn(1, 3, 32, 32)
    tokens = encoder_small(images)
    assert tokens.shape[1] == 16


def test_token_count_for_various_sizes():
    """Token count formula holds across sizes."""
    for image_size, patch_size in [(64, 16), (96, 16), (128, 16), (224, 14)]:
        enc = VisionEncoder(
            d_model=8,
            image_size=image_size,
            patch_size=patch_size,
            max_vision_tokens=512,
        )
        expected = (image_size // patch_size) ** 2
        assert enc.n_patches == expected
        images = torch.randn(1, 3, image_size, image_size)
        tokens = enc(images)
        assert tokens.shape[1] == expected


# --- Edge cases ---

def test_zero_input_does_not_crash(encoder_small):
    """All-zero image produces a valid output — no NaN, no division by zero."""
    images = torch.zeros(1, 3, 32, 32)
    tokens = encoder_small(images)
    assert not torch.isnan(tokens).any()
    assert not torch.isinf(tokens).any()


def test_extreme_value_input(encoder_small):
    """Large-magnitude input doesn't blow up to NaN/Inf in the LN-normalized output."""
    images = torch.full((1, 3, 32, 32), 100.0)
    tokens = encoder_small(images)
    assert not torch.isnan(tokens).any()
    assert not torch.isinf(tokens).any()


def test_input_below_max_vision_tokens(encoder_small):
    """An image producing fewer than max_vision_tokens patches still works.

    The position embedding must accommodate variable token counts up to
    max_vision_tokens — verify a small image doesn't index out of range.
    """
    # encoder_small has max_vision_tokens=64 and produces 16 tokens — well under.
    images = torch.randn(1, 3, 32, 32)
    tokens = encoder_small(images)
    assert tokens.shape[1] == 16  # uses positions 0..15 of the 64-slot embedding


# --- No NaN ---

def test_no_nan_output(encoder):
    """Output contains no NaN/Inf values for typical inputs."""
    images = torch.randn(4, 3, 224, 224)
    tokens = encoder(images)
    assert not torch.isnan(tokens).any()
    assert not torch.isinf(tokens).any()


# --- Gradient flow ---

def test_gradient_flows_through_encoder(encoder_small):
    """Gradients reach all trainable components."""
    images = torch.randn(2, 3, 32, 32)
    tokens = encoder_small(images)
    loss = tokens.sum()
    loss.backward()

    # patch_embed (Conv2d), proj (Linear), pos_embed (Embedding), ln (LayerNorm)
    assert encoder_small.patch_embed.weight.grad is not None
    assert encoder_small.proj.weight.grad is not None
    assert encoder_small.pos_embed.weight.grad is not None
    assert encoder_small.ln.weight.grad is not None

    assert encoder_small.patch_embed.weight.grad.abs().sum() > 0
    assert encoder_small.proj.weight.grad.abs().sum() > 0
    # pos_embed only sees rows 0..n_tokens-1 — those rows should have grad,
    # rows beyond that should be zero.
    pe_grad = encoder_small.pos_embed.weight.grad
    n_tokens = encoder_small.n_patches
    assert pe_grad[:n_tokens].abs().sum() > 0
    assert pe_grad[n_tokens:].abs().sum() == 0


# --- Determinism ---

def test_deterministic_in_eval_mode(encoder_small):
    """Same input in eval mode produces identical output across calls."""
    encoder_small.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out1 = encoder_small(images)
        out2 = encoder_small(images)
    assert torch.allclose(out1, out2)


# --- CPU compatibility ---

def test_runs_on_cpu():
    """Encoder works entirely on CPU (DirectML / CUDA-free path)."""
    enc = VisionEncoder(d_model=16, image_size=32, patch_size=8, max_vision_tokens=64)
    images = torch.randn(1, 3, 32, 32)
    tokens = enc(images)
    assert tokens.device.type == "cpu"
    assert tokens.shape[2] == 16


# --- Position embedding contract ---

def test_max_vision_tokens_bound(encoder_small):
    """An image producing more tokens than max_vision_tokens raises clearly.

    Currently the encoder will index out of range on the position embedding.
    This pins the contract: callers must size max_vision_tokens >= n_patches
    for whatever images they feed in.
    """
    # encoder_small has max_vision_tokens=64. Build an image that would
    # produce more than 64 patches: 96/8 = 12 → 144 patches. This should
    # fail rather than silently truncate or wrap.
    too_big = torch.randn(1, 3, 96, 96)
    with pytest.raises((IndexError, RuntimeError)):
        encoder_small(too_big)
