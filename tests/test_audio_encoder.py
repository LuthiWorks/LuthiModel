"""Tests for the audio encoder.

Verifies that:
1. Raw waveforms are converted to d_model token sequences
2. Token count matches expected for given audio length
3. Gradient flows through trainable components
4. Edge cases (short audio, batch sizes) are handled
"""

import torch
import pytest

from luthi.audio_encoder import AudioEncoder


@pytest.fixture
def encoder():
    return AudioEncoder(d_model=64, patch_frames=16)


@pytest.fixture
def encoder_small():
    """Small encoder for fast tests."""
    return AudioEncoder(d_model=16, patch_frames=16)


# --- Output shape ---

def test_output_shape(encoder):
    """Output has correct shape [batch, n_tokens, d_model]."""
    waveform = torch.randn(2, 16000)  # 1 second
    tokens = encoder(waveform)
    assert tokens.dim() == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[2] == 64


def test_output_shape_single_sample(encoder):
    """Works with batch size 1."""
    waveform = torch.randn(1, 16000)
    tokens = encoder(waveform)
    assert tokens.shape[0] == 1
    assert tokens.shape[2] == 64


def test_output_shape_large_batch(encoder_small):
    """Works with larger batches."""
    waveform = torch.randn(8, 16000)
    tokens = encoder_small(waveform)
    assert tokens.shape[0] == 8


# --- Token count ---

def test_token_count_one_second(encoder):
    """1 second at 16kHz → expected number of tokens."""
    waveform = torch.randn(1, 16000)
    tokens = encoder(waveform)
    expected = encoder.n_tokens_for_samples(16000)
    assert tokens.shape[1] == expected


def test_token_count_five_seconds(encoder):
    """5 seconds of audio produces ~5x the tokens of 1 second."""
    one_sec = encoder.n_tokens_for_samples(16000)
    five_sec = encoder.n_tokens_for_samples(80000)
    assert five_sec == pytest.approx(one_sec * 5, abs=1)


def test_token_count_formula():
    """n_tokens_for_samples matches the mathematical formula."""
    enc = AudioEncoder(d_model=32, hop_length=160, patch_frames=16)
    n_samples = 32000  # 2 seconds
    expected = (n_samples // 160) // 16  # n_frames // patch_frames
    assert enc.n_tokens_for_samples(n_samples) == expected


# --- Edge cases ---

def test_very_short_audio(encoder):
    """Audio shorter than one patch produces a fallback token."""
    # patch_frames=16 with hop=160 means we need at least 16 frames
    # = 16 * 160 = 2560 samples. Use fewer.
    waveform = torch.randn(1, 100)
    tokens = encoder(waveform)
    assert tokens.shape[1] >= 1
    assert not torch.isnan(tokens).any()


def test_different_audio_lengths(encoder_small):
    """Different audio lengths in a non-batched scenario."""
    for n_samples in [8000, 16000, 48000, 160000]:
        waveform = torch.randn(1, n_samples)
        tokens = encoder_small(waveform)
        assert tokens.dim() == 3
        assert not torch.isnan(tokens).any()


# --- No NaN ---

def test_no_nan_output(encoder):
    """Output contains no NaN values."""
    waveform = torch.randn(4, 32000)
    tokens = encoder(waveform)
    assert not torch.isnan(tokens).any()
    assert not torch.isinf(tokens).any()


# --- Gradient flow ---

def test_gradient_flows_through_encoder(encoder_small):
    """Gradients reach the patch embedding and projection."""
    waveform = torch.randn(2, 16000)
    tokens = encoder_small(waveform)
    loss = tokens.sum()
    loss.backward()

    # patch_embed and proj should have gradients
    assert encoder_small.patch_embed.weight.grad is not None
    assert encoder_small.proj.weight.grad is not None
    assert encoder_small.patch_embed.weight.grad.abs().sum() > 0


# --- CPU compatibility ---

def test_runs_on_cpu():
    """Encoder works entirely on CPU."""
    enc = AudioEncoder(d_model=16)
    waveform = torch.randn(1, 16000)
    tokens = enc(waveform)
    assert tokens.device.type == "cpu"
    assert tokens.shape[2] == 16
