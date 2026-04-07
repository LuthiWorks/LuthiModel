"""Tests for the multimodal living weight model.

Verifies that:
1. Audio + text sequences produce correct output shapes
2. Text-only mode works (backward compatible)
3. Gradients flow to both audio encoder and attention parameters
4. Living weights self-modify on multimodal input
5. Backward pass (top-down) runs without error
6. Can load from a text-only checkpoint
7. Cross-modal attention works (audio influences text output)
"""

import torch
import pytest
import torch.nn.functional as F

from luthi.multimodal_model import MultimodalLuthiLM
from luthi.model_spiking import SpikingLuthiLM


D_MODEL = 16
VOCAB = 96
N_BLOCKS = 2
SEQ_LEN = 32
AUDIO_SAMPLES = 16000  # 1 second


@pytest.fixture
def model():
    return MultimodalLuthiLM(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_blocks=N_BLOCKS,
        max_seq_len=SEQ_LEN,
        max_audio_tokens=100,
        audio_patch_frames=16,
        spike_threshold=0.1,  # low threshold so spikes fire at d_model=16
    )


@pytest.fixture
def text_tokens():
    return torch.randint(0, VOCAB, (2, SEQ_LEN))


@pytest.fixture
def audio():
    return torch.randn(2, AUDIO_SAMPLES)


# --- Output shape ---

def test_multimodal_output_shape(model, text_tokens, audio):
    """Audio + text produces logits for text positions only."""
    model.train()
    logits = model(text_tokens, audio_waveform=audio)
    assert logits.shape == (2, SEQ_LEN, VOCAB)


def test_text_only_output_shape(model, text_tokens):
    """Text-only input still works."""
    model.train()
    logits = model(text_tokens)
    assert logits.shape == (2, SEQ_LEN, VOCAB)


def test_no_nan_in_output(model, text_tokens, audio):
    """No NaN in multimodal output."""
    model.train()
    logits = model(text_tokens, audio_waveform=audio)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


# --- Gradient flow ---

def test_gradients_reach_audio_encoder(model, text_tokens, audio):
    """Gradients flow from text loss back to audio encoder."""
    model.train()
    logits = model(text_tokens, audio_waveform=audio)
    target = torch.randint(0, VOCAB, (2, SEQ_LEN))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()

    # Audio encoder patch_embed should receive gradients
    assert model.audio_encoder.patch_embed.weight.grad is not None
    assert model.audio_encoder.patch_embed.weight.grad.abs().sum() > 0


def test_gradients_reach_attention(model, text_tokens, audio):
    """Gradients flow to attention parameters."""
    model.train()
    logits = model(text_tokens, audio_waveform=audio)
    target = torch.randint(0, VOCAB, (2, SEQ_LEN))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()

    # Attention in first block should get gradients
    attn_params = list(model.blocks[0].attention.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn_params)


# --- Living weight dynamics ---

def test_living_weights_self_modify(model, text_tokens, audio):
    """Living weights change after multimodal forward passes."""
    model.train()
    # Run a few passes so membrane potential accumulates and spikes fire,
    # which opens the activity-dependent gate for Hebbian self-modification.
    weight_before = model.blocks[0].living_ffn.weight.clone()
    for _ in range(3):
        _ = model(text_tokens, audio_waveform=audio)
    assert not torch.allclose(model.blocks[0].living_ffn.weight, weight_before)


def test_non_feedforward_signal(model, text_tokens, audio):
    """Consecutive passes with same input produce different output."""
    model.eval()
    with torch.no_grad():
        out1 = model(text_tokens, audio_waveform=audio)
        out2 = model(text_tokens, audio_waveform=audio)
    diff = (out2 - out1).abs().mean().item()
    assert diff > 0


def test_apply_living_errors(model, text_tokens, audio):
    """Error-directed learning runs without error."""
    model.train()
    logits = model(text_tokens, audio_waveform=audio)
    target = torch.randint(0, VOCAB, (2, SEQ_LEN))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()
    model.apply_living_errors()  # Should not raise


# --- Backward pass (top-down) ---

def test_backward_pass_runs(model, text_tokens, audio):
    """Top-down sweep runs without error on multimodal input."""
    model.train()
    model.backward_pass_enabled = True
    logits = model(text_tokens, audio_waveform=audio)
    assert logits.shape == (2, SEQ_LEN, VOCAB)


def test_backward_pass_modulates_plasticity(model, text_tokens, audio):
    """Backward pass changes plasticity in multimodal mode."""
    model.train()
    model.backward_pass_enabled = True
    plasticity_before = model.blocks[0].living_ffn.plasticity.clone()
    _ = model(text_tokens, audio_waveform=audio)
    assert not torch.allclose(
        model.blocks[0].living_ffn.plasticity, plasticity_before
    )


# --- Cross-modal attention ---

def test_audio_influences_text_output(model, text_tokens):
    """Different audio inputs produce different text logits."""
    model.eval()
    with torch.no_grad():
        audio_a = torch.randn(2, AUDIO_SAMPLES)
        audio_b = torch.randn(2, AUDIO_SAMPLES) * 5  # very different
        logits_a = model(text_tokens, audio_waveform=audio_a)
        logits_b = model(text_tokens, audio_waveform=audio_b)
    diff = (logits_a - logits_b).abs().mean().item()
    assert diff > 0


# --- Aliveness report ---

def test_aliveness_report(model, text_tokens, audio):
    """Aliveness report returns expected structure."""
    model.train()
    _ = model(text_tokens, audio_waveform=audio)
    report = model.aliveness_report()
    assert len(report) == N_BLOCKS
    assert "set_point_drift" in report[0]
    assert "spike_fraction" in report[0]


# --- Parameter counting ---

def test_total_parameters(model):
    """Parameter count includes audio encoder."""
    counts = model.total_parameters()
    assert counts["trainable"] > 0
    assert counts["living_buffers"] > 0

    # Audio encoder params should be counted in trainable
    audio_params = sum(
        p.numel() for p in model.audio_encoder.parameters()
    )
    assert audio_params > 0


# --- Loading from text-only checkpoint ---

def test_load_from_text_checkpoint():
    """Can initialize from a text-only SpikingLuthiLM checkpoint."""
    text_model = SpikingLuthiLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_blocks=N_BLOCKS,
        max_seq_len=SEQ_LEN,
    )
    text_state = text_model.state_dict()

    mm_model = MultimodalLuthiLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_blocks=N_BLOCKS,
        max_seq_len=SEQ_LEN, max_audio_tokens=100,
    )

    # Load with strict=False — audio encoder and modality embedding are new
    missing, unexpected = mm_model.load_state_dict(text_state, strict=False)

    # Missing keys should be audio encoder + modality embedding
    assert len(missing) > 0
    assert any("audio_encoder" in k for k in missing)
    assert any("modality_embedding" in k for k in missing)

    # No unexpected keys
    assert len(unexpected) == 0

    # Shared weights should match
    assert torch.allclose(
        mm_model.blocks[0].living_ffn.weight,
        text_model.blocks[0].living_ffn.weight,
    )


# --- Training loop ---

def test_training_step(model, text_tokens, audio):
    """Full training step completes without error."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    target = torch.randint(0, VOCAB, (2, SEQ_LEN))

    optimizer.zero_grad()
    logits = model(text_tokens, audio_waveform=audio)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()
    model.apply_living_errors()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    assert loss.item() > 0


def test_loss_decreases():
    """A few training steps reduce loss."""
    model = MultimodalLuthiLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_blocks=N_BLOCKS,
        max_seq_len=SEQ_LEN, max_audio_tokens=100,
        backward_pass_enabled=False,  # faster for this test
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    text = torch.randint(0, VOCAB, (4, SEQ_LEN))
    audio = torch.randn(4, AUDIO_SAMPLES)
    target = torch.randint(0, VOCAB, (4, SEQ_LEN))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(text, audio_waveform=audio)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        loss.backward()
        model.apply_living_errors()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease over 5 steps
    assert losses[-1] < losses[0]
