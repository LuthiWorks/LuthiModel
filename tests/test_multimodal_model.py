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
# Small vision dims so the encoder is cheap to instantiate and forward.
# image_size / patch_size = 4, so 16 vision tokens.
VISION_IMAGE_SIZE = 32
VISION_PATCH_SIZE = 8
N_VISION_TOKENS = (VISION_IMAGE_SIZE // VISION_PATCH_SIZE) ** 2  # 16
N_AUDIO_TOKENS_PRE = 8  # small fixed count for pre-encoded audio fixture


@pytest.fixture
def model():
    return MultimodalLuthiLM(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_blocks=N_BLOCKS,
        max_seq_len=SEQ_LEN,
        max_audio_tokens=100,
        audio_patch_frames=16,
        # Small vision config so vision tests don't blow up runtime/memory.
        vision_image_size=VISION_IMAGE_SIZE,
        vision_patch_size=VISION_PATCH_SIZE,
        max_vision_tokens=N_VISION_TOKENS,
        spike_threshold=0.1,  # low threshold so spikes fire at d_model=16
    )


@pytest.fixture
def text_tokens():
    return torch.randint(0, VOCAB, (2, SEQ_LEN))


@pytest.fixture
def audio():
    return torch.randn(2, AUDIO_SAMPLES)


@pytest.fixture
def image():
    """Small RGB image batch — shape (B, 3, H, W)."""
    return torch.randn(2, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)


@pytest.fixture
def vision_tokens_pre():
    """Pre-encoded vision tokens — bypass the vision encoder."""
    return torch.randn(2, N_VISION_TOKENS, D_MODEL)


@pytest.fixture
def audio_tokens_pre():
    """Pre-encoded audio tokens — bypass the audio encoder."""
    return torch.randn(2, N_AUDIO_TOKENS_PRE, D_MODEL)


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


def test_apply_living_errors_actually_moves_weights(model, text_tokens, audio):
    """apply_living_errors() with a real backward pass moves living weights.

    The previous test only verifies the call doesn't raise. This one
    snapshots the living-FFN weights, runs a full forward + backward +
    apply_living_errors cycle, and asserts at least one block's weights
    actually moved. Otherwise error-directed learning would be silently
    a no-op and nobody would know.
    """
    model.train()
    weights_before = [b.living_ffn.weight.clone() for b in model.blocks]

    logits = model(text_tokens, audio_waveform=audio)
    target = torch.randint(0, VOCAB, (2, SEQ_LEN))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()
    model.apply_living_errors()

    # At least one block's living-FFN weights should have moved.
    moved = [
        not torch.allclose(b.living_ffn.weight, w_before)
        for b, w_before in zip(model.blocks, weights_before)
    ]
    assert any(moved), (
        "apply_living_errors() did not move any living-FFN weights — "
        "error-directed learning is silently a no-op."
    )


def test_apply_living_errors_expect_grad_raises_without_backward(model, text_tokens, audio):
    """expect_grad=True surfaces the contract violation when backward is skipped.

    The training callsites pass expect_grad=True so a refactor that
    breaks the residual gradient pathway crashes loud rather than
    silently no-opping. Verify the multimodal model honors the same
    contract that the unimodal HybridBlock does.
    """
    model.train()
    # Forward only — no backward, so residual grad is missing.
    _ = model(text_tokens, audio_waveform=audio)
    with pytest.raises(RuntimeError, match="residual gradient path"):
        model.apply_living_errors(expect_grad=True)


def test_apply_living_errors_default_silent_without_backward(model, text_tokens, audio):
    """Default expect_grad=False preserves the silent-no-op behavior.

    The inference/generation pathway calls apply_living_errors() without
    a preceding backward(); that needs to remain a clean no-op so living
    inference doesn't crash mid-generate.
    """
    model.eval()
    with torch.no_grad():
        _ = model(text_tokens, audio_waveform=audio)
    # Default call: no backward, no expect_grad, must not raise.
    model.apply_living_errors()


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


def test_backward_spike_priming_affects_membrane_potential(model, text_tokens, audio):
    """Backward spike priming actually changes the lower block's membrane state.

    multimodal_model.py:247-251 adds backward_spikes * spike_scale * 0.3 to
    each lower block's membrane_potential during the top-down sweep. If the
    priming doesn't propagate to membrane_potential, the design claim that
    "downstream feedback shapes upstream excitability" would be untestable
    talk. This test pins the contract: with priming on, the lower blocks'
    post-forward membrane state differs from with priming off, all else
    held equal.
    """
    # Snapshot every buffer that the forward pass might mutate, so we can
    # restore an identical starting state and isolate the priming effect.
    def snapshot(m):
        return {
            i: {
                "weight": m.blocks[i].living_ffn.weight.clone(),
                "set_point": m.blocks[i].living_ffn.set_point.clone(),
                "plasticity": m.blocks[i].living_ffn.plasticity.clone(),
                "momentum": m.blocks[i].living_ffn.momentum.clone(),
                "membrane_potential": m.blocks[i].living_ffn.membrane_potential.clone(),
                "input_avg_mag": m.blocks[i].living_ffn.input_avg_mag.clone(),
                "excitability_acc": m.blocks[i].living_ffn.excitability_acc.clone(),
            }
            for i in range(len(m.blocks))
        }

    def restore(m, snap):
        for i, buffers in snap.items():
            for name, val in buffers.items():
                getattr(m.blocks[i].living_ffn, name).copy_(val)

    model.train()
    starting_state = snapshot(model)

    # Run 1: backward priming ON
    model.backward_pass_enabled = True
    torch.manual_seed(0)
    _ = model(text_tokens, audio_waveform=audio)
    membrane_with_priming = [
        b.living_ffn.membrane_potential.clone() for b in model.blocks
    ]

    # Reset to the exact starting state
    restore(model, starting_state)

    # Run 2: backward priming OFF — same input, same starting state
    model.backward_pass_enabled = False
    torch.manual_seed(0)
    _ = model(text_tokens, audio_waveform=audio)
    membrane_without_priming = [
        b.living_ffn.membrane_potential.clone() for b in model.blocks
    ]

    # Lower blocks (every block except the last) receive backward priming;
    # at least one of them should show a different membrane state.
    differences = [
        not torch.allclose(membrane_with_priming[i], membrane_without_priming[i])
        for i in range(len(model.blocks) - 1)
    ]
    assert any(differences), (
        "Backward spike priming did not change membrane_potential on any "
        "lower block. The priming code at multimodal_model.py:247-251 is "
        "either not running or its effect is not reaching membrane_potential."
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


# --- Vision modality ---
#
# These tests cover the vision path of MultimodalLuthiLM. The 1024d/2-block
# model spent ~10 epochs training on COCO with a vision encoder, but no
# tests in this file previously exercised the vision branch. Without
# coverage, a refactor that breaks vision routing would silently pass.


def test_vision_output_shape(model, text_tokens, image):
    """Image + text produces logits at text positions only."""
    model.train()
    logits = model(text_tokens, image=image)
    assert logits.shape == (2, SEQ_LEN, VOCAB)


def test_vision_only_no_nan(model, text_tokens, image):
    """Vision + text forward stays finite."""
    model.train()
    logits = model(text_tokens, image=image)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_gradients_reach_vision_encoder(model, text_tokens, image):
    """Gradients from text loss flow back through to the vision encoder."""
    model.train()
    logits = model(text_tokens, image=image)
    target = torch.randint(0, VOCAB, (2, SEQ_LEN))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()

    grad = model.vision_encoder.patch_embed.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_vision_influences_text_output(model, text_tokens):
    """Different images produce different text logits."""
    model.eval()
    with torch.no_grad():
        image_a = torch.randn(2, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)
        image_b = torch.randn(2, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE) * 5
        logits_a = model(text_tokens, image=image_a)
        logits_b = model(text_tokens, image=image_b)
    assert (logits_a - logits_b).abs().mean().item() > 0


def test_living_weights_self_modify_vision(model, text_tokens, image):
    """Living-FFN weights change after vision + text forward passes."""
    model.train()
    weight_before = model.blocks[0].living_ffn.weight.clone()
    # A few passes so spikes fire and Hebbian self-modification opens.
    for _ in range(3):
        _ = model(text_tokens, image=image)
    assert not torch.allclose(
        model.blocks[0].living_ffn.weight, weight_before
    )


def test_all_modalities(model, text_tokens, audio, image):
    """Audio + vision + text together yields valid logits."""
    model.train()
    logits = model(text_tokens, audio_waveform=audio, image=image)
    assert logits.shape == (2, SEQ_LEN, VOCAB)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_vision_tokens_input(model, text_tokens, vision_tokens_pre):
    """Pre-encoded vision tokens skip the encoder and still feed the trunk."""
    # Re-init the encoder's gradients so we can assert it stays untouched.
    model.train()
    model.zero_grad()
    logits = model(text_tokens, vision_tokens=vision_tokens_pre)
    assert logits.shape == (2, SEQ_LEN, VOCAB)

    # Backprop and confirm the encoder did not receive gradient — pre-encoded
    # tokens bypass it entirely.
    target = torch.randint(0, VOCAB, (2, SEQ_LEN))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()
    assert model.vision_encoder.patch_embed.weight.grad is None or \
        model.vision_encoder.patch_embed.weight.grad.abs().sum() == 0


def test_audio_tokens_input(model, text_tokens, audio_tokens_pre):
    """Pre-encoded audio tokens skip the encoder and still feed the trunk."""
    model.train()
    model.zero_grad()
    logits = model(text_tokens, audio_tokens=audio_tokens_pre)
    assert logits.shape == (2, SEQ_LEN, VOCAB)

    target = torch.randint(0, VOCAB, (2, SEQ_LEN))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()
    # Audio encoder must not receive gradient when tokens were pre-encoded.
    assert model.audio_encoder.patch_embed.weight.grad is None or \
        model.audio_encoder.patch_embed.weight.grad.abs().sum() == 0
