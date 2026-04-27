"""Tests for the hybrid block (Phase 2).

Validates the combined architecture: scalar attention + living FFN + episode store.
Properties to verify:
1. The block is non-feedforward (output changes on consecutive passes)
2. Attention weights train via backprop
3. Living FFN self-modifies but is NOT touched by the optimizer
4. Episode store records and recalls experiences
5. No NaN or instability
6. Error-directed learning works through the hybrid block
7. Blocks can be stacked
"""

import torch
import torch.nn as nn
import pytest
from luthi import HybridBlock, ScalarAttention, EpisodeStore


@pytest.fixture
def block():
    torch.manual_seed(42)
    return HybridBlock(d_model=16)


@pytest.fixture
def block_64():
    torch.manual_seed(42)
    return HybridBlock(d_model=64)


# ── Non-Feedforward ─────────────────────────────────────────────────


class TestNonFeedforward:
    def test_consecutive_passes_differ(self, block):
        """The hybrid block must NOT be a pure function."""
        x = torch.randn(1, 4, 16)
        out1 = block(x)
        out2 = block(x)
        diff = (out2 - out1).abs().mean().item()
        assert diff > 0, "Hybrid block must be non-feedforward"

    def test_living_weights_change(self, block):
        """Living FFN weights must change during forward pass."""
        weights_before = block.living_ffn.weight.clone()
        x = torch.randn(2, 4, 16)
        block(x)
        diff = (block.living_ffn.weight - weights_before).abs().sum().item()
        assert diff > 0


# ── Attention Training ──────────────────────────────────────────────


class TestAttentionTraining:
    def test_attention_has_parameters(self, block):
        """Attention Q/K/V/O weights must be nn.Parameters (trainable)."""
        param_names = {n for n, _ in block.attention.named_parameters()}
        assert "q_proj.weight" in param_names
        assert "k_proj.weight" in param_names
        assert "v_proj.weight" in param_names
        assert "o_proj.weight" in param_names

    def test_living_ffn_has_no_parameters(self, block):
        """Living FFN must have NO nn.Parameters — all buffers."""
        ffn_params = list(block.living_ffn.parameters())
        assert len(ffn_params) == 0, (
            f"Living FFN should have 0 parameters, got {len(ffn_params)}"
        )

    def test_gradients_flow_to_attention(self, block):
        """Backprop must produce gradients for attention weights."""
        x = torch.randn(2, 4, 16)
        out = block(x)
        loss = out.sum()
        loss.backward()

        for name, param in block.attention.named_parameters():
            assert param.grad is not None, f"No gradient for attention.{name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for attention.{name}"

    def test_optimizer_updates_attention(self, block):
        """An optimizer step must change attention weights."""
        optimizer = torch.optim.SGD(block.parameters(), lr=0.01)
        q_before = block.attention.q_proj.weight.data.clone()

        x = torch.randn(2, 4, 16)
        out = block(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        q_after = block.attention.q_proj.weight.data
        diff = (q_after - q_before).abs().sum().item()
        assert diff > 0, "Optimizer must update attention weights"


# ── Episode Store ────────────────────────────────────────────────────


class TestEpisodeStore:
    def test_episodes_accumulate(self, block):
        """Episodes should be stored after processing."""
        for _ in range(10):
            x = torch.randn(2, 4, 16) * 3.0
            block(x)
        assert block.episode_store.episode_count.item() > 0

    def test_episode_capacity(self):
        """Episode store must respect capacity limit."""
        torch.manual_seed(42)
        block = HybridBlock(d_model=16, num_episodes=4)
        for _ in range(20):
            x = torch.randn(2, 4, 16) * 3.0
            block(x)
        assert block.episode_store.episode_count.item() <= 4

    def test_standalone_episode_store(self):
        """EpisodeStore should work independently."""
        torch.manual_seed(42)
        store = EpisodeStore(d_model=16, num_episodes=8, salience_threshold=0.0)

        x_input = torch.randn(2, 4, 16)
        x_output = torch.randn(2, 4, 16)

        # Should pass through and store
        result = store(x_input, x_output)
        assert result.shape == x_output.shape
        assert store.episode_count.item() > 0


# ── Stability ────────────────────────────────────────────────────────


class TestStability:
    def test_no_nan_many_passes(self, block):
        """200 passes must not produce NaN."""
        for _ in range(200):
            x = torch.randn(2, 4, 16)
            out = block(x)
            assert not torch.isnan(out).any(), "NaN in output"

    def test_no_nan_at_64d(self, block_64):
        """Stability at 64 dimensions."""
        for _ in range(100):
            x = torch.randn(2, 8, 64)
            out = block_64(x)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


# ── Error-Directed Learning ─────────────────────────────────────────


class TestErrorDirected:
    def test_apply_living_error_via_gradient(self, block):
        """Error-directed learning should work via autograd gradient."""
        x = torch.randn(2, 4, 16)
        target = torch.randn(2, 4, 16)

        weights_before = block.living_ffn.weight.clone()

        out = block(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()

        # Apply error-directed learning using the captured gradient
        block.apply_living_error()

        weights_after = block.living_ffn.weight.clone()
        # The error-directed update should change weights beyond
        # what Hebbian already did in forward
        # (We can't easily isolate the error-directed change from Hebbian,
        # but we can verify the method runs without error)
        assert not torch.isnan(weights_after).any()

    def test_apply_living_error_direct(self, block):
        """Direct error signal should update living weights."""
        x = torch.randn(2, 4, 16)
        block(x)

        weights_before = block.living_ffn.weight.clone()
        error = torch.randn(2, 4, 16)
        block.apply_living_error_direct(error)

        diff = (block.living_ffn.weight - weights_before).abs().sum().item()
        assert diff > 0, "Direct error should update living weights"

    def test_apply_living_error_expect_grad_silent_when_default(self, block):
        """With default expect_grad=False, missing grad is a silent no-op.

        This preserves the inference/generation contract where
        apply_living_errors() may be called without backward() having run
        (e.g. from luthi.generate.generate_text in living mode).
        """
        x = torch.randn(2, 4, 16)
        with torch.no_grad():
            block(x)
        # No backward — _ffn_output.grad is None. Default call must not raise.
        block.apply_living_error()

    def test_apply_living_error_expect_grad_raises_on_missing(self, block):
        """With expect_grad=True, missing grad raises — catches broken residual.

        The training callsites pass expect_grad=True so a refactor that
        breaks the residual gradient pathway crashes loud rather than
        silently no-opping (which would mean error-directed learning
        stops working without anyone noticing).
        """
        x = torch.randn(2, 4, 16, requires_grad=True)
        out = block(x)
        # Forward ran with grad enabled, so _ffn_output is stored and
        # retain_grad() was called. But no backward() — grad stays None.
        assert block._ffn_output is not None
        assert block._ffn_output.grad is None

        with pytest.raises(RuntimeError, match="residual gradient path"):
            block.apply_living_error(expect_grad=True)


# ── Stacking ─────────────────────────────────────────────────────────


class TestStacking:
    def test_two_blocks_stack(self):
        """Two hybrid blocks should compose without errors."""
        torch.manual_seed(42)
        block1 = HybridBlock(d_model=16)
        block2 = HybridBlock(d_model=16)

        x = torch.randn(2, 4, 16)
        h = block1(x)
        out = block2(h)

        assert out.shape == (2, 4, 16)
        assert not torch.isnan(out).any()

    def test_stacked_blocks_stable(self):
        """Stacked blocks should remain stable over many passes."""
        torch.manual_seed(42)
        blocks = nn.ModuleList([HybridBlock(d_model=16) for _ in range(3)])

        for _ in range(100):
            x = torch.randn(2, 4, 16)
            for b in blocks:
                x = b(x)
            assert not torch.isnan(x).any()
            assert not torch.isinf(x).any()

    def test_stacked_gradients(self):
        """Gradients should flow through stacked blocks to all attention layers."""
        torch.manual_seed(42)
        blocks = nn.ModuleList([HybridBlock(d_model=16) for _ in range(3)])

        x = torch.randn(2, 4, 16)
        h = x
        for b in blocks:
            h = b(h)
        loss = h.sum()
        loss.backward()

        for i, b in enumerate(blocks):
            for name, param in b.attention.named_parameters():
                assert param.grad is not None, (
                    f"No gradient for block {i} attention.{name}"
                )


# ── Input Shape Handling ─────────────────────────────────────────────


class TestShapes:
    def test_standard_3d_input(self, block):
        """[batch, seq_len, d_model] should work."""
        x = torch.randn(4, 8, 16)
        out = block(x)
        assert out.shape == (4, 8, 16)

    def test_single_token(self, block):
        """[batch, 1, d_model] should work (single token)."""
        x = torch.randn(1, 1, 16)
        out = block(x)
        assert out.shape == (1, 1, 16)


# ── Diagnostics ──────────────────────────────────────────────────────


class TestDiagnostics:
    def test_aliveness_returns_combined_metrics(self, block):
        """aliveness() should include both living FFN and episode store metrics."""
        x = torch.randn(2, 4, 16)
        block(x)
        metrics = block.aliveness()
        assert "weight_mean" in metrics  # From living FFN
        assert "output_episodes_stored" in metrics  # From episode store


# ── Attention Standalone ─────────────────────────────────────────────


class TestScalarAttention:
    def test_causal_masking(self):
        """Causal attention: position i should not attend to position j > i."""
        torch.manual_seed(42)
        attn = ScalarAttention(d_model=16)

        # Create input where position 0 has all zeros, position 1 has data
        x = torch.zeros(1, 2, 16)
        x[0, 1, :] = 1.0

        out_causal = attn(x, causal=True)
        out_full = attn(x, causal=False)

        # With causal masking, position 0 can't see position 1,
        # so its output should differ from non-causal
        pos0_diff = (out_causal[0, 0] - out_full[0, 0]).abs().sum().item()
        assert pos0_diff > 0, "Causal masking should affect position 0's output"

    def test_output_shape(self):
        torch.manual_seed(42)
        attn = ScalarAttention(d_model=32)
        x = torch.randn(4, 10, 32)
        out = attn(x)
        assert out.shape == (4, 10, 32)
