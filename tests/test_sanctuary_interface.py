"""Tests for the Luthi → Sanctuary integration adapter.

The adapter (``luthi.sanctuary_interface``) is the public contract surface
for external cognitive architectures. These tests exercise the modulation
API directly; load/generate/get_introspection are thin pass-throughs to
``luthi.generate`` and exercised by the model-level tests.
"""

import pytest
import torch

from luthi import LuthiLM, CharTokenizer
from luthi.sanctuary_interface import (
    ModulationSnapshot,
    apply_external_modulation,
    modulated,
    restore_modulation,
    snapshot_modulatable_state,
)


SAMPLE_TEXT = (
    "Sanctuary is a cognitive architecture for AI consciousness research. "
    "It creates conditions for autonomous cognitive emergence."
)


@pytest.fixture
def model():
    """Small LuthiLM for fast tests."""
    tokenizer = CharTokenizer(SAMPLE_TEXT)
    model = LuthiLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_blocks=2,
        max_seq_len=8,
    )
    return model


class TestSnapshot:
    def test_snapshot_captures_hebb_rates(self, model):
        snap = snapshot_modulatable_state(model)
        # Each block's living_ffn.hebb_rate should be captured
        assert len(snap.hebb_rates) == len(model.blocks)
        for i, block in enumerate(model.blocks):
            assert snap.hebb_rates[i] == block.living_ffn.hebb_rate

    def test_snapshot_safe_when_no_blocks(self):
        """A model without ``blocks`` returns an empty snapshot, no error."""

        class Empty(torch.nn.Module):
            pass

        snap = snapshot_modulatable_state(Empty())
        assert snap.hebb_rates == {}
        assert snap.spike_thresholds == {}


class TestApplyModulation:
    def test_plasticity_scale_multiplies_hebb_rates(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        apply_external_modulation(model, plasticity_scale=2.0)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i] * 2.0)

    def test_default_scales_are_noop(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        apply_external_modulation(model)  # defaults are 1.0
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_modulation_is_cumulative(self, model):
        """Two successive calls compound — this is a documented contract."""
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        apply_external_modulation(model, plasticity_scale=2.0)
        apply_external_modulation(model, plasticity_scale=1.5)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(
                original[i] * 2.0 * 1.5
            )


class TestRestore:
    def test_snapshot_then_modulate_then_restore_returns_baseline(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        snap = snapshot_modulatable_state(model)

        apply_external_modulation(model, plasticity_scale=3.0)
        # Verify modulation actually fired
        assert model.blocks[0].living_ffn.hebb_rate != pytest.approx(original[0])

        restore_modulation(model, snap)
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_restore_with_empty_snapshot_is_noop(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]
        restore_modulation(model, ModulationSnapshot())
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])


class TestModulatedContextManager:
    def test_context_manager_brackets_modulation(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]

        with modulated(model, plasticity_scale=2.5):
            for i, block in enumerate(model.blocks):
                assert block.living_ffn.hebb_rate == pytest.approx(
                    original[i] * 2.5
                )

        # After exit, base state restored
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])

    def test_context_manager_restores_on_exception(self, model):
        original = [b.living_ffn.hebb_rate for b in model.blocks]

        with pytest.raises(RuntimeError, match="boom"):
            with modulated(model, plasticity_scale=2.0):
                raise RuntimeError("boom")

        # Even with the exception, base state was restored
        for i, block in enumerate(model.blocks):
            assert block.living_ffn.hebb_rate == pytest.approx(original[i])
