"""Tests for `get_introspection` across v1 and v2 substrates.

The introspection function is the wire that flows substrate state into
Sanctuary's `ExperientialSignals.knowledge_signals`. Different living-
layer implementations expose different observable state — these tests
lock in which fields appear on which substrate, so a future change
that breaks one path is caught loudly.

Background: until 2026-05-25 the introspection only read v1-shape
fields (plasticity, set_point, excitability, membrane, spike,
refractory, episodes). On v2 (predictive coding) the spiking/
excitability fields are silently absent, AND v2's native signals
(error_acc, prediction Frobenius, precision EMA) were not read at
all. This test file documents both halves of that fix.
"""

from __future__ import annotations

import pytest

from luthi import CharTokenizer, LuthiLM
from luthi.sanctuary_interface import get_introspection
from luthi.v2 import PredictiveCodingLM


SAMPLE_TEXT = "The quiet room. The light through the window."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def v1_model():
    """Small v1 (Hebbian) model. Non-spiking — has plasticity but no
    membrane/spike attrs."""
    tokenizer = CharTokenizer(SAMPLE_TEXT)
    return LuthiLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16, n_blocks=2, max_seq_len=16,
    )


@pytest.fixture
def v2_model():
    """Small v2 (predictive coding) model. PC layer has plasticity +
    set_point + error_acc + prediction + precision."""
    tokenizer = CharTokenizer(SAMPLE_TEXT)
    return PredictiveCodingLM(
        vocab_size=tokenizer.vocab_size,
        d_model=16, n_blocks=2, max_seq_len=16,
    )


# ---------------------------------------------------------------------------
# Shared fields (populate on both v1 and v2)
# ---------------------------------------------------------------------------


class TestSharedFields:
    def test_v1_populates_plasticity_and_drift(self, v1_model):
        state = get_introspection(v1_model)
        assert len(state["blocks"]) == 2
        for block in state["blocks"]:
            assert "plasticity_mean" in block
            assert "set_point_drift" in block

    def test_v2_populates_plasticity_and_drift(self, v2_model):
        state = get_introspection(v2_model)
        assert len(state["blocks"]) == 2
        for block in state["blocks"]:
            assert "plasticity_mean" in block
            assert "set_point_drift" in block


# ---------------------------------------------------------------------------
# v2-specific signals — the gap closed on 2026-05-25
# ---------------------------------------------------------------------------


class TestV2SpecificSignals:
    """These tests are the contract: v2's native signals must appear
    in the introspection output. If a future refactor of
    PredictiveCodingLayer renames or removes these attrs, these
    tests fail loudly so we know to update the introspection wire
    rather than silently dropping the signal.
    """

    def test_v2_exposes_error_acc(self, v2_model):
        state = get_introspection(v2_model)
        for block in state["blocks"]:
            assert "error_acc_mean" in block
            assert "error_acc_max" in block
            # error_acc starts at zero; both stats should be finite.
            assert isinstance(block["error_acc_mean"], float)
            assert isinstance(block["error_acc_max"], float)

    def test_v2_exposes_pred_frob(self, v2_model):
        state = get_introspection(v2_model)
        for block in state["blocks"]:
            assert "pred_frob" in block
            # Frobenius norm is non-negative.
            assert block["pred_frob"] >= 0.0

    def test_v2_exposes_precision_mean(self, v2_model):
        state = get_introspection(v2_model)
        for block in state["blocks"]:
            assert "precision_mean" in block
            assert isinstance(block["precision_mean"], float)


# ---------------------------------------------------------------------------
# v2 lacks v1-only fields (negative tests)
# ---------------------------------------------------------------------------


class TestV2LacksV1OnlyFields:
    """PredictiveCodingLayer is not a spiking variant. The hasattr
    gating in get_introspection should silently skip the v1 fields
    that have no v2 analogue.
    """

    def test_v2_no_membrane_or_spike(self, v2_model):
        state = get_introspection(v2_model)
        for block in state["blocks"]:
            assert "membrane_mean" not in block
            assert "spike_fraction" not in block
            assert "refractory_fraction" not in block

    def test_v2_no_excitability(self, v2_model):
        state = get_introspection(v2_model)
        for block in state["blocks"]:
            assert "excitability_mean" not in block


# ---------------------------------------------------------------------------
# v1 does NOT have v2-specific fields (regression guard)
# ---------------------------------------------------------------------------


class TestV1LacksV2OnlyFields:
    """v1's LivingLayerV6 doesn't have error_acc / prediction matrix /
    precision EMA. The new v2 fields must NOT appear on v1
    introspection output — otherwise the gating is broken.
    """

    def test_v1_no_error_acc(self, v1_model):
        state = get_introspection(v1_model)
        for block in state["blocks"]:
            assert "error_acc_mean" not in block
            assert "error_acc_max" not in block

    def test_v1_no_pred_frob(self, v1_model):
        state = get_introspection(v1_model)
        for block in state["blocks"]:
            assert "pred_frob" not in block

    def test_v1_no_precision_mean(self, v1_model):
        state = get_introspection(v1_model)
        for block in state["blocks"]:
            assert "precision_mean" not in block
