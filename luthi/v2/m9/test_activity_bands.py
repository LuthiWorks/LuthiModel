"""Unit tests for §A.1 ActivityBands and decoder activity() methods.

Run from project root:
    python -m luthi.v2.m9.test_activity_bands

Spec-correctness properties (per 4.8's 2026-06-11 gate-repairs):
- decoder.activity() reads raw pre-sigmoid magnitudes (not intensity heads)
- Activity varies with the input a_t even at random init -- the F4 root-
  cause fix
- ActivityBands warmup gates the predicates: before min_warmup, K-M9-5
  is reported DISARMED (visible) and external_stasis returns all-False
- After warmup, external_stasis fires per-batch when ALL modalities
  are below their bands
- text_active flips with text activity vs band -- the F1 P3 emission
  signal
- armed_per_modality / k_m9_5_armed surface the §A.1 armed-state log
"""

from __future__ import annotations

import torch
import torch.nn as nn

from luthi.v2.m9.activity_bands import ActivityBandConfig, ActivityBands
from luthi.v2.m9.decoders import (
    AttentionDecoder,
    DecoderRegistry,
    MemoryDecoder,
    TextDecoder,
)


D = 16
V = 32
B = 4


def _build_registry() -> DecoderRegistry:
    return DecoderRegistry(
        text=TextDecoder(nn.Linear(D, V), D, V),
        attention=AttentionDecoder(d_model=D, n_modalities=3),
        memory=MemoryDecoder(d_model=D),
    )


# ---------- decoder.activity() ----------

def test_text_activity_shape_and_positive():
    dec = TextDecoder(nn.Linear(D, V), D, V)
    a = torch.randn(B, D)
    act = dec.activity(a)
    assert act.shape == (B,)
    assert (act >= 0).all()


def test_attention_activity_shape_and_positive():
    dec = AttentionDecoder(d_model=D, n_modalities=3)
    a = torch.randn(B, D)
    act = dec.activity(a)
    assert act.shape == (B,)
    assert (act >= 0).all()


def test_memory_activity_shape_and_positive():
    dec = MemoryDecoder(d_model=D)
    a = torch.randn(B, D)
    act = dec.activity(a)
    assert act.shape == (B,)
    assert (act >= 0).all()


def test_decoder_activity_varies_with_action():
    """The F4 root-cause fix: activity must vary with a_t even at
    random init. Sigmoid intensity heads can sit constant across
    inputs; raw pre-sigmoid activity cannot.
    """
    torch.manual_seed(0)
    reg = _build_registry()
    a1 = torch.randn(B, D) * 3.0
    a2 = torch.randn(B, D) * 3.0
    act1 = reg.activity(a1)
    act2 = reg.activity(a2)
    # At least one modality should differ meaningfully across inputs.
    diffs = [
        (act1[m] - act2[m]).abs().max().item()
        for m in ("text", "attention", "memory")
    ]
    assert max(diffs) > 1e-3, (
        f"raw activity should vary with input a_t at random init, got diffs={diffs}"
    )


def test_registry_activity_returns_three_modalities():
    reg = _build_registry()
    a = torch.randn(B, D)
    act = reg.activity(a)
    assert set(act.keys()) == {"text", "attention", "memory"}
    for v in act.values():
        assert v.shape == (B,)


# ---------- ActivityBands warmup ----------

def test_bands_disarmed_before_warmup():
    bands = ActivityBands(config=ActivityBandConfig(min_warmup=8))
    # Push 7 samples -- one less than min_warmup.
    for _ in range(7):
        bands.observe({
            "text": torch.tensor([1.0]),
            "attention": torch.tensor([1.0]),
            "memory": torch.tensor([1.0]),
        })
    assert not bands.k_m9_5_armed()
    armed = bands.armed_per_modality()
    for m in armed.values():
        assert not m


def test_external_stasis_all_false_before_warmup():
    """The K-M9-5 backstop must not fire against an uncalibrated band
    -- but its disarmed state must be VISIBLE via armed_per_modality.
    """
    bands = ActivityBands(config=ActivityBandConfig(min_warmup=8))
    act = {
        "text": torch.zeros(B),
        "attention": torch.zeros(B),
        "memory": torch.zeros(B),
    }
    stasis = bands.external_stasis(act)
    assert not torch.any(stasis), (
        "external_stasis must be False before warmup (disarmed band)"
    )


# ---------- ActivityBands after warmup ----------

def test_bands_armed_after_warmup():
    bands = ActivityBands(config=ActivityBandConfig(min_warmup=8))
    for _ in range(10):
        bands.observe({
            "text": torch.tensor([1.0]),
            "attention": torch.tensor([1.0]),
            "memory": torch.tensor([1.0]),
        })
    assert bands.k_m9_5_armed()


def test_external_stasis_fires_when_all_silent():
    """After warmup, external_stasis = True for batch element where
    every modality's activity is below threshold.
    """
    bands = ActivityBands(
        config=ActivityBandConfig(min_warmup=4, silence_k=0.5)
    )
    # Warm the bands with activity ~ 1.0 each.
    for _ in range(8):
        bands.observe({
            "text": torch.tensor([1.0]),
            "attention": torch.tensor([1.0]),
            "memory": torch.tensor([1.0]),
        })
    # Probe with activity ~ 0.0 (way below median - silence_k * MAD).
    silent_act = {
        "text": torch.zeros(B),
        "attention": torch.zeros(B),
        "memory": torch.zeros(B),
    }
    stasis = bands.external_stasis(silent_act)
    assert torch.all(stasis), (
        "all-silent activity after warmup should fire external_stasis"
    )


def test_external_stasis_off_when_any_active():
    """One modality above band breaks stasis -- only ALL silent fires."""
    bands = ActivityBands(
        config=ActivityBandConfig(min_warmup=4, silence_k=0.5)
    )
    for _ in range(8):
        bands.observe({
            "text": torch.tensor([1.0]),
            "attention": torch.tensor([1.0]),
            "memory": torch.tensor([1.0]),
        })
    mixed = {
        "text": torch.full((B,), 5.0),  # active
        "attention": torch.zeros(B),
        "memory": torch.zeros(B),
    }
    assert not torch.any(bands.external_stasis(mixed))


def test_text_active_per_batch():
    """F1 P3 emission signal: text activity vs band per batch element."""
    bands = ActivityBands(
        config=ActivityBandConfig(min_warmup=4, silence_k=0.5)
    )
    for _ in range(8):
        bands.observe({
            "text": torch.tensor([2.0]),
            "attention": torch.tensor([1.0]),
            "memory": torch.tensor([1.0]),
        })
    act = {
        "text": torch.tensor([0.0, 0.0, 5.0, 5.0]),
        "attention": torch.zeros(4),
        "memory": torch.zeros(4),
    }
    text_active = bands.text_active(act)
    assert text_active.tolist() == [False, False, True, True]


# ---------- F4 armable-by-construction ----------

def test_band_arms_on_any_seed_after_observations():
    """The F4 fix property: K-M9-5 is armable on every init given
    sufficient observations -- not 26/256 like the old sigmoid head.
    256 random decoder seeds, each pushed activity through their
    bands; every band should arm after warmup.
    """
    torch.manual_seed(0)
    n_inits = 16  # 256 is overkill for a unit test; 16 is plenty.
    a_fixed = torch.zeros(1, D)
    armed_counts = 0
    for seed in range(n_inits):
        torch.manual_seed(seed)
        reg = _build_registry()
        bands = ActivityBands(config=ActivityBandConfig(min_warmup=4))
        # Push 10 cycles of activity into the bands.
        for _ in range(10):
            bands.observe(reg.activity(a_fixed))
        if bands.k_m9_5_armed():
            armed_counts += 1
    assert armed_counts == n_inits, (
        f"only {armed_counts}/{n_inits} inits armed -- F4 fix not delivering"
    )


def main() -> int:
    tests = [
        test_text_activity_shape_and_positive,
        test_attention_activity_shape_and_positive,
        test_memory_activity_shape_and_positive,
        test_decoder_activity_varies_with_action,
        test_registry_activity_returns_three_modalities,
        test_bands_disarmed_before_warmup,
        test_external_stasis_all_false_before_warmup,
        test_bands_armed_after_warmup,
        test_external_stasis_fires_when_all_silent,
        test_external_stasis_off_when_any_active,
        test_text_active_per_batch,
        test_band_arms_on_any_seed_after_observations,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
