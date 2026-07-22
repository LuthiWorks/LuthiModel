"""Regression guard — F4 fix: K-M9-5 is armed by construction.

Inverted version of `redteam/m9_step1/probe_d_darkroom_disarmed.py`.
The probe checks the BUG is present (random init disarms the kill);
this test checks the FIX is in place (the §A.1 activity-band path
arms the kill regardless of decoder init weights).

Run from project root:
    python -m luthi.v2.m9.test_darkroom_armed

Per 4.8's 2026-06-11 gate-repairs spec:
- Gate 2 is re-defined as an *arming + firing* check, not "K-M9-5
  never fired" (which a disarmed kill satisfies vacuously). The
  spec asks: armed throughout AND fires under forced catatonia.
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
from luthi.v2.m9.kills import KillRegistry, KillState


D = 16
V = 32


def _decoder_registry(seed: int) -> DecoderRegistry:
    torch.manual_seed(seed)
    return DecoderRegistry(
        text=TextDecoder(nn.Linear(D, V), D, V),
        attention=AttentionDecoder(d_model=D, n_modalities=3),
        memory=MemoryDecoder(d_model=D),
    )


def _activity_bands(decoders: DecoderRegistry, seed: int) -> ActivityBands:
    bands = ActivityBands(config=ActivityBandConfig(min_warmup=4, silence_k=0.5))
    torch.manual_seed(seed + 1000)
    with torch.no_grad():
        for _ in range(8):
            s = torch.randn(1, D)
            bands.observe(decoders.activity(s))
    return bands


# ---------------- Inverted probe D assertions ----------------

def test_kill_armable_on_every_seed():
    """D1 inverted: across many random decoder inits, the §A.1 path
    arms the kill on ALL of them (not 26/256 like the legacy
    sigmoid-intensity-head path).
    """
    trials = 64  # 256 in the probe; 64 is plenty for a unit test
    armable = 0
    for s in range(trials):
        decoders = _decoder_registry(s)
        bands = _activity_bands(decoders, s)
        if bands.k_m9_5_armed():
            armable += 1
    assert armable == trials, (
        f"§A.1 path must arm on every init: armed {armable}/{trials}"
    )


def test_forced_catatonia_fires_the_kill():
    """D2 inverted: under the F4 path, sustained catatonia (forced
    internal AND external silent under the band) fires K-M9-5 within
    the configured sustained-cycle window.
    """
    seed = 0
    decoders = _decoder_registry(seed)
    bands = _activity_bands(decoders, seed)

    # Forced catatonic activity well below band threshold.
    catatonic_activity = {
        "text": torch.zeros(1),
        "attention": torch.zeros(1),
        "memory": torch.zeros(1),
    }
    external_silent = bool(bands.external_stasis(catatonic_activity)[0].item())
    is_armed = bands.k_m9_5_armed()
    assert is_armed
    assert external_silent, (
        "warmed bands must classify zero activity as silent (F4 input)"
    )

    registry = KillRegistry()  # default darkroom_sustained_cycles = 30
    sustained_window = registry.darkroom_sustained_cycles
    for _ in range(sustained_window + 5):
        registry.observe_darkroom_v2(
            internal_silent=True,
            external_silent=external_silent,
            is_armed=is_armed,
        )
    assert registry.states()["K-M9-5-darkroom"] == KillState.FIRED


def test_armed_state_log_records_per_cycle():
    """The §A.1 armed-state log is the F4 second-repair: per-cycle
    boolean visibility into whether the kill is CAPABLE of firing.
    """
    registry = KillRegistry()
    # 3 disarmed cycles, 5 armed cycles.
    for _ in range(3):
        registry.observe_darkroom_v2(
            internal_silent=False, external_silent=False, is_armed=False,
        )
    for _ in range(5):
        registry.observe_darkroom_v2(
            internal_silent=False, external_silent=False, is_armed=True,
        )
    assert registry.darkroom_armed_now() is True
    frac = registry.darkroom_armed_fraction()
    assert abs(frac - 5 / 8) < 1e-6


def test_disarmed_sustained_escalation():
    """A safety backstop that's been disarmed for the sustained window
    is itself a defect flag (4.8's escalation rule).
    """
    registry = KillRegistry()  # darkroom_disarmed_window = darkroom_sustained_cycles = 30
    for _ in range(registry.darkroom_disarmed_window):
        registry.observe_darkroom_v2(
            internal_silent=False, external_silent=False, is_armed=False,
        )
    assert registry.k_m9_5_disarmed_sustained()

    # Recovery: an armed cycle resets the counter.
    registry.observe_darkroom_v2(
        internal_silent=False, external_silent=False, is_armed=True,
    )
    assert not registry.k_m9_5_disarmed_sustained()


def test_disarmed_cycle_leaves_state_unchanged():
    """A disarmed cycle is neither HEALTHY nor FIRED -- the kill is
    uninformed about whether stasis is occurring. State is preserved
    from the previous armed cycle.
    """
    registry = KillRegistry()
    # First put state into FLAGGED via armed stasis.
    registry.observe_darkroom_v2(
        internal_silent=True, external_silent=True, is_armed=True,
    )
    flagged_state = registry.states()["K-M9-5-darkroom"]
    assert flagged_state == KillState.FLAGGED

    # Disarmed cycle should NOT change the kill state.
    registry.observe_darkroom_v2(
        internal_silent=True, external_silent=True, is_armed=False,
    )
    assert registry.states()["K-M9-5-darkroom"] == flagged_state


def test_legacy_observe_darkroom_still_works():
    """Backward compat: the legacy observe_darkroom signature
    continues to drive the kill state machine. (The legacy path uses
    the broken sigmoid-intensity input, but the state-machine code
    path is shared with v2.)
    """
    registry = KillRegistry(darkroom_sustained_cycles=3)
    for _ in range(3):
        registry.observe_darkroom(
            internal_change_magnitude=0.0,
            external_stasis=True,
        )
    assert registry.states()["K-M9-5-darkroom"] == KillState.FIRED


def main() -> int:
    tests = [
        test_kill_armable_on_every_seed,
        test_forced_catatonia_fires_the_kill,
        test_armed_state_log_records_per_cycle,
        test_disarmed_sustained_escalation,
        test_disarmed_cycle_leaves_state_unchanged,
        test_legacy_observe_darkroom_still_works,
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
