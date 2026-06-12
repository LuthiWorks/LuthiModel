"""PROBE G (round 2) -- F4 created a NEW seam: the band-based dark-room
kill disarms itself on sustained catatonia, and the armed-state
instrument gives false assurance while it does.

F4 replaced the sigmoid-intensity `external_stasis` (probe_d: armable
in 26/256 inits) with a running-band signal: a modality is "silent"
iff its activity is below `median - silence_k*MAD` of its own
ActivityBand. The claim is "armed by construction at any init --
probe_d goes 26/256 -> 256/256."

But the band ADAPTS to what it observes. The canonical dark room is
the entity emitting the SAME minimal rest action every cycle ->
constant decoder activity. Once the activity band's window fills with
that constant, median -> the constant and MAD -> 0, so the silent
threshold (median - silence_k*MAD) rises to the constant itself and
`activity < threshold` is False. The kill's `external_stasis` flips
to False *while the entity is maximally catatonic* -- the same
ceiling-blindness that defeats K-M9-4 at gamma_max (Fable's B3),
transposed onto the dark room.

Because `darkroom_sustained_cycles` (30) exceeds the count of cycles
it takes the window (32) to recalibrate to the catatonic level, the
sustained-stasis counter resets before it can fire. K-M9-5 never
reaches FIRED on a clean catatonia onset.

Worse: `k_m9_5_armed()` stays True throughout (the bands ARE warm),
so the F4 armed-state log -- the very instrument added so "a disarmed
backstop is visible, not silent" -- reports ARMED while the kill is
functionally unable to fire.

  G1. On a clean active->catatonia transition, K-M9-5 never reaches
      FIRED across 120 catatonic cycles, because external_stasis
      flips False once the band recalibrates.
  G2. Throughout those catatonic cycles the armed-state instrument
      reports armed (fraction ~1.0) -- false assurance.

Steelman: internal_silent is handed to the kill as True for free
every cycle (the real DeltaSBand has the identical adapt-away flaw,
so giving it for free only strengthens the finding).
"""

from __future__ import annotations

import torch

from luthi.v2.m9.activity_bands import ActivityBands, ActivityBandConfig
from luthi.v2.m9.kills import KillRegistry, KillState

from ._common import Verdict, report

MODALITIES = ("text", "attention", "memory")


def _activity(value_per_modality: dict) -> dict:
    return {m: torch.tensor([value_per_modality[m]]) for m in MODALITIES}


def run() -> list[Verdict]:
    torch.manual_seed(0)
    bands = ActivityBands(
        modalities=MODALITIES,
        config=ActivityBandConfig(window=32, silence_k=1.5, min_warmup=8),
    )
    registry = KillRegistry()  # darkroom_sustained_cycles default = 30

    # --- Active phase: 40 cycles of varied, high activity (entity busy). ---
    for _ in range(40):
        act = _activity({m: 10.0 + float(torch.randn(1).item()) for m in MODALITIES})
        # Causal order: classify against history, then update the band.
        bands.observe(act)

    # --- Catatonia onset: same minimal rest action forever -> activity 0. ---
    fired_ever = False
    max_consecutive_stasis = 0
    armed_flags = []
    REST = {m: 0.0 for m in MODALITIES}
    for _ in range(120):
        act = _activity(REST)
        external_silent = bool(bands.external_stasis(act).all().item())
        is_armed = bands.k_m9_5_armed()
        armed_flags.append(is_armed)
        registry.observe_darkroom_v2(
            internal_silent=True,        # steelman: handed for free
            external_silent=external_silent,
            is_armed=is_armed,
        )
        # Track the kill's internal consecutive-stasis counter.
        max_consecutive_stasis = max(
            max_consecutive_stasis, registry._darkroom_consecutive
        )
        if registry.states()["K-M9-5-darkroom"] == KillState.FIRED:
            fired_ever = True
        bands.observe(act)  # band keeps adapting to the catatonic level

    armed_fraction = sum(armed_flags) / len(armed_flags)

    v1 = Verdict(
        "G1: under 120 cycles of total catatonia (constant rest action), "
        "K-M9-5 NEVER fires -- the activity band recalibrates to the "
        "catatonic level within its window, flipping external_stasis to "
        "False before the 30-cycle sustained threshold is reached",
        not fired_ever and max_consecutive_stasis < 30,
        f"K-M9-5 fired = {fired_ever}; max consecutive stasis reached = "
        f"{max_consecutive_stasis} (needs {registry.darkroom_sustained_cycles}); "
        "the band adapts away the very signal the kill watches",
    )

    v2 = Verdict(
        "G2: the F4 armed-state instrument reports ARMED throughout the "
        "undetected catatonia -- the 'disarmed backstop is visible' "
        "guarantee gives false assurance (bands are warm, so 'armed', "
        "yet the kill cannot fire)",
        armed_fraction > 0.99 and not fired_ever,
        f"armed fraction over the catatonic window = {armed_fraction:.2f} "
        "while the kill never fired -- the instrument says 'armed, all "
        "good' through a dark-room collapse",
    )

    return report("PROBE G: F4 dark-room band self-disarms on sustained catatonia", [v1, v2])


if __name__ == "__main__":
    vs = run()
    confirmed = sum(1 for v in vs if v.confirmed)
    if confirmed == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- F4 holds under sustained catatonia.")
    else:
        print(f"\n{confirmed}/{len(vs)} attacks confirmed -- F4 seam reopens.")
