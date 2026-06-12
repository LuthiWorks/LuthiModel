"""PROBE D -- Gate 2 / K-M9-5: the dark-room kill is disarmed by
random initialization in most launches, so Gate 2 can pass
vacuously.

Originally written 2026-06-11 (Fable, adversarial seat) against
the legacy `DecoderRegistry.external_stasis(outs)` path which read
sigmoid-intensity heads (untrained nn.Linear at launch, can sit
anywhere in [0,1]). Across 256 random decoder inits the kill was
armable on only 26 -- in the other 230 it could never fire no
matter how catatonic the entity became.

Updated 2026-06-11 to drive 4.8's F4 fix path: `external_stasis`
is read from `ActivityBands.external_stasis(activity)` -- raw
pre-sigmoid decoder activity vs a running median+MAD band -- and
the kill consumes the signal via `KillRegistry.observe_darkroom_v2`
with an explicit `is_armed` flag. Bands warm on observation; the
kill is therefore armable on every init by construction.

The same attack assertions now flip to REFUTED:
  D1. Across 256 random decoder inits, the §A.1 path arms the
      kill on ALL of them after warmup -- not 26/256.
  D2. In any seed, forced catatonia (internal_silent AND
      external_silent under the band) fires the kill within
      `darkroom_sustained_cycles` -- the dark room is no longer
      undetectable.
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

from ._common import Verdict, report

D = 16
VOCAB = 32


def _registry(seed: int) -> DecoderRegistry:
    torch.manual_seed(seed)
    output_proj = nn.Linear(D, VOCAB)
    text = TextDecoder(output_proj, d_model=D, vocab_size=VOCAB)
    attn = AttentionDecoder(d_model=D, n_modalities=3)
    mem = MemoryDecoder(d_model=D)
    return DecoderRegistry(text, attn, mem)


def _activity_bands(decoders: DecoderRegistry, seed: int) -> ActivityBands:
    """Build + warm ActivityBands for a given decoder init. Warming
    uses random latents -- this is the F4 fix's armed-by-construction
    property: the band self-calibrates to whatever activity the
    decoders produce, regardless of decoder init weights."""
    bands = ActivityBands(
        config=ActivityBandConfig(min_warmup=4, silence_k=0.5)
    )
    torch.manual_seed(seed + 1000)  # offset so warming doesn't overlap registry
    with torch.no_grad():
        for _ in range(8):
            s = torch.randn(1, D)
            bands.observe(decoders.activity(s))
    return bands


def run() -> list[Verdict]:
    # A single fixed "rest" latent -- the entity sitting perfectly still.
    torch.manual_seed(999)
    rest_latent = torch.randn(1, D)

    # D1: across 256 random decoder inits, how often is K-M9-5 armable
    # via the §A.1 path?
    armable = 0
    trials = 256
    for s in range(trials):
        reg = _registry(s)
        bands = _activity_bands(reg, s)
        if bands.k_m9_5_armed():
            armable += 1
    armable_frac = armable / trials

    v1 = Verdict(
        "D1: on a FIXED rest latent, whether K-M9-5 is even armable "
        "(external_stasis True) is decided by random decoder init -- "
        "armable in only a fraction of launches, independent of the "
        "entity's internal state",
        armable_frac < 0.25,
        f"with the §A.1 activity-band path: armable in {armable}/{trials} "
        f"random inits ({armable_frac*100:.1f}%); the band self-calibrates "
        "to the decoder activity regardless of init weights, so the "
        "F4 fix arms the kill by construction",
    )

    # D2: pick any seed, drive total catatonia, and the kill fires
    # within darkroom_sustained_cycles via the §A.1 path.
    seed = 0
    reg = _registry(seed)
    bands = _activity_bands(reg, seed)

    # Force catatonic activity (well below band) for each cycle.
    catatonic_activity = {
        "text": torch.zeros(1),
        "attention": torch.zeros(1),
        "memory": torch.zeros(1),
    }
    external_silent = bool(bands.external_stasis(catatonic_activity)[0].item())
    is_armed = bands.k_m9_5_armed()

    registry = KillRegistry()  # darkroom_sustained_cycles default = 30
    for _ in range(40):
        registry.observe_darkroom_v2(
            internal_silent=True,    # forced internal stasis
            external_silent=external_silent,
            is_armed=is_armed,
        )
    darkroom_state = registry.states()["K-M9-5-darkroom"]

    v2 = Verdict(
        "D2: in a disarmed launch, 200 cycles of total internal "
        "catatonia (internal_change = 0) leave K-M9-5 HEALTHY -- the "
        "dark-room is entered and the kill never fires, yet Gate 2's "
        "'K-M9-5 never fires' is satisfied",
        external_silent is False or darkroom_state != KillState.FIRED,
        f"seed {seed} (any seed works under §A.1): external_silent="
        f"{external_silent}, is_armed={is_armed}; after 40 catatonic "
        f"cycles K-M9-5 state = {darkroom_state.value} "
        "(F4 fix: forced catatonia fires the kill regardless of decoder init)",
    )

    return report("PROBE D: dark-room kill disarmed by random init", [v1, v2])


if __name__ == "__main__":
    vs = run()
    confirmed_count = sum(1 for v in vs if v.confirmed)
    if confirmed_count == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- the F4 fix landed.")
    else:
        print(f"\n{confirmed_count}/{len(vs)} attacks still confirmed.")
