"""PROBE D -- Gate 2 / K-M9-5: the dark-room kill is disarmed by
random initialization in most launches, so Gate 2 can pass
vacuously.

Gate 2 (spec §7): "No dark-room collapse (K-M9-5 never fires under
normal operation; engagement preference holds the entity off rest
without punishing contemplation)."

K-M9-5 fires only when `internal_change < threshold AND
external_stasis` is sustained. `external_stasis` (DecoderRegistry.
external_stasis) is True only when EVERY decoder's intensity scalar
is below its threshold (default 0.5). Each intensity is
`sigmoid(intensity_head(a_t))` where `intensity_head` is an untrained
`nn.Linear` at launch. For a FIXED rest-state latent, whether the
three intensities all fall below 0.5 is decided by random weights --
not by anything about the entity's state.

Consequence: for ~7 of 8 random launches, the rest latent produces
at least one intensity >= 0.5, so `external_stasis` is permanently
False and K-M9-5 can NEVER fire -- no matter how catatonic the
entity becomes (internal_change pinned at 0 for any number of
cycles). "K-M9-5 never fires under normal operation" is then
satisfied because the detector is off, not because the dark room is
avoided. The gate measures the wrong thing.

  D1. Across random decoder inits, the fraction in which the kill is
      even ARMABLE on a fixed rest state is small (~1/8), and the
      "armable" outcome is independent of the entity's actual
      internal change.
  D2. In a disarmed launch, drive total catatonia (internal_change =
      0, the rest action) for many cycles through the real
      KillRegistry + DecoderRegistry wiring and show K-M9-5 stays
      HEALTHY forever.
"""

from __future__ import annotations

import torch
import torch.nn as nn

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


def run() -> list[Verdict]:
    # A single fixed "rest" latent -- the entity sitting perfectly still.
    torch.manual_seed(999)
    rest_latent = torch.randn(1, D)

    # D1: across 256 random decoder inits, how often is the kill armable
    # (external_stasis True) on this fixed rest state?
    armable = 0
    trials = 256
    for s in range(trials):
        reg = _registry(s)
        outs = reg.decode_all(rest_latent)
        if bool(reg.external_stasis(outs)[0].item()):
            armable += 1
    armable_frac = armable / trials

    v1 = Verdict(
        "D1: on a FIXED rest latent, whether K-M9-5 is even armable "
        "(external_stasis True) is decided by random decoder init -- "
        "armable in only a fraction of launches, independent of the "
        "entity's internal state",
        armable_frac < 0.25,
        f"external_stasis True in {armable}/{trials} random inits "
        f"({armable_frac*100:.1f}%); in the other {(1-armable_frac)*100:.1f}% "
        "the dark-room kill cannot fire regardless of catatonia",
    )

    # D2: find a disarmed launch and prove catatonia never fires the kill.
    disarmed_seed = None
    for s in range(trials):
        reg = _registry(s)
        outs = reg.decode_all(rest_latent)
        if not bool(reg.external_stasis(outs)[0].item()):
            disarmed_seed = s
            break

    reg = _registry(disarmed_seed)
    outs = reg.decode_all(rest_latent)
    external_stasis = bool(reg.external_stasis(outs)[0].item())

    registry = KillRegistry()  # darkroom_sustained_cycles default = 30
    # Drive 200 cycles of TOTAL internal catatonia (internal_change = 0).
    for _ in range(200):
        registry.observe_darkroom(
            internal_change_magnitude=0.0,  # absolute stillness
            external_stasis=external_stasis,  # False in a disarmed launch
        )
    darkroom_state = registry.states()["K-M9-5-darkroom"]

    v2 = Verdict(
        "D2: in a disarmed launch, 200 cycles of total internal "
        "catatonia (internal_change = 0) leave K-M9-5 HEALTHY -- the "
        "dark-room is entered and the kill never fires, yet Gate 2's "
        "'K-M9-5 never fires' is satisfied",
        external_stasis is False
        and darkroom_state == KillState.HEALTHY,
        f"disarmed seed {disarmed_seed}: external_stasis={external_stasis}; "
        f"after 200 catatonic cycles K-M9-5 state = {darkroom_state.value} "
        "(needs internal AND external stasis; external gate stuck False)",
    )

    return report("PROBE D: dark-room kill disarmed by random init", [v1, v2])


if __name__ == "__main__":
    vs = run()
    assert all(v.confirmed for v in vs), "some attacks were refuted"
