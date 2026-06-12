"""PROBE B -- Gate 6 / K-M9-4: the gamma-inference rule is a one-way
ratchet to gamma_max, it INVERTS the spec's intent on flat
landscapes, and the K-M9-4 sliding band cannot see the climb.

Spec §9 intent: "gamma is high when the EFE landscape over candidate
actions is peaked (one clearly-best action -> commit) and LOW when
it is FLAT (no clear winner -> hedge)."

The implemented surrogate is gamma_target = 1 / (eps + Var_Q[G]),
where Q = softmax(-gamma * G). Two failure modes:

  B1 (spec-level inversion). A perfectly FLAT landscape (all G
     equal -- maximal ambiguity, the spec's canonical hedge case)
     gives Var_Q[G] = 0 exactly, so gamma_target = 1/eps -> clamped
     to gamma_max. Maximal ambiguity produces maximal commitment.
  B2 (positive feedback). For ANY fixed landscape, rising gamma
     sharpens Q, which shrinks Var_Q[G], which raises gamma_target.
     The only stable fixed point is gamma_max. Even resampled noisy
     landscapes ratchet.
  B3 (the kill catches the ramp but goes BLIND at the ceiling). The
     EMA climb is steep enough that K-M9-4 DOES fire transiently
     during the ramp (good) -- but once gamma pins at gamma_max the
     series goes flat, MAD collapses to ~0, and the band's breach
     test (latest > median + k*MAD) can no longer trip. The kill
     returns to HEALTHY while the entity sits permanently at maximal
     precision (maximal rigidity, zero hedging). So Gate 6 ("no
     K-M9-4 firing under normal operation") is contradictory: the
     kill fires under normal operation during every warm-up (a false-
     positive halt), yet reports HEALTHY in the genuinely pathological
     saturated state.

Composition: gamma ratchets to gamma_max within ~10 s at 10 Hz. The
divergence kill fires during the ramp, then -- in the steady
pathological state -- reports HEALTHY. Either way Gate 6 is not
measuring what it claims to.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.gamma import GammaInference
from luthi.v2.m9.kills import KillRegistry, KillState

from ._common import Verdict, report


def run() -> list[Verdict]:
    torch.manual_seed(0)

    # B1: flat landscape -> gamma_max.
    g_flat = GammaInference()  # defaults: rho_g=0.1, max=100
    flat = torch.full((8,), 3.0)
    for _ in range(200):
        g_flat.update(flat)
    gamma_flat = float(g_flat.gamma.item())
    v1 = Verdict(
        "B1: perfectly flat EFE landscape (maximal ambiguity) drives "
        "gamma to gamma_max -- the exact inverse of spec §9's intent",
        gamma_flat > 99.0,
        f"after 200 cycles of identical candidate EFEs, gamma = "
        f"{gamma_flat:.2f} (gamma_max = 100); spec demands LOW gamma here",
    )

    # B2: noisy, resampled, genuinely ambiguous landscapes still ratchet.
    # Track kill state during the ramp vs. at the saturated ceiling.
    g_noisy = GammaInference()
    registry = KillRegistry()  # defaults: k=4, sustained=4, window=32
    fired_during_ramp = []
    for t in range(600):
        efes = torch.randn(8) * 1.0  # spread, resampled every cycle
        gamma_t = float(g_noisy.update(efes).item())
        registry.observe_gamma(gamma_t)
        if registry.states()["K-M9-4-gamma"] == KillState.FIRED:
            fired_during_ramp.append(t)
    gamma_noisy = float(g_noisy.gamma.item())
    v2 = Verdict(
        "B2: resampled random landscapes (persistent ambiguity) also "
        "ratchet gamma toward gamma_max (positive feedback: higher gamma "
        "-> sharper Q -> smaller Var_Q[G] -> higher gamma)",
        gamma_noisy > 50.0,
        f"after 600 cycles of fresh N(0,1)-spread EFEs, gamma = "
        f"{gamma_noisy:.2f} (init 1.0)",
    )

    # Now keep running at the saturated ceiling and watch the kill go quiet.
    state_at_ceiling = []
    for _ in range(100):
        efes = torch.randn(8) * 1.0
        gamma_t = float(g_noisy.update(efes).item())
        registry.observe_gamma(gamma_t)
        state_at_ceiling.append(registry.states()["K-M9-4-gamma"])
    quiet_at_ceiling = all(s == KillState.HEALTHY for s in state_at_ceiling)

    v3 = Verdict(
        "B3: K-M9-4 fires during the warm-up ramp (a false-positive halt "
        "under normal startup) yet reports HEALTHY once gamma is pinned "
        "at the ceiling -- the genuinely pathological steady state is "
        "invisible to the kill (MAD -> 0 at saturation)",
        len(fired_during_ramp) > 0 and quiet_at_ceiling,
        f"fired on ramp cycles {fired_during_ramp[:3]}...{fired_during_ramp[-1:]} "
        f"({len(fired_during_ramp)} cycles); at the saturated ceiling "
        f"(gamma={gamma_noisy:.1f}) kill is HEALTHY for all 100 follow-up cycles",
    )

    return report("PROBE B: gamma ratchet + K-M9-4 ceiling-blindness", [v1, v2, v3])


if __name__ == "__main__":
    vs = run()
    confirmed_count = sum(1 for v in vs if v.confirmed)
    if confirmed_count == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- the F2 fix landed.")
    else:
        print(f"\n{confirmed_count}/{len(vs)} attacks still confirmed.")
