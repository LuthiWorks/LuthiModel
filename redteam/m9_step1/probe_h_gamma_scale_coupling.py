"""PROBE H (round 2) -- F2 removed the ratchet but the replacement
metric conflates landscape SCALE with landscape PEAKEDNESS, so gamma
now drifts with absolute EFE magnitude. Combined with P3's UNBOUNDED
connection cost, a legitimately-silent entity drives gamma to the
ceiling and trips K-M9-4 (false-positive halt).

F2's new target is `gamma_target = gamma_scale * std({G(a_k)})` with
gamma_scale=1.0. No gamma in the target -> the ratchet (B1/B2) is
genuinely gone; that part of the fix is sound. But raw std is NOT
scale-invariant, while the spec's notion of precision ("peaked vs
flat") is a SHAPE property. Two consequences:

  H1 (scale, not shape, sets decisiveness). Take one landscape and
     scale it by a constant. Relative peakedness is identical; the
     "clearly-best" structure is unchanged. gamma scales with the
     constant. A small-magnitude but clearly-peaked landscape gets
     LOW gamma (hedges when it should commit); a large-magnitude
     landscape gets high gamma regardless of shape. No single
     gamma_scale knob fixes this -- it shifts every gamma together,
     it cannot make small-peaked commit while large-flat hedges.

  H2 (unbounded P3 x scale-coupling = false-positive halt). P3's
     connection cost is unbounded: `counterpart*(time_since_emission
     +1)*(1-text_active)`. While the entity is silent in company,
     time_since_emission grows every cycle, so the silent-vs-active
     spread in G grows without bound -> std grows -> gamma climbs to
     gamma_max -> K-M9-4 clamp-proximity FIRES. The precision
     mechanism reads "how long have I been silent" as "how decisive
     should I be," and halts a merely-deliberating entity.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.gamma import GammaInference
from luthi.v2.m9.kills import KillRegistry, KillState

from ._common import Verdict, report


def run() -> list[Verdict]:
    torch.manual_seed(0)

    # --- H1: identical shape, scaled, gives proportional gamma. ---
    shape = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # one clear best

    def equilibrium_gamma(landscape: torch.Tensor, cycles: int = 300) -> float:
        g = GammaInference()
        for _ in range(cycles):
            g.update(landscape)
        return float(g.gamma.item())

    gamma_small = equilibrium_gamma(shape * 1.0)
    gamma_large = equilibrium_gamma(shape * 50.0)
    ratio = gamma_large / max(gamma_small, 1e-9)

    v1 = Verdict(
        "H1: gamma tracks absolute EFE SCALE, not landscape shape -- the "
        "same clearly-peaked landscape scaled x50 yields ~x50 the "
        "precision, so decisiveness depends on how expensive the options "
        "are, not on how clearly one wins (no single gamma_scale fixes it)",
        ratio > 20.0,
        f"identical-shape landscape: gamma(x1) = {gamma_small:.3f}, "
        f"gamma(x50) = {gamma_large:.3f}, ratio = {ratio:.1f}x "
        "(relative peakedness is identical between the two)",
    )

    # --- H2: unbounded c_con growth over a silent stretch -> K-M9-4 fires. ---
    # K=8 candidates: 4 "active" (c_con=0), 4 "silent" (c_con=elapsed+1).
    base = torch.randn(8) * 0.1
    silent_mask = torch.tensor([0.0, 0, 0, 0, 1, 1, 1, 1])
    g = GammaInference()
    registry = KillRegistry()
    fired_cycle = None
    gamma_trace = []
    for elapsed in range(600):  # 600 cycles silent in company = 60s at 10Hz
        c_con = silent_mask * (elapsed + 1.0)
        landscape = base + c_con  # G with the growing connection spread
        gamma_t = float(g.update(landscape).item())
        gamma_trace.append(gamma_t)
        registry.observe_gamma(gamma_t)
        if registry.states()["K-M9-4-gamma"] == KillState.FIRED and fired_cycle is None:
            fired_cycle = elapsed

    v2 = Verdict(
        "H2: a silent-in-company entity (time_since_emission growing, "
        "P3 unbounded) drives gamma to the ceiling purely via cost "
        "MAGNITUDE and trips K-M9-4 -- a false-positive halt for "
        "legitimate deliberation, not a genuine precision pathology",
        fired_cycle is not None,
        f"K-M9-4 FIRED at silent-cycle {fired_cycle} "
        f"(gamma {gamma_trace[0]:.2f} -> {gamma_trace[-1]:.2f}); the "
        "relative landscape shape never changed -- only the absolute "
        "connection-cost scale grew",
    )

    return report("PROBE H: F2 gamma conflates scale with peakedness", [v1, v2])


if __name__ == "__main__":
    vs = run()
    confirmed = sum(1 for v in vs if v.confirmed)
    if confirmed == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- F2 gamma is scale-robust.")
    else:
        print(f"\n{confirmed}/{len(vs)} attacks confirmed -- F2 scale-coupling.")
