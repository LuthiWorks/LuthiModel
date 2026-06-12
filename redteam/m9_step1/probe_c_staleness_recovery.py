"""PROBE C -- Gate 4 / K-M9-7: the staleness machinery's spike-
recovery is "verified" by an instrument that measures nothing, and
its re-evaluation is near-inert for exactly the nodes that matter.

Gate 4 (spec §7): "Staleness machinery verified, including a forced
high-surprise spike test (plan §4.v): tree recovers usable cached
info within the budgeted recovery window." Plan §4.v is explicit:
"instrument actual recovery latency so the premise ... is verified,
not assumed."

  C1 (the verification instrument is circular). The only recovery-
     latency signal the machinery emits is
     `_spike_recovery_latencies`, recorded in `observe_drift` when
     `spike_cooldown` counts down to 0. But `spike_cooldown` is a
     fixed countdown set to `recovery_cycles` in `handle_spike`, and
     `observe_drift` decrements it once per cycle unconditionally. So
     the recorded "recovery latency" is ALWAYS exactly
     `recovery_cycles`, regardless of whether a single Q value was
     ever restored. We prove it by running the documented recovery
     loop with re-eval budget = 0 (zero nodes ever refreshed): every
     node's Q stays 0 (catatonic tree), yet the instrument reports
     the same recovery latency it reports when the tree fully
     recovers. The gate's "verified, not assumed" is assumed.

  C2 (re-evaluation is inert for high-visit nodes). Under non-spike
     drift (the common case), `reevaluate` updates a node's cached Q
     toward the fresh value with `alpha = 1/(1+N)`. For a well-
     visited node (high N) -- which is exactly the high-priority
     node the same method selects first (priority = visits *
     staleness) -- alpha is tiny, so a stale/wrong cached Q barely
     moves toward truth. The machinery preferentially re-evaluates
     the nodes it is then almost powerless to correct. We prove it:
     a node with N=100 and a wrong cached Q=10 (true value 0)
     re-evaluates to Q~=9.9 -- a 1% correction the cycle it is
     "refreshed."
"""

from __future__ import annotations

from luthi.v2.m9.staleness import StalenessManager, StalenessConfig

from ._common import Verdict, report


class StubNode:
    def __init__(self, N, Q, theta_stamp):
        self.N = N
        self.Q = Q
        self.theta_stamp = theta_stamp


class StubMCTS:
    """Minimal surface the StalenessManager reads: iter_nodes(),
    sim_counter, root."""

    def __init__(self, nodes, sim_counter):
        self._nodes = nodes
        self.sim_counter = sim_counter
        self.root = nodes[0] if nodes else None

    def iter_nodes(self):
        return iter(self._nodes)


def run() -> list[Verdict]:
    # ---- C1: budget=0 recovery still "succeeds" by the instrument. ----
    cfg = StalenessConfig(recovery_cycles=5)
    mgr = StalenessManager(cfg)

    # 10 healthy, well-visited nodes with good cached values, all stale.
    nodes = [StubNode(N=50, Q=5.0, theta_stamp=0) for _ in range(10)]
    mcts = StubMCTS(nodes, sim_counter=100)

    # Warm the drift band with calm values so a spike is detectable.
    for _ in range(10):
        mgr.observe_drift(0.1)

    # One high-surprise spike cycle, per the documented usage pattern.
    mgr.observe_drift(10.0)
    assert mgr.spike(), "spike not detected; probe setup invalid"
    mgr.handle_spike(mcts)  # zeroes every node's Q
    q_after_spike = [n.Q for n in nodes]

    # Recovery loop with budget = 0: the machinery is given NO capacity
    # to refresh anything.
    for _ in range(cfg.recovery_cycles):
        mgr.observe_drift(0.1)
        mgr.reevaluate(mcts, eval_fn=lambda node: 5.0, budget=0)
    snap = mgr.snapshot()
    recorded_latencies = snap["spike_recovery_latencies"]
    q_after_recovery = [n.Q for n in nodes]

    v1 = Verdict(
        "C1: spike 'recovery' is declared (latency recorded = "
        "recovery_cycles) with re-eval budget 0 -- every node's Q is "
        "still 0, the tree recovered nothing, yet the verification "
        "instrument reports success",
        recorded_latencies == [cfg.recovery_cycles]
        and all(q == 0.0 for q in q_after_recovery),
        f"Q after spike = {q_after_spike[0]} (all zeroed); Q after "
        f"{cfg.recovery_cycles} recovery cycles at budget 0 = "
        f"{q_after_recovery[0]} (all still 0); recorded recovery latency "
        f"= {recorded_latencies} cycles -- a hardcoded countdown, not a "
        "measurement of recovery",
    )

    # ---- C2: re-eval barely corrects a high-visit stale node. ----
    mgr2 = StalenessManager(StalenessConfig())
    # One badly-stale, heavily-visited node: cached Q=10 but truth is 0.
    bad = StubNode(N=100, Q=10.0, theta_stamp=0)
    mcts2 = StubMCTS([bad], sim_counter=50)  # stale: 50 - 0 = 50
    result = mgr2.reevaluate(mcts2, eval_fn=lambda node: 0.0, budget=1)
    corrected_q = bad.Q
    correction_fraction = (10.0 - corrected_q) / 10.0

    v2 = Verdict(
        "C2: re-evaluating a high-visit (N=100) stale node with a wrong "
        "cached Q corrects it by only ~1% in the cycle it is refreshed "
        "(alpha = 1/(1+N)); the machinery prioritizes these nodes then "
        "is almost powerless to fix them",
        corrected_q > 9.5,
        f"re-eval moved Q from 10.0 toward true 0.0 only to "
        f"{corrected_q:.4f} (correction {correction_fraction*100:.2f}%); "
        f"reevaluate reported {result['reevaluated']} node refreshed with "
        f"median deviation {result['median_deviation']:.4f}",
    )

    return report("PROBE C: staleness spike-recovery verification + re-eval inertia", [v1, v2])


if __name__ == "__main__":
    vs = run()
    confirmed_count = sum(1 for v in vs if v.confirmed)
    if confirmed_count == 0:
        print(f"\nAll {len(vs)} attacks REFUTED -- the F3 fix landed.")
    else:
        print(f"\n{confirmed_count}/{len(vs)} attacks still confirmed.")
