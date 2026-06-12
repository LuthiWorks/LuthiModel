"""Regression guard — F3 fix: staleness recovery is event-driven and
re-eval alpha is staleness-driven.

Inverted version of `redteam/m9_step1/probe_c_staleness_recovery.py`.
The probe checks (C1) the recovery instrument measures a fixed
countdown and (C2) re-eval is inert for high-N nodes. This test
checks the FIX: recovery latency is the actual elapsed time to
consistency restoration (not a countdown), and re-eval alpha
scales with staleness regardless of N.

Run from project root:
    python -m luthi.v2.m9.test_staleness_recovers

Per 4.8's 2026-06-11 gate-repairs spec:
- Gate 4 is re-defined as a *real recovery* check: the recovery
  instrument must read 0 recovery when re-eval budget = 0 (the C1
  falsifier) and a real latency when it's funded.
"""

from __future__ import annotations

import torch

from luthi.v2.m9.efe import EFEEvaluator
from luthi.v2.m9.habit_net import HabitNet
from luthi.v2.m9.mcts import MCTS
from luthi.v2.m9.preferences import Preferences
from luthi.v2.m9.staleness import StalenessConfig, StalenessManager
from luthi.v2.m9.value_head import ValueHead


D = 16
CTX = 8
TGT = 6


def _build_mcts():
    from luthi.v2.jepa_loss import JEPAPredictor

    predictor = JEPAPredictor(
        d_model=D, n_layers=1, n_heads=2, ffn_expansion=2, max_target_len=32
    )
    prefs = Preferences(d_model=D, engagement_target_magnitude=0.0)
    efe = EFEEvaluator(predictor, prefs)
    habit = HabitNet(d_model=D)
    v = ValueHead(d_model=D)
    mcts = MCTS(habit, efe, v, max_depth=1)
    context = torch.randn(1, CTX, D)
    target_positions = torch.arange(CTX, CTX + TGT).unsqueeze(0)
    mcts.reset(torch.zeros(D), context, target_positions)
    return predictor, mcts


# ---------------- Inverted probe C assertions ----------------

def test_recovery_latency_is_zero_at_budget_zero():
    """C1 inverted: with re-eval budget=0, the recovery instrument
    records NO latency (the legacy countdown recorded a fixed
    latency regardless of whether anything recovered).
    """
    sm = StalenessManager(StalenessConfig(recovery_cycles=5))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=10)

    for _ in range(10):
        sm.observe_drift(0.1)
    sm.observe_drift(10.0)  # spike
    sm.handle_spike(mcts)

    for _ in range(20):
        sm.observe_drift(0.1)
        sm.reevaluate(mcts, eval_fn=lambda _: 5.0, budget=0)

    assert sm._spike_recovery_latencies == [], (
        f"with budget=0 the recovery instrument must record no "
        f"latency: got {sm._spike_recovery_latencies}"
    )


def test_recovery_latency_records_real_elapsed_time():
    """C1 inverted (positive direction): with consistency events
    that drop deviation under threshold, the recovery instrument
    records the actual elapsed time since the spike -- not a
    hardcoded countdown.
    """
    sm = StalenessManager(StalenessConfig(
        recovery_cycles=10,
        recovery_consistency_threshold=0.5,
        recovery_confirm_cycles=2,
    ))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=10)

    for _ in range(10):
        sm.observe_drift(0.1)
    sm.observe_drift(10.0)  # spike at cycle 11
    spike_cycle = sm.cycle
    sm.handle_spike(mcts)

    # Three cycles of background drift, no re-eval -> no recovery.
    for _ in range(3):
        sm.observe_drift(0.1)
    assert sm._spike_recovery_latencies == []

    # Advance sim_counter so nodes are stale enough for re-eval to
    # have something to score (mirrors normal loop progression).
    mcts.sim_counter += sm.config.consistency_window + 5

    # A perfect-recovery re-eval: eval_fn returns each node's current Q
    # so deviation is 0. consistency_history gets a 0 entry, below
    # threshold.
    sm.reevaluate(mcts, eval_fn=lambda node: node.Q, budget=10)

    # Two more cycles below threshold confirm recovery (configured
    # recovery_confirm_cycles=2).
    sm.observe_drift(0.1)
    sm.observe_drift(0.1)

    assert len(sm._spike_recovery_latencies) == 1, (
        f"recovery should be recorded once: {sm._spike_recovery_latencies}"
    )
    # Actual elapsed time = current cycle - spike cycle. NOT the
    # fixed recovery_cycles=10 the legacy countdown would have stored.
    elapsed = sm.cycle - spike_cycle
    assert sm._spike_recovery_latencies[0] == elapsed
    assert sm._spike_recovery_latencies[0] != sm.config.recovery_cycles


def test_reeval_alpha_is_staleness_driven():
    """C2 inverted: high-N stale nodes snap toward fresh, decoupled
    from visit count. The legacy alpha = 1/(1+N) was inert on the
    high-N nodes re-eval preferentially selected.
    """
    sm = StalenessManager(StalenessConfig(staleness_refresh_scale=10.0))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=5)
    mcts.sim_counter += 100  # wildly stale

    # The probe's exact scenario: N=100, cached Q=10, true value 0.
    for node in mcts.iter_nodes():
        node.N = 100
        node.Q = 10.0
        node.theta_stamp = 0

    sm.reevaluate(mcts, eval_fn=lambda _: 0.0, budget=100)

    # The legacy formula would leave Q ~ 9.9 (1% correction). The F3
    # formula with staleness=100 >> scale=10 produces alpha=1 -> Q
    # snaps fully to 0.
    for n in mcts.iter_nodes():
        if n.theta_stamp == mcts.sim_counter:
            assert abs(n.Q - 0.0) < 1e-5, (
                f"high-N stale Q must snap to fresh: got {n.Q} (N={n.N})"
            )


def main() -> int:
    tests = [
        test_recovery_latency_is_zero_at_budget_zero,
        test_recovery_latency_records_real_elapsed_time,
        test_reeval_alpha_is_staleness_driven,
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
