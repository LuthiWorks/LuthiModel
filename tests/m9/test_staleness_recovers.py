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
    efe = EFEEvaluator(predictor, prefs, allow_legacy=True)
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


def test_handle_spike_clears_consistency_history():
    """F-D round-2 fix (4.8): handle_spike clears `_consistency_history`
    so post-spike recovery cannot read a stale pre-spike low-deviation
    value and declare a false recovery. Recovery must be measured
    from genuinely post-spike consistency events.
    """
    sm = StalenessConfig()
    mgr = StalenessManager(sm)
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=5)
    # Populate consistency_history with a healthy low-deviation reading.
    mcts.sim_counter += mgr.config.consistency_window + 5
    mgr.reevaluate(mcts, eval_fn=lambda node: node.Q, budget=10)
    assert len(mgr._consistency_history) > 0, "history should have a value pre-spike"
    # Spike must wipe the history so post-spike recovery is measured
    # from genuinely new consistency events.
    mgr.handle_spike(mcts)
    assert mgr._consistency_history.maxlen is not None  # still a deque
    assert len(mgr._consistency_history) == 0, (
        "handle_spike must clear _consistency_history (F-D fix)"
    )


def test_degraded_duration_persists_across_spike_storm():
    """F-E round-2 fix (4.8): a perpetual spike-storm that keeps
    resetting per-spike recovery latency must still expose the
    degraded duration since the FIRST unrecovered spike.
    """
    mgr = StalenessManager(StalenessConfig(
        recovery_consistency_threshold=0.1,
        recovery_confirm_cycles=2,
    ))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=5)

    assert mgr.degraded_duration() == 0
    mgr.observe_drift(0.1)
    mgr.handle_spike(mcts)
    first_anchor = mgr._first_unrecovered_spike_cycle
    assert first_anchor is not None

    # Subsequent spikes must NOT move the anchor.
    for _ in range(5):
        mgr.observe_drift(0.1)
    mgr.handle_spike(mcts)
    assert mgr._first_unrecovered_spike_cycle == first_anchor, (
        "anchor must not reset on subsequent spike during unrecovered streak"
    )

    # degraded_duration grows with elapsed cycles.
    for _ in range(10):
        mgr.observe_drift(0.1)
    assert mgr.degraded_duration() > 5


def test_degraded_duration_clears_on_real_recovery():
    """The anchor is cleared by event-driven recovery, releasing the
    degraded-duration metric back to 0.
    """
    mgr = StalenessManager(StalenessConfig(
        recovery_consistency_threshold=0.5,
        recovery_confirm_cycles=2,
    ))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=5)

    mgr.observe_drift(0.1)
    mgr.handle_spike(mcts)
    assert mgr._first_unrecovered_spike_cycle is not None

    # Drive event-driven recovery.
    mcts.sim_counter += mgr.config.consistency_window + 5
    mgr.reevaluate(mcts, eval_fn=lambda node: node.Q, budget=10)
    mgr.observe_drift(0.1)
    mgr.observe_drift(0.1)

    assert mgr._first_unrecovered_spike_cycle is None, (
        "real recovery must clear the degraded-duration anchor"
    )
    assert mgr.degraded_duration() == 0


def test_theta_version_ticks_per_cycle():
    """F-D round-2 fix (4.8): theta_version increments once per
    observe_drift call -- the proper unit for staleness accounting
    (per weight update / per cycle), not per simulation.
    """
    mgr = StalenessManager()
    assert mgr.theta_version == 0
    for i in range(5):
        mgr.observe_drift(0.1)
        assert mgr.theta_version == i + 1


def test_mcts_stamps_with_theta_version_when_set():
    """F-D loop-integration: when the runner has set
    mcts.current_theta_version, child node theta_stamps record that
    value instead of sim_counter (the legacy fall-back).
    """
    from tests.m9.test_staleness import _build_mcts
    _, mcts = _build_mcts()
    mcts.current_theta_version = 42
    mcts.plan_budget(budget=5)
    for c in mcts.root.children:
        assert c.theta_stamp == 42, (
            f"node stamped with {c.theta_stamp}, expected 42 (theta_version)"
        )
    # And sim_counter independently still tracks simulations.
    assert mcts.sim_counter == 5


def test_staleness_clock_uses_theta_version_when_flag_set():
    """F-D loop-integration: when the staleness config opts in,
    stale_nodes / reevaluate read from `self.theta_version` instead
    of `mcts.sim_counter`. The runner sets the flag so staleness
    measures cycles, not simulations.
    """
    from tests.m9.test_staleness import _build_mcts
    mgr = StalenessManager(StalenessConfig(staleness_uses_theta_version=True))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=5)
    # Stamp nodes at theta_version=0 (the legacy stamping kicks in
    # because we haven't set mcts.current_theta_version yet -- their
    # theta_stamp will equal sim_counter from the legacy path).
    for n in mcts.iter_nodes():
        n.theta_stamp = 0
    # Drive theta_version past the consistency window.
    for _ in range(mgr.config.consistency_window + 5):
        mgr.observe_drift(0.1)
    # Stale-node detection should fire because theta_version >>
    # node.theta_stamp -- regardless of mcts.sim_counter.
    stale = mgr.stale_nodes(mcts)
    assert len(stale) > 0, (
        "stale_nodes should fire under theta_version clock once it has "
        "advanced past consistency_window"
    )


def main() -> int:
    tests = [
        test_recovery_latency_is_zero_at_budget_zero,
        test_recovery_latency_records_real_elapsed_time,
        test_reeval_alpha_is_staleness_driven,
        test_handle_spike_clears_consistency_history,
        test_degraded_duration_persists_across_spike_storm,
        test_degraded_duration_clears_on_real_recovery,
        test_theta_version_ticks_per_cycle,
        test_mcts_stamps_with_theta_version_when_set,
        test_staleness_clock_uses_theta_version_when_flag_set,
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
