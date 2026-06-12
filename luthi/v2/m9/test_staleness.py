"""Unit tests for cross-cycle staleness machinery (plan §4).

Run from project root:
    python -m luthi.v2.m9.test_staleness

Tests cover each of the five sub-pieces of plan §4:
  (i)   Recency-decay applied per cycle; drift-modulated rate.
  (ii)  Stale-node identification by theta_stamp; priority order
        (visits x staleness).
  (iii) Re-evaluation snaps stale Q toward fresh values; records
        deviation into the consistency history.
  (iv)  Held-head failover: snapshot refresh schedule; failover
        triggers on sustained breach; recovers on consistency drop.
  (v)   High-surprise spike: detected on a sharp outlier; handler
        enters failover, drops Q, sets recovery cooldown; recovery
        latency recorded.
"""

from __future__ import annotations

import torch

from luthi.v2.jepa_loss import JEPAPredictor
from luthi.v2.m9.efe import EFEEvaluator
from luthi.v2.m9.habit_net import HabitNet
from luthi.v2.m9.mcts import MCTS, MCTSNode
from luthi.v2.m9.preferences import Preferences
from luthi.v2.m9.staleness import (
    DriftBand,
    StalenessConfig,
    StalenessManager,
)
from luthi.v2.m9.value_head import ValueHead


D = 16
B = 1
CTX = 8
TGT = 6


def _build_mcts():
    predictor = JEPAPredictor(
        d_model=D, n_layers=1, n_heads=2, ffn_expansion=2, max_target_len=32
    )
    prefs = Preferences(d_model=D, engagement_target_magnitude=0.0)
    efe = EFEEvaluator(predictor, prefs, allow_legacy=True)
    habit = HabitNet(d_model=D, log_std_init=0.0)
    v = ValueHead(d_model=D)
    mcts = MCTS(habit, efe, v, max_depth=1)
    mcts.reset(
        torch.zeros(D),
        torch.randn(B, CTX, D),
        torch.arange(CTX, CTX + TGT).unsqueeze(0).expand(B, -1),
    )
    return predictor, mcts


# ----------------- DriftBand -----------------

def test_drift_band_median_mad():
    band = DriftBand(window=8)
    for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
        band.push(x)
    assert band.median() == 3.0
    # MAD of {1,2,3,4,5} from median 3 = median(|x-3|) = median(2,1,0,1,2) = 1.0
    assert abs(band.mad() - 1.0) < 1e-5


def test_drift_band_warmup_gate():
    band = DriftBand(window=8)
    assert not band.is_warm(min_samples=4)
    for x in [1.0, 1.0, 1.0]:
        band.push(x)
    assert not band.is_warm(min_samples=4)
    band.push(1.0)
    assert band.is_warm(min_samples=4)


# ----------------- Per-cycle observation -----------------

def test_observe_drift_advances_cycle():
    sm = StalenessManager()
    sm.observe_drift(0.1)
    sm.observe_drift(0.1)
    assert sm.cycle == 2


# ----------------- (i) Recency-decay -----------------

def test_recency_decay_reduces_visits():
    sm = StalenessManager(StalenessConfig(base_recency_decay=0.5))
    # Warm the drift band so effective_decay is well-defined.
    for _ in range(10):
        sm.observe_drift(0.1)
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=20)
    # Pre-decay total visit mass.
    pre = sum(node.N for node in mcts.iter_nodes())
    sm.decay(mcts)
    post = sum(node.N for node in mcts.iter_nodes())
    assert post < pre, f"decay should reduce visits: pre={pre}, post={post}"


def test_decay_rate_modulated_by_drift():
    """Higher recent drift -> faster decay (smaller rho)."""
    sm = StalenessManager(StalenessConfig(base_recency_decay=0.9))
    # Low-drift baseline.
    for _ in range(10):
        sm.observe_drift(0.1)
    base_rho = sm.effective_decay()
    # Big spike.
    sm.observe_drift(5.0)
    spiked_rho = sm.effective_decay()
    assert spiked_rho < base_rho, (
        f"effective decay should drop under high drift: "
        f"baseline={base_rho:.3f}, spiked={spiked_rho:.3f}"
    )


# ----------------- (v) Spike detection + handler -----------------

def test_spike_detection_fires_on_outlier():
    sm = StalenessManager(StalenessConfig(spike_k=3.0))
    for _ in range(20):
        sm.observe_drift(0.1)
    assert not sm.spike(), "no spike on baseline drift"
    sm.observe_drift(100.0)
    assert sm.spike(), "spike should fire on clear outlier"


def test_spike_handler_enters_failover_and_drops_q():
    sm = StalenessManager()
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=15)
    # Give every node a non-zero Q so we can verify the drop.
    for node in mcts.iter_nodes():
        node.Q = 5.0
    sm.handle_spike(mcts)
    assert sm.in_failover()
    assert sm.spike_cooldown == sm.config.recovery_cycles
    for node in mcts.iter_nodes():
        assert node.Q == 0.0, "spike handler should reset Q values"


def test_spike_recovery_latency_recorded():
    """F3 C1: recovery latency is event-driven -- declared when the
    consistency deviation returns under threshold for
    `recovery_confirm_cycles` consecutive cycles, with the recorded
    value being the ACTUAL cycle count since the spike, not a fixed
    countdown.
    """
    sm = StalenessManager(StalenessConfig(
        recovery_cycles=10,
        recovery_consistency_threshold=0.5,
        recovery_confirm_cycles=2,
    ))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=5)

    sm.observe_drift(0.1)  # cycle 1
    sm.handle_spike(mcts)  # last_spike_cycle = 1, in_recovery=True

    # No real recovery if no re-eval happens (the C1 fix).
    sm.observe_drift(0.1)  # cycle 2 -- no consistency event yet
    sm.observe_drift(0.1)  # cycle 3 -- no consistency event yet
    assert sm._spike_recovery_latencies == [], (
        "with no consistency event, no latency should be recorded"
    )

    # Advance sim_counter so nodes count as stale and re-eval has
    # something to score (the same way the real loop progresses).
    mcts.sim_counter += sm.config.consistency_window + 5

    # Now drive consistency_history under threshold via reevaluate
    # with a fresh-matching eval_fn.
    def eval_fresh(node):
        return node.Q  # zero deviation
    sm.reevaluate(mcts, eval_fn=eval_fresh, budget=10)

    # Two more observe_drift cycles below threshold confirm recovery.
    sm.observe_drift(0.1)  # cycle 4 -- counter 1
    sm.observe_drift(0.1)  # cycle 5 -- counter 2 -> recovery recorded
    assert len(sm._spike_recovery_latencies) == 1, (
        f"recovery should be recorded once: {sm._spike_recovery_latencies}"
    )
    # Latency = cycle - last_spike_cycle = 5 - 1 = 4 (real elapsed time,
    # not a fixed countdown).
    assert sm._spike_recovery_latencies[0] == 4


# ----------------- (ii) + (iii) Stale-node identification + re-eval -----------------

def test_stale_nodes_priority_order():
    """Re-eval priority = visits * staleness; highest first."""
    sm = StalenessManager()
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=15)
    # Advance sim_counter so existing nodes are now stale.
    mcts.sim_counter += sm.config.consistency_window + 10
    # Hand-edit two nodes' visit counts to make priority unambiguous.
    children = mcts.root.children
    assert len(children) >= 2
    children[0].N = 100  # high visits
    children[1].N = 1    # low visits
    children[0].theta_stamp = 0
    children[1].theta_stamp = 0
    stale = sm.stale_nodes(mcts)
    assert children[0] in stale
    assert children[1] in stale
    idx0 = stale.index(children[0])
    idx1 = stale.index(children[1])
    assert idx0 < idx1, (
        f"high-visit stale node should come first: idx0={idx0}, idx1={idx1}"
    )


def test_reevaluate_snaps_high_staleness_regardless_of_N():
    """F3 C2: re-eval alpha is staleness-driven, not N-driven.

    The legacy form `alpha = 1/(1+N)` made re-eval inert for the
    high-N nodes it preferentially selected (probe_c C2). The F3
    fix decouples staleness correction from N: any node with
    staleness >= staleness_refresh_scale snaps fully to fresh,
    regardless of N.
    """
    sm = StalenessManager(StalenessConfig(staleness_refresh_scale=10.0))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=10)
    mcts.sim_counter += 100  # everything wildly stale (>> scale)

    # Mix of low-N and high-N; both should snap because staleness is
    # huge.
    for i, node in enumerate(mcts.iter_nodes()):
        node.Q = 999.0
        node.theta_stamp = 0
        node.N = 0 if i % 2 == 0 else 99

    def eval_fn(node):
        return 1.0

    sm.reevaluate(mcts, eval_fn=eval_fn, budget=100)
    for n in mcts.iter_nodes():
        if n.theta_stamp == mcts.sim_counter:
            assert abs(n.Q - 1.0) < 1e-5, (
                f"high-staleness Q must snap regardless of N: got {n.Q} (N={n.N})"
            )


def test_reevaluate_alpha_respects_low_staleness_floor():
    """F3 C2: low-staleness nodes still receive at least
    `alpha_refresh_min` correction -- mildly stale Q is gently
    nudged, not snapped.
    """
    sm = StalenessManager(StalenessConfig(
        staleness_refresh_scale=100.0,  # large scale -> low staleness gives small alpha
        alpha_refresh_min=0.1,
    ))
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=10)
    # Advance the sim_counter so existing nodes are stale enough to be
    # selected for re-eval, but only mildly stale (1 unit) -- under the
    # new alpha formula this should produce alpha = alpha_refresh_min.
    target_staleness = sm.config.consistency_window + 1  # just past stale cutoff
    initial_sim = list(mcts.iter_nodes())[0].theta_stamp
    mcts.sim_counter = initial_sim + target_staleness

    for node in mcts.iter_nodes():
        node.Q = 10.0
        # theta_stamp left at initial_sim => staleness = target_staleness
        node.N = 50  # high N so the legacy formula would barely move

    fresh = 0.0

    def eval_fn(node):
        return fresh

    sm.reevaluate(mcts, eval_fn=lambda _: fresh, budget=100)
    # alpha ~ target_staleness / 100 ~ 0.17 (depending on consistency_window),
    # clamped to >= alpha_refresh_min = 0.1. Either way the move is
    # significantly larger than the legacy 1/(1+50) = ~0.02 would have
    # produced.
    for n in mcts.iter_nodes():
        if n.theta_stamp == mcts.sim_counter:
            # Legacy: 10 * 0.98 + 0 * 0.02 = 9.8 (barely moves).
            # F3 with alpha >= 0.1: 10 * 0.9 + 0 * 0.1 = 9.0 or smaller.
            assert n.Q <= 9.0, (
                f"low-staleness high-N Q must still get >= alpha_refresh_min "
                f"correction (not stuck near 10.0 like the legacy formula): got {n.Q}"
            )


def test_reevaluate_pushes_consistency_history():
    sm = StalenessManager()
    _, mcts = _build_mcts()
    mcts.plan_budget(budget=10)
    mcts.sim_counter += 100
    for node in mcts.iter_nodes():
        node.theta_stamp = 0

    def eval_fn(node):
        return 0.5

    sm.reevaluate(mcts, eval_fn=eval_fn, budget=5)
    assert len(sm._consistency_history) == 1


# ----------------- (iv) Held-head failover -----------------

def test_held_head_snapshot_created():
    sm = StalenessManager()
    predictor, _ = _build_mcts()
    assert sm.held_predictor is None
    sm.maybe_refresh_held_head(predictor)
    assert sm.held_predictor is not None
    # Held should not require grad.
    for p in sm.held_predictor.parameters():
        assert not p.requires_grad


def test_held_head_refresh_respects_cadence():
    sm = StalenessManager(StalenessConfig(held_head_refresh_every=5))
    predictor, _ = _build_mcts()
    # First call -> snapshot.
    sm.maybe_refresh_held_head(predictor)
    first_id = id(sm.held_predictor)
    # Advance fewer cycles than cadence.
    for _ in range(3):
        sm.observe_drift(0.1)
    sm.maybe_refresh_held_head(predictor)
    assert id(sm.held_predictor) == first_id, (
        "should not refresh before cadence"
    )
    # Now hit the cadence.
    for _ in range(3):
        sm.observe_drift(0.1)
    sm.maybe_refresh_held_head(predictor)
    assert id(sm.held_predictor) != first_id, (
        "should refresh once cadence elapses"
    )


def test_failover_triggers_on_sustained_breach():
    cfg = StalenessConfig(
        failover_consistency_threshold=0.5,
        failover_breach_cycles=3,
        consistency_window=8,
    )
    sm = StalenessManager(cfg)
    # Push consistency-history values above threshold for the breach window.
    for _ in range(3):
        sm._consistency_history.append(1.0)
        sm.update_failover_state()
    assert sm.in_failover(), (
        "should enter failover after sustained breach"
    )


def test_failover_recovers_when_consistency_drops():
    cfg = StalenessConfig(
        failover_consistency_threshold=0.5,
        failover_breach_cycles=2,
        consistency_window=8,
    )
    sm = StalenessManager(cfg)
    # Enter failover.
    for _ in range(2):
        sm._consistency_history.append(1.0)
        sm.update_failover_state()
    assert sm.in_failover()
    # Consistency drops below threshold; spike_cooldown == 0 so recovery
    # can proceed.
    sm._consistency_history.append(0.1)
    sm.update_failover_state()
    assert not sm.in_failover()


# ----------------- Snapshot -----------------

def test_snapshot_contains_expected_keys():
    sm = StalenessManager()
    for _ in range(8):
        sm.observe_drift(0.1)
    snap = sm.snapshot()
    for key in (
        "cycle", "drift_median", "drift_mad", "drift_latest",
        "effective_decay", "spike_active", "spike_cooldown",
        "in_failover", "consistency_breaches", "consistency_latest",
        "spike_recovery_latencies",
    ):
        assert key in snap, f"missing key {key}"


def main() -> int:
    tests = [
        test_drift_band_median_mad,
        test_drift_band_warmup_gate,
        test_observe_drift_advances_cycle,
        test_recency_decay_reduces_visits,
        test_decay_rate_modulated_by_drift,
        test_spike_detection_fires_on_outlier,
        test_spike_handler_enters_failover_and_drops_q,
        test_spike_recovery_latency_recorded,
        test_stale_nodes_priority_order,
        test_reevaluate_snaps_high_staleness_regardless_of_N,
        test_reevaluate_alpha_respects_low_staleness_floor,
        test_reevaluate_pushes_consistency_history,
        test_held_head_snapshot_created,
        test_held_head_refresh_respects_cadence,
        test_failover_triggers_on_sustained_breach,
        test_failover_recovers_when_consistency_drops,
        test_snapshot_contains_expected_keys,
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
