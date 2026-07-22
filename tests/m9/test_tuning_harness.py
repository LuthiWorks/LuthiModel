"""Validates the staleness tuning harness (tuning_harness.py).

Two modes, per Fable's "max_depth >= 2 OR an aged-node harness":
 - run_aged_node_regime deterministically ages retained nodes (θ clock advanced
   without re-planning), so it exercises the re-eval + deviation machinery on a
   SMOKE model NOW — the plumbing signal ahead of the trained checkpoint.
 - run_natural_regime drives real θ movement at depth >= 2; it is the real
   tuning entry but needs a stable (trained) policy to make nodes persist and
   age, so on smoke it just has to drive cleanly and collect the plumbing.
The tuning pass reruns these on a trained checkpoint for real deviation values.
"""

from __future__ import annotations

import pytest

from luthi.v2.m9.runner import M9Config
from luthi.v2.m9.tuning_harness import (
    run_natural_regime, run_aged_node_regime, StalenessRegimeStats,
)

# Reuse the staleness-live smoke fixture (trainer builder + state generator).
from tests.m9.test_staleness_live import _trainer, _state_and_context


def _next_batch(trainer):
    def next_batch(cycle):
        return "text", trainer.data_loader.next_batch("text")
    return next_batch


def _next_state(cycle):
    return _state_and_context(seed=100 + cycle)


def test_aged_node_regime_produces_reevals():
    """Deterministic: aged retained nodes are re-evaluated on a smoke model,
    populating the consistency-deviation distribution the F2 thresholds tune
    against (plumbing signal — realistic VALUES still need a checkpoint)."""
    with _trainer(M9Config(
        mcts_persistent_tree=True, mcts_max_depth=2, mcts_budget_per_cycle=24,
    )) as trainer:
        stats = run_aged_node_regime(trainer, _next_state, cycles=15)

    assert stats.total_reevaluated > 0, (
        "aged retained nodes must be re-evaluated (the machinery under test)"
    )
    assert len(stats.deviation_samples) > 0, (
        "the consistency-deviation distribution must be collected"
    )
    summ = stats.summary()
    assert summ["cycles"] == 15
    assert summ["deviation_mean"] is not None


def test_natural_regime_drives_and_collects():
    """Real θ movement at depth >= 2: on a smoke model re-evals may be zero
    (an untrained policy churns the retained subtree — documented), so the
    contract here is only that it drives cleanly and advances the θ clock."""
    with _trainer(M9Config(
        mcts_persistent_tree=True, mcts_max_depth=2, mcts_budget_per_cycle=24,
    )) as trainer:
        stats = run_natural_regime(
            trainer, _next_batch(trainer), _next_state, cycles=12,
        )
    assert stats.summary()["cycles"] == 12
    # θ advanced once per cycle (train_step), so the clock is monotonic.
    assert stats.theta_versions == sorted(stats.theta_versions)
    assert stats.theta_versions[-1] >= 12


def test_summary_shape_is_tuning_ready():
    """The summary carries exactly what the tuning pass reads (distribution
    stats + failover events), and nothing it must not (no threshold set here)."""
    stats = StalenessRegimeStats()
    stats.record({"reevaluated": 3, "median_deviation": 0.4,
                  "in_failover": False, "theta_version": 1}, cycle=0)
    stats.record({"reevaluated": 0, "median_deviation": None,
                  "in_failover": True, "theta_version": 2}, cycle=1)
    summ = stats.summary()
    assert summ["deviation_n"] == 1          # the None/zero-reeval cycle excluded
    assert summ["deviation_mean"] == pytest.approx(0.4)
    assert summ["failover_cycles"] == [1]
    assert stats.any_failover is True


def test_natural_regime_rejects_depth1():
    """At depth 1 the natural regime refuses to run — the advanced root is
    childless and nothing can age, so it would silently collect zeros (the
    false-'covered' trap). Loud is correct."""
    with _trainer(M9Config(
        mcts_persistent_tree=True, mcts_max_depth=1,
    )) as trainer:
        with pytest.raises(ValueError, match="max_depth"):
            run_natural_regime(
                trainer, _next_batch(trainer), _next_state, cycles=1,
            )


def test_regimes_reject_nonpersistent():
    with _trainer(M9Config(
        mcts_persistent_tree=False, mcts_max_depth=2,
    )) as trainer:
        with pytest.raises(ValueError, match="persistent"):
            run_aged_node_regime(trainer, _next_state, cycles=1)
        with pytest.raises(ValueError, match="persistent"):
            run_natural_regime(
                trainer, _next_batch(trainer), _next_state, cycles=1,
            )
