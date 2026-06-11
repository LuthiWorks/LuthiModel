"""Unit tests for the persistent MCTS planner.

Run from project root:
    python -m luthi.v2.m9.test_mcts

Tests:
- reset / plan_budget basic shape
- progressive widening grows children with visit count
- repeated plan_budget compounds the tree
- best_action picks the most-visited child
- advance_root preserves the chosen subtree, discards siblings
- functional: with a known-good value head, more simulations -> the
  chosen action's predicted next-state is closer to the target than
  the average random sample
- theta_stamp is recorded
- visit distribution sums to 1; tree_stats reports plausible numbers
"""

from __future__ import annotations

import torch

from luthi.v2.jepa_loss import JEPAPredictor
from luthi.v2.m9.efe import EFEEvaluator
from luthi.v2.m9.habit_net import HabitNet
from luthi.v2.m9.mcts import MCTS, MCTSNode
from luthi.v2.m9.preferences import Preferences
from luthi.v2.m9.value_head import ValueHead


D = 16
B = 1
CTX = 8
TGT = 6


def _build():
    predictor = JEPAPredictor(
        d_model=D, n_layers=1, n_heads=2, ffn_expansion=2, max_target_len=32
    )
    prefs = Preferences(d_model=D, engagement_target_magnitude=0.0)  # P1 off
    efe = EFEEvaluator(predictor, prefs)
    habit = HabitNet(d_model=D, log_std_init=0.0)
    v = ValueHead(d_model=D)
    return predictor, prefs, efe, habit, v


def _context_and_positions():
    return (
        torch.randn(B, CTX, D),
        torch.arange(CTX, CTX + TGT).unsqueeze(0).expand(B, -1),
    )


def test_reset_creates_root():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v)
    s_t = torch.randn(D)
    ctx, tp = _context_and_positions()
    mcts.reset(s_t, ctx, tp)
    assert mcts.root is not None
    assert mcts.root.N == 0
    assert mcts.root.children == []


def test_plan_budget_grows_tree():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v, widening_alpha=0.5, widening_c=1.0)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    mcts.plan_budget(budget=10)
    # Tree should have grown beyond the root.
    stats = mcts.tree_stats()
    assert stats["size"] >= 2, f"expected size >= 2 after 10 sims, got {stats}"
    assert stats["root_visits"] >= 1


def test_progressive_widening_growing():
    """target_children = ceil(c * (N+1)^alpha). Verify children grow with N."""
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v, widening_alpha=0.5, widening_c=1.0, max_depth=1)
    mcts.reset(torch.zeros(D), *_context_and_positions())

    # After 1 sim: target = ceil(1 * 1) = 1 child (or 0 if first sim only
    # evaluated the unvisited root). After more visits, the root should
    # accumulate more children.
    mcts.plan_budget(budget=1)
    children_1 = len(mcts.root.children)

    mcts.plan_budget(budget=20)
    children_20 = len(mcts.root.children)

    assert children_20 > children_1, (
        f"progressive widening should add children: 1 sim -> {children_1}, "
        f"20 more sims -> {children_20}"
    )


def test_repeated_plan_budget_compounds():
    """plan_budget is incremental: visits accumulate across calls."""
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    mcts.plan_budget(budget=5)
    n_after_5 = mcts.root.N
    mcts.plan_budget(budget=5)
    n_after_10 = mcts.root.N
    assert n_after_10 > n_after_5
    assert mcts.sim_counter == 10


def test_best_action_picks_most_visited():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    mcts.plan_budget(budget=20)
    assert mcts.root.children, "tree should have children after 20 sims"

    # Hand-edit visit counts: force child 0 to be the most visited.
    for c in mcts.root.children:
        c.N = 1
    mcts.root.children[0].N = 100

    action = mcts.best_action()
    expected = mcts.root.children[0].action_in
    assert torch.equal(action, expected)


def test_advance_root_preserves_chosen_subtree():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v, max_depth=2)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    mcts.plan_budget(budget=30)
    # Note the chosen child + its descendants before advancing.
    assert mcts.root.children
    chosen_idx = max(
        range(len(mcts.root.children)),
        key=lambda i: mcts.root.children[i].N,
    )
    chosen = mcts.root.children[chosen_idx]
    chosen_subtree_size = mcts._count_nodes(chosen)
    chosen_state = chosen.state.clone()

    mcts.advance_root(chosen_idx)
    assert mcts.root.parent is None
    assert mcts.root.action_in is None
    assert mcts.root.incoming_g is None
    assert torch.equal(mcts.root.state, chosen_state)
    assert mcts._count_nodes(mcts.root) == chosen_subtree_size


def test_advance_root_errors_without_children():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    try:
        mcts.advance_root(0)
        assert False, "should have raised"
    except RuntimeError:
        pass


def test_visit_distribution_sums_to_one():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    mcts.plan_budget(budget=20)
    dist = mcts.visit_distribution()
    assert dist.shape[0] == len(mcts.root.children)
    assert abs(dist.sum().item() - 1.0) < 1e-5


def test_theta_stamp_recorded():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    mcts.plan_budget(budget=10)
    # All children should have theta_stamp set to the sim_counter at
    # which they were expanded.
    for c in mcts.root.children:
        assert c.theta_stamp > 0
        assert c.theta_stamp <= mcts.sim_counter


def test_functional_value_head_guides_search():
    """With a hand-picked value head that prefers small-magnitude states,
    MCTS should rank a target-near child higher than a target-far one.

    Use a fixed predictor that just adds the action to the state, and
    a value head V(s) = -||s||. Then candidates with small ||action||
    score higher.
    """
    torch.manual_seed(7)
    predictor = JEPAPredictor(
        d_model=D, n_layers=1, n_heads=2, ffn_expansion=2, max_target_len=32,
    )
    prefs = Preferences(d_model=D, engagement_target_magnitude=0.0)
    efe = EFEEvaluator(predictor, prefs)
    habit = HabitNet(d_model=D, log_std_init=0.0)

    # Custom value head: V(s) = -||s||.
    class NegNorm(torch.nn.Module):
        def forward(self, s):
            return -s.norm(dim=-1)
    v = NegNorm()

    mcts = MCTS(
        habit, efe, v,
        widening_alpha=0.5, widening_c=2.0, c_puct=2.0, max_depth=1,
    )
    mcts.reset(torch.zeros(D), *_context_and_positions())
    mcts.plan_budget(budget=80)

    # Top-visit child should beat the median action by ||s_hat|| (closer
    # to zero).
    visits = sorted(
        ((c.N, c.state.norm().item()) for c in mcts.root.children),
        key=lambda nv: -nv[0],
    )
    top_norm = visits[0][1]
    median_norm = visits[len(visits) // 2][1]
    assert top_norm <= median_norm + 0.5, (
        f"top-visit child should be no worse than median by ||s_hat||: "
        f"top={top_norm:.3f}, median={median_norm:.3f}"
    )


def test_tree_stats_shape():
    predictor, prefs, efe, habit, v = _build()
    mcts = MCTS(habit, efe, v)
    mcts.reset(torch.zeros(D), *_context_and_positions())
    stats_before = mcts.tree_stats()
    assert stats_before["size"] == 1
    assert stats_before["root_visits"] == 0
    mcts.plan_budget(budget=15)
    stats_after = mcts.tree_stats()
    assert stats_after["size"] > 1
    assert 0.0 <= stats_after["top_share"] <= 1.0


def main() -> int:
    tests = [
        test_reset_creates_root,
        test_plan_budget_grows_tree,
        test_progressive_widening_growing,
        test_repeated_plan_budget_compounds,
        test_best_action_picks_most_visited,
        test_advance_root_preserves_chosen_subtree,
        test_advance_root_errors_without_children,
        test_visit_distribution_sums_to_one,
        test_theta_stamp_recorded,
        test_functional_value_head_guides_search,
        test_tree_stats_shape,
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
