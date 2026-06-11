"""Persistent MCTS over continuous actions with progressive widening.

Per spec §3 build list item 3 and plan §3:
- Continuous action space (256d at launch). Children proposed by
  the HabitNet's per-state Gaussian. Progressive widening: a node's
  child count grows as `ceil(widening_c * (N + 1) ** widening_alpha)`
  (Couetoux et al. 2011) so the tree avoids the curse of dimension-
  ality without committing prematurely to a small action set.
- Persistent tree across cycles: `plan_budget(budget)` adds at most
  `budget` simulations to the existing tree and returns. Each cycle
  the act-phase reads the current best action; the plan-budget
  phase contributes more expansion. A full planning step spans ~3-10
  cycles. Cross-cycle staleness machinery (recency-decay, drift-tied
  refresh, held-head failover) lives in the separate `staleness`
  module and reads / writes node `theta_stamp` and Q/N here.
- Budget-limited expansion: caller controls cost. The 20-30 ms
  plan-budget phase fits ~5-10 simulations at our scale; the tree
  grows over many cycles, amortizing.

Value convention. EFE is a cost (lower = better); MCTS backs up
expected `-G`. The value at a child = `-G(action leading here) +
V_head(child.state)` (one-step bootstrap when no descendants), or
the running mean of backed-up values when descendants exist. Best
action at the root = highest-visit child (the standard MCTS
choice; also robust to value-estimate noise).

Step-1 simplifications, documented for later revision:
- Unbatched: tree operates on a single state (the entity is planning
  from one place). Training-time MCTS over batched states is a
  separate concern; the predictor itself is still batched.
- PUCT prior `P(child)` uses softmax-normalized habit-net log-probs
  over the current children (recomputed at each PUCT step) -- not a
  fixed prior set at expansion time, which would freeze as the
  habit-net updates. Cheap and tracks habit-net updates naturally.
- Theta-stamping records *which simulation counter* a node was last
  evaluated under. Staleness compares against the current counter;
  the actual `||Delta theta||` accumulator lives in the staleness
  module (this file only exposes the hook).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class MCTSNode:
    """One node in the persistent tree.

    state           : [D] state vector at this node
    parent          : parent MCTSNode, or None for root
    action_in       : [D] action that led to this node (None for root)
    incoming_g      : G(a) of the action that led here (None for root) --
                      cached so backup can compose value = -g_in + V(s)
    N               : visit count
    Q               : running-mean backed-up value (the MCTS estimate
                      of expected -G over future rollout from here)
    children        : list of child MCTSNodes (grown by progressive widening)
    habit_log_prob  : log-prob of this node's action under the habit
                      net at the time of creation (used to recompute
                      PUCT priors on the fly)
    theta_stamp     : simulation counter at which this node was last
                      evaluated (the staleness module reads this)
    """

    state: torch.Tensor
    parent: "MCTSNode | None" = None
    action_in: torch.Tensor | None = None
    incoming_g: float | None = None
    N: int = 0
    Q: float = 0.0
    children: list = field(default_factory=list)
    habit_log_prob: float = 0.0
    theta_stamp: int = 0


class MCTS:
    """Persistent MCTS with progressive widening and habit-net proposals.

    Usage pattern:
        mcts = MCTS(habit_net, efe_evaluator, value_head, ...)
        mcts.reset(s_t, context_latents, target_positions)
        # Each cycle:
        mcts.plan_budget(budget=5)
        best_action = mcts.best_action()  # the chosen a_t this cycle
        # After acting, environment provides s_{t+1}; advance the tree:
        mcts.advance_root(chosen_child_index)  # discard siblings, root := child
    """

    def __init__(
        self,
        habit_net: nn.Module,
        efe_evaluator: nn.Module,
        value_head: nn.Module | None = None,
        widening_alpha: float = 0.5,
        widening_c: float = 1.0,
        c_puct: float = 1.0,
        max_depth: int = 1,
    ):
        self.habit_net = habit_net
        self.efe = efe_evaluator
        self.value_head = value_head
        self.widening_alpha = widening_alpha
        self.widening_c = widening_c
        self.c_puct = c_puct
        # Max tree depth. Step-1 launches H=1 (depth-1 tree); higher H
        # extends the rollout. Spec §2 cap is "rollout-compute-within-
        # inter-update-interval" not a depth cap; setting via this param.
        self.max_depth = max_depth
        # Persistent root + cached context.
        self.root: MCTSNode | None = None
        self._context_latents: torch.Tensor | None = None
        self._target_positions: torch.Tensor | None = None
        # Simulation counter -- monotonically increasing; staleness reads.
        self.sim_counter: int = 0

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------
    def reset(
        self,
        root_state: torch.Tensor,
        context_latents: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> None:
        """Start a fresh tree at `root_state`. Caches per-cycle context
        the predictor needs (the encoder output context and the target
        position queries) so simulations don't have to re-thread them.
        """
        if root_state.dim() == 2:
            root_state = root_state.squeeze(0)
        self.root = MCTSNode(state=root_state.detach())
        self._context_latents = context_latents
        self._target_positions = target_positions
        self.sim_counter = 0

    def advance_root(self, chosen_child_index: int) -> None:
        """After acting on a child's action, slide the tree forward.

        Drops siblings; the chosen child becomes the new root. The
        sub-tree under it is preserved (its visit counts carry forward,
        amortizing prior planning work). This is the persistence the
        spec asks for across cycles when the entity actually commits
        to an action.
        """
        if self.root is None or not self.root.children:
            raise RuntimeError(
                "Cannot advance: tree has no children yet"
            )
        if not (0 <= chosen_child_index < len(self.root.children)):
            raise IndexError(chosen_child_index)
        new_root = self.root.children[chosen_child_index]
        new_root.parent = None
        new_root.action_in = None
        new_root.incoming_g = None
        self.root = new_root

    # ------------------------------------------------------------------
    # Public planning API.
    # ------------------------------------------------------------------
    def plan_budget(
        self,
        budget: int,
        observation_kwargs: dict | None = None,
    ) -> None:
        """Run up to `budget` simulations from the current root.

        `observation_kwargs` are threaded into EFE evaluation -- any
        of decoder_reencodes / counterpart_present /
        time_since_emission / a_reencoded (per Preferences). Missing
        keys zero those preference features (Preferences.pragmatic_cost
        zero-on-missing).
        """
        if self.root is None:
            raise RuntimeError("Call reset() before plan_budget()")
        for _ in range(budget):
            self.sim_counter += 1
            self._simulate(observation_kwargs or {})

    def best_action(self) -> torch.Tensor | None:
        """Most-visited child's `action_in` -- the standard MCTS choice
        (robust to value-estimate noise). Returns None if the tree
        has not been expanded yet.
        """
        if self.root is None or not self.root.children:
            return None
        best = max(self.root.children, key=lambda c: c.N)
        return best.action_in

    def visit_distribution(self) -> torch.Tensor:
        """Per-child visit-count distribution at the root. Used for
        habit-net distillation (cross-entropy target) and for the
        K-M9-2 MCTS-pathology kill (entropy collapse signal).
        """
        if self.root is None or not self.root.children:
            return torch.zeros(0)
        counts = torch.tensor(
            [c.N for c in self.root.children], dtype=torch.float32
        )
        total = counts.sum()
        if total == 0:
            return torch.zeros_like(counts)
        return counts / total

    def tree_stats(self) -> dict:
        """Summary for instrumentation. Used by per-cycle action log
        and by the K-M9-2 entropy / K-M9-7 staleness kills.
        """
        if self.root is None:
            return {"size": 0, "root_visits": 0, "top_share": 0.0}
        size = self._count_nodes(self.root)
        if not self.root.children:
            return {
                "size": size,
                "root_visits": self.root.N,
                "top_share": 0.0,
            }
        dist = self.visit_distribution()
        return {
            "size": size,
            "root_visits": self.root.N,
            "top_share": float(dist.max().item()),
        }

    # ------------------------------------------------------------------
    # Core simulation steps.
    # ------------------------------------------------------------------
    def _simulate(self, observation_kwargs: dict) -> None:
        """One simulation: select -> (maybe expand) -> evaluate -> backup."""
        node, depth = self._select(self.root)
        # Decide whether to expand the chosen node. Three cases:
        # (a) node is unvisited (N=0): evaluate as a leaf, no expand.
        # (b) node has children-room per progressive widening AND we're
        #     under max_depth: expand one child.
        # (c) at max_depth: just re-evaluate (refresh).
        if node.N == 0:
            value = self._evaluate(node)
            self._backup(node, value)
            return

        if depth < self.max_depth and self._has_widening_room(node):
            child = self._expand(node, observation_kwargs)
            if child is None:
                # Expansion failed (e.g. habit net produced None). Just
                # re-evaluate the parent.
                value = self._evaluate(node)
                self._backup(node, value)
                return
            value = self._evaluate(child)
            self._backup(child, value)
        else:
            # No room to grow; re-evaluate this node to refresh.
            value = self._evaluate(node)
            self._backup(node, value)

    def _select(self, root: MCTSNode) -> tuple[MCTSNode, int]:
        """PUCT traversal. Returns the selected leaf-ish node + depth."""
        node = root
        depth = 0
        while node.children and not self._has_widening_room(node):
            node = self._best_puct_child(node)
            depth += 1
        return node, depth

    def _has_widening_room(self, node: MCTSNode) -> bool:
        """Progressive widening: target_children = ceil(c * (N+1)^alpha)."""
        target = math.ceil(self.widening_c * (node.N + 1) ** self.widening_alpha)
        return len(node.children) < target

    def _best_puct_child(self, node: MCTSNode) -> MCTSNode:
        """PUCT: child argmax of Q + c_puct * P * sqrt(N_parent)/(1+N_child).

        Prior P over current children = softmax of stored habit_log_prob.
        Re-normalized each call so newly-added children appear in the
        prior distribution naturally.
        """
        logps = torch.tensor(
            [c.habit_log_prob for c in node.children], dtype=torch.float64
        )
        priors = torch.softmax(logps, dim=0)
        n_parent = max(node.N, 1)
        scores = []
        for i, child in enumerate(node.children):
            puct = (
                child.Q
                + self.c_puct
                * float(priors[i].item())
                * math.sqrt(n_parent)
                / (1.0 + child.N)
            )
            scores.append(puct)
        return node.children[int(torch.tensor(scores).argmax().item())]

    def _expand(
        self,
        node: MCTSNode,
        observation_kwargs: dict,
    ) -> MCTSNode | None:
        """Propose one action from the habit net; create child via predictor."""
        s = node.state.unsqueeze(0)  # [1, D]
        sample = self.habit_net.sample(s, K=1)
        action = sample["candidates"][0, 0].detach()  # [D]
        log_prob = float(sample["log_prob"][0, 0].detach().item())

        # Evaluate G(action) via EFE on the *current* root context
        # (single-state planning; deeper rollouts re-use the same cached
        # context which is fine for H=1 -- when H grows we'll need to
        # re-encode along the rollout).
        with torch.no_grad():
            out = self.efe.compute_g(
                s_t=node.state.unsqueeze(0),
                a_t=action.unsqueeze(0),
                context_latents=self._context_latents,
                target_positions=self._target_positions,
                **observation_kwargs,
            )
        s_hat = out["s_hat_next"][0].detach()  # [D]
        g = float(out["G"][0].item())

        child = MCTSNode(
            state=s_hat,
            parent=node,
            action_in=action,
            incoming_g=g,
            habit_log_prob=log_prob,
            theta_stamp=self.sim_counter,
        )
        node.children.append(child)
        return child

    def _evaluate(self, node: MCTSNode) -> float:
        """Bootstrap value at `node` from the value head, or zero if no
        head. Composes with the cost of the action that *led* here so
        the propagated value is `-g_in + V(state)`.

        At a non-root node the value of "being at this state" is
        -V_head(state) ... actually V_head is supposed to estimate
        expected -G; so V(state) IS the value-of-being-here. Compose
        with the action's own cost outside this method, in _backup.
        """
        node.theta_stamp = self.sim_counter
        if self.value_head is None:
            return 0.0
        with torch.no_grad():
            v = self.value_head(node.state.unsqueeze(0)).detach()
        return float(v.item())

    def _backup(self, node: MCTSNode, leaf_value: float) -> None:
        """Walk up tree updating N and Q.

        The propagated value at each step is `-incoming_g + accumulated`
        where accumulated starts at `leaf_value` and gathers the cost of
        each edge traversed. This makes Q the MCTS estimate of expected
        `-G` over the path from a node to the rollout horizon.
        """
        value = leaf_value
        while node is not None:
            if node.incoming_g is not None:
                # Cost of the edge into this node contributes to the
                # value at the parent's perspective.
                value = -node.incoming_g + value
            # Running-mean update of Q at this node.
            node.N += 1
            node.Q = node.Q + (value - node.Q) / node.N
            node = node.parent

    def _count_nodes(self, node: MCTSNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)

    # ------------------------------------------------------------------
    # Hooks for the staleness module to read/write.
    # ------------------------------------------------------------------
    def iter_nodes(self):
        """Yield every node in the tree, root first (DFS)."""
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children)
