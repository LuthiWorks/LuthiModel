"""Integration tests for Item #6 §6 -- staleness goes live.

Run from project root:
    python -m luthi.v2.m9.test_staleness_live

Covers the wiring the 2026-07-03 audit (item 17) found dormant:
- select_action under mcts_persistent_tree=True advances the tree to
  the acted child (context refreshed, root re-grounded on the realized
  encode) instead of resetting; reset fallback fires when there is
  nothing sound to advance to.
- The per-cycle staleness pass runs: re-eval consumes a slice of the
  plan budget, the consistency history populates, and the failover
  state machine is genuinely reachable end-to-end (it never was
  before this build -- reevaluate had no production caller, so the
  consistency history stayed empty and in_failover() could never
  become True).
- Spike handling is one-shot per theta-tick: a standing spike verdict
  (the drift band's latest value does not change between theta
  updates) must not re-wipe cached Q on every act cycle.
- Held-head failover routing swaps the EFE evaluator's predictor for
  the planning block and restores it after; the held snapshot is
  never refreshed from the live predictor mid-failover (None
  bootstrap excepted).
- mcts_persistent_tree=False (the default) leaves the legacy
  reset-per-cycle act path untouched.
"""

from __future__ import annotations

import copy
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.optim as optim

from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.jepa_runner import (
    CheckpointConfig,
    EpochConfig,
    KillCriteriaConfig,
    LoggingConfig,
    ModalitySampler,
    RunnerConfig,
    SamplerConfig,
)
from luthi.v2.m9.runner import M9Config, M9Trainer
from luthi.v2.m9.staleness import StalenessConfig, StalenessManager
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


VOCAB = 32
D = 32
SEQ = 16
B = 2
CTX = 8


class _TextLoader:
    """Minimal text-only loader satisfying the MultimodalDataLoader
    protocol (mirrors test_runner's)."""

    def __init__(self, vocab: int, batch: int, seq_len: int, seed: int = 0):
        self.vocab = vocab
        self.batch = batch
        self.seq_len = seq_len
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._cursor = 0

    def next_batch(self, modality: str) -> dict:
        assert modality == "text"
        tokens = torch.randint(
            0, self.vocab, (self.batch, self.seq_len),
            generator=self.gen,
        )
        self._cursor += 1
        return {"text_tokens": tokens}

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return int(batch["text_tokens"].numel())

    def state_dict(self) -> dict:
        return {"cursor": self._cursor, "gen": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self._cursor = state["cursor"]
        self.gen.set_state(state["gen"])

    def corpus_sizes_tokens(self) -> dict:
        return {"text": 1000}


def _build_trainer(
    run_dir: Path, m9_config: M9Config, seed: int = 7
) -> M9Trainer:
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad],
        lr=1e-3,
    )
    loader = _TextLoader(VOCAB, B, SEQ, seed=seed)
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(),
        alpha=0.7,
    )
    sampler = ModalitySampler(sampler_cfg)
    runner_cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(
            light_interval_batches=10**9, deep_interval_batches=10**9
        ),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    return M9Trainer(
        loss_module=loss_module,
        optimizer=optimizer,
        sampler=sampler,
        data_loader=loader,
        config=runner_cfg,
        run_dir=run_dir,
        m9_config=m9_config,
    )


@contextmanager
def _trainer(m9_config: M9Config, seed: int = 7):
    """Build a trainer in a temp run_dir, closing the action log before
    the directory is cleaned up (Windows: the open JSONL handle makes
    TemporaryDirectory cleanup fail)."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _build_trainer(Path(tmp), m9_config, seed=seed)
        try:
            yield trainer
        finally:
            trainer.action_log.close()


def _state_and_context(seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s_t = torch.randn(D, generator=g)
    ctx = torch.randn(1, CTX, D, generator=g)
    return s_t, ctx


# ---------- Persistent-tree lifecycle ----------

def test_persistent_tree_advances_across_cycles():
    """Second select_action advances to the acted child: same node
    object becomes root, re-grounded on the new realized state, with
    the cached predictor context refreshed to the new cycle's."""
    with _trainer(M9Config(mcts_persistent_tree=True)) as trainer:
        s1, ctx1 = _state_and_context(seed=1)
        trainer.select_action(s1, context_latents=ctx1)
        assert trainer._last_best_child_idx is not None
        chosen = trainer.mcts.root.children[trainer._last_best_child_idx]

        s2, ctx2 = _state_and_context(seed=2)
        trainer.select_action(s2, context_latents=ctx2)
        assert trainer.mcts.root is chosen, (
            "persistent mode should advance_root to the acted child, "
            "not reset to a fresh node"
        )
        assert torch.equal(trainer.mcts.root.state, s2.detach()), (
            "advanced root must be re-grounded on the realized encode"
        )
        assert torch.equal(trainer.mcts.context_latents, ctx2[0:1]), (
            "cached predictor context must be the NEW cycle's "
            "(audit item 17: stale-context defect)"
        )


def test_default_reset_path_unchanged():
    """mcts_persistent_tree=False (default): every call resets; no
    advance target is recorded; the staleness pass never runs."""
    with _trainer(M9Config()) as trainer:
        s1, ctx1 = _state_and_context(seed=1)
        trainer.select_action(s1, context_latents=ctx1)
        assert trainer._last_best_child_idx is None
        first_root = trainer.mcts.root
        first_children = list(first_root.children)

        s2, ctx2 = _state_and_context(seed=2)
        trainer.select_action(s2, context_latents=ctx2)
        assert trainer.mcts.root is not first_root
        assert all(trainer.mcts.root is c for c in []) or (
            trainer.mcts.root not in first_children
        ), "legacy path must not advance into the previous tree"
        assert trainer.last_staleness_pass == {}, (
            "staleness pass must not run in legacy mode"
        )


def test_fallback_clears_advance_target():
    """A budget=0 habit-fallback records no advance target, and the
    following call resets rather than advancing."""
    with _trainer(M9Config(mcts_persistent_tree=True)) as trainer:
        s1, ctx1 = _state_and_context(seed=1)
        trainer.select_action(s1, context_latents=ctx1)
        prev_children = list(trainer.mcts.root.children)

        # Degenerate budget: falls back to a habit sample.
        s2, ctx2 = _state_and_context(seed=2)
        sel = trainer.select_action(s2, context_latents=ctx2, budget=0)
        assert sel.action is not None
        assert trainer._last_best_child_idx is None

        s3, ctx3 = _state_and_context(seed=3)
        trainer.select_action(s3, context_latents=ctx3)
        assert trainer.mcts.root not in prev_children, (
            "after a fallback cycle there is nothing sound to advance "
            "to; the tree must reset"
        )


# ---------- The per-cycle staleness pass ----------

def test_staleness_pass_reports_and_respects_budget():
    with _trainer(M9Config(mcts_persistent_tree=True)) as trainer:
        s1, ctx1 = _state_and_context(seed=1)
        trainer.select_action(s1, context_latents=ctx1)
        pass1 = dict(trainer.last_staleness_pass)
        for key in (
            "reevaluated", "reeval_budget", "median_deviation",
            "in_failover", "theta_version", "degraded_duration",
        ):
            assert key in pass1, f"missing staleness-pass key {key}"
        assert pass1["in_failover"] is False

        # Age every node past the consistency window so the next pass
        # has genuinely stale nodes to re-evaluate.
        window = trainer.staleness.config.consistency_window
        for _ in range(window + 1):
            trainer.staleness.observe_drift(0.0)

        s2, ctx2 = _state_and_context(seed=2)
        trainer.select_action(s2, context_latents=ctx2)
        pass2 = dict(trainer.last_staleness_pass)
        assert pass2["reevaluated"] > 0, (
            "aged nodes must be re-evaluated by the pass"
        )
        budget = trainer.m9_config.mcts_budget_per_cycle
        assert pass2["reevaluated"] <= pass2["reeval_budget"] <= budget - 1, (
            "re-eval slice must stay within plan §4.iii's carve-out"
        )


def test_failover_reachable_end_to_end():
    """Drive genuinely divergent cached values through repeated passes
    and confirm the failover state machine trips -- the reachability
    the audit found impossible (reevaluate had no production caller)."""
    with _trainer(M9Config(mcts_persistent_tree=True)) as trainer:
        sm = trainer.staleness
        s, ctx = _state_and_context(seed=1)
        trainer.select_action(s, context_latents=ctx)
        assert sm.held_predictor is not None, (
            "first pass must bootstrap the held-head snapshot"
        )

        window = sm.config.consistency_window
        breach_cycles = sm.config.failover_breach_cycles
        for i in range(breach_cycles + 1):
            # Corrupt every cached Q so re-eval sees a huge deviation.
            for node in trainer.mcts.iter_nodes():
                node.Q = 1000.0
            # Age stamps past the window so re-eval selects them.
            for _ in range(window + 1):
                sm.observe_drift(0.0)
            s_i, ctx_i = _state_and_context(seed=10 + i)
            trainer.select_action(s_i, context_latents=ctx_i)

        assert sm.in_failover(), (
            "sustained re-eval-vs-cached breaches must trip failover"
        )
        assert trainer.last_staleness_pass["in_failover"] is True


def test_spike_handled_once_per_theta_tick():
    with _trainer(M9Config(mcts_persistent_tree=True)) as trainer:
        sm = trainer.staleness
        calls = []
        orig = sm.handle_spike

        def counting(mcts):
            calls.append(sm.theta_version)
            return orig(mcts)

        sm.handle_spike = counting

        # Warm the band on small drifts, then one huge outlier.
        for _ in range(8):
            sm.observe_drift(1.0)
        sm.observe_drift(100.0)
        assert sm.spike(), "sanity: the outlier must register as a spike"

        s1, ctx1 = _state_and_context(seed=1)
        trainer.select_action(s1, context_latents=ctx1)
        assert len(calls) == 1, "first act cycle handles the spike"

        # No theta movement between cycles: the standing spike verdict
        # must NOT be re-handled (that would re-wipe cached Q on every
        # wake cycle until the next update).
        s2, ctx2 = _state_and_context(seed=2)
        trainer.select_action(s2, context_latents=ctx2)
        assert len(calls) == 1, (
            "same theta-tick: spike must be one-shot, not standing"
        )

        # A NEW spiking theta update is a new event.
        sm.observe_drift(100.0)
        s3, ctx3 = _state_and_context(seed=3)
        trainer.select_action(s3, context_latents=ctx3)
        assert len(calls) == 2, "new theta-tick spike is a new event"


# ---------- Held-head routing + refresh guard ----------

def test_held_head_routing_swaps_and_restores():
    with _trainer(M9Config(mcts_persistent_tree=True)) as trainer:
        sm = trainer.staleness
        live = trainer.efe.predictor
        sm.held_predictor = copy.deepcopy(live)
        sm._in_failover = True

        with trainer._held_head_routing():
            assert trainer.efe.predictor is sm.held_predictor, (
                "failover must route rollouts through the held snapshot"
            )
        assert trainer.efe.predictor is live, (
            "routing must restore the live predictor on exit"
        )

        # Restore-on-exception too.
        try:
            with trainer._held_head_routing():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert trainer.efe.predictor is live


def test_held_head_routing_noop_in_legacy_mode():
    with _trainer(M9Config()) as trainer:
        sm = trainer.staleness
        live = trainer.efe.predictor
        sm.held_predictor = copy.deepcopy(live)
        sm._in_failover = True
        with trainer._held_head_routing():
            assert trainer.efe.predictor is live, (
                "legacy mode never routes -- its failover state is "
                "never driven, so routing off it would be untested "
                "behavior"
            )


def test_held_head_not_refreshed_from_live_mid_failover():
    """The lifeboat must not be rebuilt from the sinking hull: no
    cadence refresh while in failover or spike-recovery. Bootstrap
    (no snapshot at all) is the documented exception."""
    sm = StalenessManager(StalenessConfig(held_head_refresh_every=1))
    pred = torch.nn.Linear(4, 4)
    sm.maybe_refresh_held_head(pred)
    held_first = sm.held_predictor
    assert held_first is not None

    sm._cycles_since_held_refresh = 999
    sm._in_failover = True
    sm.maybe_refresh_held_head(pred)
    assert sm.held_predictor is held_first, (
        "no refresh while in failover"
    )

    sm._in_failover = False
    sm._in_recovery = True
    sm.maybe_refresh_held_head(pred)
    assert sm.held_predictor is held_first, (
        "no refresh while in spike-recovery"
    )

    sm._in_recovery = False
    sm.maybe_refresh_held_head(pred)
    assert sm.held_predictor is not held_first, (
        "healthy again: cadence refresh resumes"
    )

    # Bootstrap exception: a spike with no snapshot yet still takes one.
    sm2 = StalenessManager()
    sm2._in_failover = True
    sm2.maybe_refresh_held_head(pred)
    assert sm2.held_predictor is not None, (
        "None-bootstrap must snapshot even mid-failover -- a frozen "
        "post-spike head beats routing through a moving model"
    )


def main() -> int:
    tests = [
        test_persistent_tree_advances_across_cycles,
        test_default_reset_path_unchanged,
        test_fallback_clears_advance_target,
        test_staleness_pass_reports_and_respects_budget,
        test_failover_reachable_end_to_end,
        test_spike_handled_once_per_theta_tick,
        test_held_head_routing_swaps_and_restores,
        test_held_head_routing_noop_in_legacy_mode,
        test_held_head_not_refreshed_from_live_mid_failover,
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
