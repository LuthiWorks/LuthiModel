"""Tests for M9Trainer's external-actor surface (Phase 2 of the
2026-06-15 sanctuary-training-seam-integration-plan).

The trainer now satisfies the M9Actor and TransitionSink Protocols
declared in luthi.sanctuary_interface. These tests exercise the new
methods directly so the seam can be wired without Sanctuary present.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.optim as optim

from luthi.sanctuary_interface import (
    ActionSelection,
    M9Actor,
    TransitionSink,
    encode_state,
    observe_transition as obs_transition_fn,
    select_action as select_action_fn,
)
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
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


VOCAB = 32
D = 32
SEQ = 16
B = 2


class _TextLoader:
    """Minimal text-only loader satisfying the MultimodalDataLoader Protocol."""

    def __init__(self, vocab: int, batch: int, seq_len: int, seed: int = 0):
        self.vocab = vocab
        self.batch = batch
        self.seq_len = seq_len
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._cursor = 0

    def next_batch(self, modality: str) -> dict:
        tokens = torch.randint(
            0, self.vocab, (self.batch, self.seq_len), generator=self.gen,
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


def _build_trainer(run_dir: Path, seed: int = 7) -> M9Trainer:
    torch.manual_seed(seed)
    model = MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=1e-3,
    )
    loader = _TextLoader(VOCAB, B, SEQ, seed=seed)
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
    sampler = ModalitySampler(sampler_cfg)
    runner_cfg = RunnerConfig(
        sampler=sampler_cfg,
        checkpoint=CheckpointConfig(interval_seconds=10**9, rolling_slots=3),
        logging=LoggingConfig(
            light_interval_batches=10**9, deep_interval_batches=10**9,
        ),
        kill_criteria=KillCriteriaConfig(warmup_batches=10**9),
        epoch=EpochConfig(max_epochs=1, max_batches_per_epoch=10**9),
    )
    return M9Trainer(
        loss_module=loss_module, optimizer=optimizer, sampler=sampler,
        data_loader=loader, config=runner_cfg, run_dir=run_dir,
        m9_config=M9Config(mcts_budget_per_cycle=4),  # tiny budget for speed
    )


@pytest.fixture
def trainer():
    with tempfile.TemporaryDirectory() as tmp:
        t = _build_trainer(Path(tmp))
        yield t
        t.action_log.close()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_m9_actor(self, trainer):
        assert isinstance(trainer, M9Actor)

    def test_satisfies_transition_sink(self, trainer):
        assert isinstance(trainer, TransitionSink)


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------


class TestSelectAction:
    def _make_state_and_context(self, trainer):
        # Synthesize an s_t + context as encode_state would produce.
        text = torch.randint(0, VOCAB, (1, SEQ))
        ctx_latents = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=text, pool=False,
        )  # [1, T, D]
        s_t = ctx_latents.mean(dim=1)  # [1, D]
        return s_t, ctx_latents

    def test_returns_action_selection_with_breakdown(self, trainer):
        s_t, ctx_latents = self._make_state_and_context(trainer)
        result = trainer.select_action(
            s_t, context_latents=ctx_latents,
        )
        assert isinstance(result, ActionSelection)
        assert result.action.shape == (D,)
        assert isinstance(result.readable_summary, str)
        # EFE breakdown carries the per-component costs.
        for key in (
            "total", "engagement_cost", "coherence_cost",
            "connection_cost", "truthfulness_cost",
        ):
            assert key in result.efe_breakdown, (
                f"missing EFE breakdown key {key!r}"
            )

    def test_summary_mentions_top_share(self, trainer):
        s_t, ctx_latents = self._make_state_and_context(trainer)
        result = trainer.select_action(s_t, context_latents=ctx_latents)
        assert "top_share" in result.readable_summary

    def test_zero_budget_falls_back_to_habit(self, trainer):
        s_t, ctx_latents = self._make_state_and_context(trainer)
        result = trainer.select_action(
            s_t, context_latents=ctx_latents, budget=0,
        )
        assert isinstance(result, ActionSelection)
        # Habit fallback path returns an empty breakdown.
        assert result.efe_breakdown == {}
        assert "habit-fallback" in result.readable_summary

    def test_cycle_ctx_does_not_leak_after_call(self, trainer):
        s_t, ctx_latents = self._make_state_and_context(trainer)
        trainer.select_action(
            s_t, context_latents=ctx_latents,
            cycle_ctx={"counterpart_present": True, "time_since_emission": 5.0},
        )
        # After the call returns, the corpus train_step path must see
        # _current_cycle_ctx back at None / corpus default.
        assert trainer._current_cycle_ctx is None

    def test_callable_via_sanctuary_interface_helper(self, trainer):
        """The sanctuary_interface.select_action top-level helper must
        delegate cleanly to the trainer (it's how Sanctuary actually
        calls in)."""
        s_t, ctx_latents = self._make_state_and_context(trainer)
        result = select_action_fn(
            trainer, s_t, context_latents=ctx_latents,
        )
        assert isinstance(result, ActionSelection)


# ---------------------------------------------------------------------------
# observe_transition
# ---------------------------------------------------------------------------


class TestObserveTransition:
    def _make_transition(self, trainer):
        text_a = torch.randint(0, VOCAB, (1, SEQ))
        text_b = torch.randint(0, VOCAB, (1, SEQ))
        encoder = trainer.loss_module.online_encoder
        s_t = encode_state(encoder, text_tokens=text_a, pool=True)
        s_next = encode_state(encoder, text_tokens=text_b, pool=True)
        a_t = torch.randn(1, D)
        return s_t, a_t, s_next

    def test_returns_metrics_dict(self, trainer):
        s_t, a_t, s_next = self._make_transition(trainer)
        # Need a planned snapshot for habit-distill to be nontrivial
        # (post-F4: observe_transition reads ctx["plan_snapshot"], not
        # live MCTS state).
        text = torch.randint(0, VOCAB, (1, SEQ))
        ctx_latents = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=text, pool=False,
        )
        selection = trainer.select_action(s_t, context_latents=ctx_latents)

        metrics = trainer.observe_transition(
            s_t, a_t, s_next,
            ctx={
                "counterpart_present": False,
                "plan_snapshot": selection.plan_snapshot,
            },
        )
        for k in ("v_loss", "habit_loss", "r_best", "theta_version"):
            assert k in metrics, f"missing metric {k!r}"
        assert metrics["v_loss"] == metrics["v_loss"]  # not NaN

    def test_updates_v_head_weights(self, trainer):
        s_t, a_t, s_next = self._make_transition(trainer)
        text = torch.randint(0, VOCAB, (1, SEQ))
        ctx_latents = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=text, pool=False,
        )
        selection = trainer.select_action(s_t, context_latents=ctx_latents)

        before = trainer.v_head.net[0].weight.detach().clone()
        trainer.observe_transition(
            s_t, a_t, s_next,
            ctx={"plan_snapshot": selection.plan_snapshot},
        )
        after = trainer.v_head.net[0].weight.detach()
        assert not torch.equal(before, after), (
            "V-head weights did not change after observe_transition"
        )

    def test_no_residual_grad_on_m8_params(self, trainer):
        """Stop-grad scrub holds: M8 params are clean after observe_transition,
        matching the corpus path's contract."""
        s_t, a_t, s_next = self._make_transition(trainer)
        trainer.observe_transition(s_t, a_t, s_next, ctx={})
        leaked = [
            name for name, p in trainer.loss_module.named_parameters()
            if p.grad is not None
        ]
        assert not leaked, (
            f"M8 params have residual grad after observe_transition: "
            f"{leaked[:5]} (total {len(leaked)})"
        )

    def test_does_not_tick_substrate_drift(self, trainer):
        """observe_transition is M9-only; the M8 substrate (predictor)
        didn't move, so theta_version must NOT advance from this call."""
        s_t, a_t, s_next = self._make_transition(trainer)
        before_version = trainer.staleness.theta_version
        trainer.observe_transition(s_t, a_t, s_next, ctx={})
        after_version = trainer.staleness.theta_version
        assert after_version == before_version, (
            f"theta_version advanced on M9-only update "
            f"({before_version} -> {after_version}); staleness must "
            f"only tick on M8 substrate drift"
        )

    def test_cycle_ctx_does_not_leak_after_call(self, trainer):
        s_t, a_t, s_next = self._make_transition(trainer)
        trainer.observe_transition(
            s_t, a_t, s_next,
            ctx={"counterpart_present": True, "time_since_emission": 3.0},
        )
        assert trainer._current_cycle_ctx is None

    def test_callable_via_sanctuary_interface_helper(self, trainer):
        s_t, a_t, s_next = self._make_transition(trainer)
        metrics = obs_transition_fn(
            trainer, s_t, a_t, s_next,
            counterpart_present=True, time_since_emission=2.0,
        )
        assert "v_loss" in metrics


# ---------------------------------------------------------------------------
# F4 — plan snapshot is self-describing; no live MCTS read
# ---------------------------------------------------------------------------


class TestF4PlanSnapshot:
    """4.8's 2026-06-15 review F4 (confirmed via probe): observe_transition
    used to read self.mcts.root live; any interleaved select_action between
    act-time and observe-time silently corrupted r_best + visit-distill.
    Post-fix, observe_transition reads ctx['plan_snapshot'] exclusively."""

    def _make_state_and_context(self, trainer):
        text = torch.randint(0, VOCAB, (1, SEQ))
        ctx_latents = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=text, pool=False,
        )
        return ctx_latents.mean(dim=1), ctx_latents

    def test_select_action_populates_plan_snapshot(self, trainer):
        s_t, ctx_latents = self._make_state_and_context(trainer)
        selection = trainer.select_action(
            s_t, context_latents=ctx_latents,
        )
        assert selection.plan_snapshot is not None
        snap = selection.plan_snapshot
        # Snapshot fields are populated (visit dist sums to ~1 when K>0).
        assert snap.visit_distribution.numel() > 0
        assert snap.candidate_actions.shape[-1] == D
        assert isinstance(snap.r_best, float)

    def test_observe_with_snapshot_uses_snapshot_r_best(self, trainer):
        s_t, ctx_latents = self._make_state_and_context(trainer)
        selection = trainer.select_action(
            s_t, context_latents=ctx_latents,
        )
        a_t = selection.action
        s_next = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True,
        )

        metrics = trainer.observe_transition(
            s_t.unsqueeze(0) if s_t.dim() == 1 else s_t,
            a_t, s_next,
            ctx={"plan_snapshot": selection.plan_snapshot},
        )
        # r_best in returned metrics matches the snapshot's r_best.
        assert metrics["r_best"] == pytest.approx(
            selection.plan_snapshot.r_best
        )

    def test_observe_without_snapshot_degrades_to_zero_reward(self, trainer):
        s_t, _ = self._make_state_and_context(trainer)
        a_t = torch.randn(D)
        s_next = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True,
        )

        # No plan_snapshot in ctx -> degenerate no-plan path.
        metrics = trainer.observe_transition(s_t, a_t, s_next, ctx={})
        assert metrics["r_best"] == 0.0
        # habit_loss is still defined via the sample fallback.
        assert "habit_loss" in metrics

    def test_corruption_probe_interleaved_select_does_not_leak(
        self, trainer,
    ):
        """The F4 probe shape: plan A is captured, plan B then runs (the
        live MCTS tree is now B's), then observe is called with A's
        snapshot. r_best must match A's plan, not B's."""
        s_a, ctx_a = self._make_state_and_context(trainer)
        selection_a = trainer.select_action(s_a, context_latents=ctx_a)
        snapshot_a = selection_a.plan_snapshot
        r_best_a = snapshot_a.r_best

        # Interleave a second select_action that would mutate the live tree.
        s_b, ctx_b = self._make_state_and_context(trainer)
        selection_b = trainer.select_action(s_b, context_latents=ctx_b)
        r_best_b = selection_b.plan_snapshot.r_best

        # The live mcts.root now corresponds to plan B. Pre-fix, an
        # observe_transition with no snapshot would read from there and
        # mis-attribute B's r_best to A's transition.
        s_next = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True,
        )
        metrics = trainer.observe_transition(
            s_a.unsqueeze(0) if s_a.dim() == 1 else s_a,
            selection_a.action, s_next,
            ctx={"plan_snapshot": snapshot_a},
        )

        # The transition is self-describing: observe reads from A's
        # snapshot, not from the live (B-shaped) tree.
        assert metrics["r_best"] == pytest.approx(r_best_a)
        # Sanity: if B and A differ (the usual case), the wrong-plan
        # mistake would have been observable. The test is meaningful
        # whenever r_best_a != r_best_b; if they happen to coincide,
        # the assertion above still locks the snapshot-read semantics.
        _ = r_best_b  # noted for the probe's framing

    def test_no_snapshot_with_live_plan_warns_once(self, trainer, caplog):
        """F4-residual (4.8 round-2 review): if the caller forgot to thread
        the snapshot but the trainer has a non-degenerate plan live, log
        a warning once. Catches the threading regression at first bite."""
        import logging
        s_t, ctx_latents = self._make_state_and_context(trainer)
        # Produce a non-degenerate plan.
        trainer.select_action(s_t, context_latents=ctx_latents)
        assert trainer.mcts.root is not None
        assert trainer.mcts.root.children  # plan is live

        s_next = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True,
        )

        with caplog.at_level(logging.WARNING, logger="luthi.v2.m9.runner"):
            # Forget to thread the snapshot.
            trainer.observe_transition(
                s_t.unsqueeze(0) if s_t.dim() == 1 else s_t,
                torch.zeros(D), s_next, ctx={},
            )
            # Second call -- gate should be tripped, no second warning.
            trainer.observe_transition(
                s_t.unsqueeze(0) if s_t.dim() == 1 else s_t,
                torch.zeros(D), s_next, ctx={},
            )

        warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "plan_snapshot" in r.getMessage()
        ]
        assert len(warns) == 1, (
            f"expected exactly one threading-regression warning; "
            f"got {len(warns)}"
        )

    def test_degenerate_plan_snapshot_falls_back_cleanly(self, trainer):
        """Budget=0 select_action returns a habit-fallback ActionSelection
        whose plan_snapshot is empty (no children). observe_transition
        treats it like the no-snapshot case (r_best=0 + habit sample)."""
        s_t, ctx_latents = self._make_state_and_context(trainer)
        selection = trainer.select_action(
            s_t, context_latents=ctx_latents, budget=0,
        )
        assert selection.plan_snapshot is not None
        assert selection.plan_snapshot.visit_distribution.numel() == 0

        s_next = encode_state(
            trainer.loss_module.online_encoder,
            text_tokens=torch.randint(0, VOCAB, (1, SEQ)), pool=True,
        )
        metrics = trainer.observe_transition(
            s_t.unsqueeze(0) if s_t.dim() == 1 else s_t,
            selection.action, s_next,
            ctx={"plan_snapshot": selection.plan_snapshot},
        )
        assert metrics["r_best"] == 0.0


# ---------------------------------------------------------------------------
# _cycle_observation_kwargs wiring
# ---------------------------------------------------------------------------


class TestCycleObservationKwargs:
    def test_returns_empty_when_no_ctx(self, trainer):
        assert trainer._cycle_observation_kwargs() == {}

    def test_promotes_scalar_counterpart_to_tensor(self, trainer):
        trainer._current_cycle_ctx = {"counterpart_present": True}
        kwargs = trainer._cycle_observation_kwargs()
        assert "counterpart_present" in kwargs
        assert isinstance(kwargs["counterpart_present"], torch.Tensor)
        assert kwargs["counterpart_present"].item() == 1.0

    def test_promotes_scalar_time_since_emission_to_tensor(self, trainer):
        trainer._current_cycle_ctx = {"time_since_emission": 7.5}
        kwargs = trainer._cycle_observation_kwargs()
        assert "time_since_emission" in kwargs
        assert isinstance(kwargs["time_since_emission"], torch.Tensor)
        assert kwargs["time_since_emission"].item() == 7.5

    def test_passes_tensor_inputs_through(self, trainer):
        cp = torch.tensor([1.0])
        ts = torch.tensor([4.2])
        trainer._current_cycle_ctx = {
            "counterpart_present": cp, "time_since_emission": ts,
        }
        kwargs = trainer._cycle_observation_kwargs()
        assert kwargs["counterpart_present"] is cp
        assert kwargs["time_since_emission"] is ts

    def test_ignores_extra_keys(self, trainer):
        trainer._current_cycle_ctx = {"counterpart_present": True, "foo": "bar"}
        kwargs = trainer._cycle_observation_kwargs()
        assert "foo" not in kwargs
