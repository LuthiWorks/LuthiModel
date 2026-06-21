"""Adversarial probes for the 2026-06-15 training-seam build (review F4 / F7a / F7b).

Run:  python -m redteam.seam.probe_seam_review

Convention mirrors redteam/m9_step1: a Verdict with confirmed=True means the
attack landed (the concern is real). These three were accepted by 4.7 in the
2026-06-15 review round; the empirical result decides whether the sketched
fixes are warranted.

  F4  -- observe_transition reads the LIVE mcts tree for r_best + habit target,
         so a transition is not self-contained: a select_action between the
         realized action and its observe_transition corrupts the reward.
  F7a -- model.encode() (the encode_state path) self-modifies the living
         weights, so computing s_t mutates the substrate as a side effect.
  F7b -- training s_t = context-fraction (0.8) encode; actor s_t = full encode.
         If they differ, the V-head bootstraps on out-of-distribution input.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import torch
import torch.optim as optim

from luthi.sanctuary_interface import encode_state
from luthi.seam_types import PlanSnapshot
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

from redteam.m9_step1._common import Verdict, report

VOCAB = 32
D = 32
SEQ = 16
B = 2
CONTEXT_FRACTION = 0.8  # JEPALoss default; the training-time context slice.


def _build_model(seed: int = 7) -> MultimodalPredictiveCodingLM:
    torch.manual_seed(seed)
    return MultimodalPredictiveCodingLM(
        vocab_size=VOCAB, d_model=D, n_blocks=2, n_heads=2,
        ffn_expansion=1, max_seq_len=SEQ,
        max_audio_tokens=SEQ, max_vision_tokens=SEQ,
        backward_pass_enabled=False,
    )


def _build_trainer(run_dir: Path, seed: int = 7) -> M9Trainer:
    model = _build_model(seed)
    loss_module = JEPALoss(online_encoder=model)
    optimizer = optim.AdamW(
        [p for p in loss_module.parameters() if p.requires_grad], lr=1e-3,
    )

    class _TextLoader:
        def __init__(self):
            self.gen = torch.Generator(device="cpu").manual_seed(seed)
            self._cursor = 0

        def next_batch(self, modality):
            t = torch.randint(0, VOCAB, (B, SEQ), generator=self.gen)
            self._cursor += 1
            return {"text_tokens": t}

        def batch_token_count(self, modality, batch):
            return int(batch["text_tokens"].numel())

        def state_dict(self):
            return {"cursor": self._cursor, "gen": self.gen.get_state()}

        def load_state_dict(self, s):
            self._cursor = s["cursor"]; self.gen.set_state(s["gen"])

        def corpus_sizes_tokens(self):
            return {"text": 1000}

    loader = _TextLoader()
    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens=loader.corpus_sizes_tokens(), alpha=0.7,
    )
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
        loss_module=loss_module, optimizer=optimizer,
        sampler=ModalitySampler(sampler_cfg),
        data_loader=loader, config=runner_cfg, run_dir=run_dir,
        m9_config=M9Config(mcts_budget_per_cycle=4),
    )


def _tree_r_best(trainer) -> float:
    """Replicate observe_transition's r_best read from the current tree."""
    root = trainer.mcts.root
    if root is None or not root.children:
        return 0.0
    gs = [c.incoming_g for c in root.children if c.incoming_g is not None]
    if not gs:
        return 0.0
    return float(-min(gs))


# ---------------------------------------------------------------------------
# F4 -- transition is not self-contained (live-tree coupling)
# ---------------------------------------------------------------------------


def probe_f4() -> list[Verdict]:
    verdicts: list[Verdict] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = _build_trainer(Path(tmp))
        model = trainer.loss_module.online_encoder
        try:
            # Re-tested against 4.7's 2026-06-15 PlanSnapshot fix. The fix makes
            # the transition self-describing: observe_transition reads r_best +
            # habit target from ctx["plan_snapshot"], frozen at select_action
            # time, NOT from live self.mcts.root. To prove the snapshot is
            # authoritative model-independently (untrained tiny model gives flat
            # ~0 EFE), thread a SENTINEL r_best that the live tree never carries.
            for _ in range(8):
                trainer.train_step("text", trainer.data_loader.next_batch("text"))

            torch.manual_seed(1)
            tokens_a = torch.randint(0, VOCAB, (1, SEQ))
            tokens_b = torch.randint(0, VOCAB, (1, SEQ))
            tokens_next = torch.randint(0, VOCAB, (1, SEQ))

            ctx_a = encode_state(model, text_tokens=tokens_a, pool=False)
            s_a = ctx_a.mean(dim=1)
            sel_a = trainer.select_action(s_a, context_latents=ctx_a)

            SENTINEL = 7.0  # a value the live tree's r_best (~0) never produces
            real_snap = sel_a.plan_snapshot
            sentinel_snap = PlanSnapshot(
                visit_distribution=real_snap.visit_distribution,
                candidate_actions=real_snap.candidate_actions,
                r_best=SENTINEL,
            )

            # Interleave a different plan -> resets live mcts.root (the old
            # corruption vector). With the fix, this must NOT affect the result.
            ctx_b = encode_state(model, text_tokens=tokens_b, pool=False)
            trainer.select_action(ctx_b.mean(dim=1), context_latents=ctx_b)
            live_r = _tree_r_best(trainer)  # B's live tree (what the old bug read)

            s_next = encode_state(model, text_tokens=tokens_next, pool=True)
            # (1) snapshot threaded: result must come from the snapshot.
            m_snap = trainer.observe_transition(
                s_a, sel_a.action, s_next, ctx={"plan_snapshot": sentinel_snap},
            )
            r_used = float(m_snap["r_best"])
            # (2) no snapshot (my stale probe's path): documents the degenerate
            #     r=0 that produced the earlier false-positive.
            m_nosnap = trainer.observe_transition(
                s_a, sel_a.action, s_next, ctx={},
            )
            r_nosnap = float(m_nosnap["r_best"])

            defended = abs(r_used - SENTINEL) < 1e-6 and abs(r_used - live_r) > 1e-6
            confirmed = not defended  # attack lands only if the snapshot is ignored
            detail = (
                f"threaded snapshot r_best={SENTINEL}; observe_transition used "
                f"r_best={r_used:.6f} (live interleaved tree r_best={live_r:.6f}); "
                f"no-snapshot path r_best={r_nosnap:.6f}. "
                + (
                    "Snapshot is authoritative -- an interleaved select_action "
                    "cannot corrupt the transition. F4 FIXED. (My earlier "
                    "CONFIRMED was a stale-probe artifact: empty ctx -> the "
                    "degenerate r=0 path, and used_b passed vacuously on 0<=0.)"
                    if defended else
                    "Snapshot ignored; live tree leaked into the transition."
                )
            )
            verdicts.append(Verdict(
                "F4: interleaved select_action corrupts the realized transition "
                "(re-tested vs the 2026-06-15 PlanSnapshot fix)",
                confirmed, detail,
            ))
        except Exception:
            verdicts.append(Verdict(
                "F4: probe raised", False,
                "ERROR:\n" + traceback.format_exc(),
            ))
    return verdicts


# ---------------------------------------------------------------------------
# F7a -- encode() mutates the living substrate
# ---------------------------------------------------------------------------


def probe_f7a() -> list[Verdict]:
    verdicts: list[Verdict] = []
    model = _build_model()
    try:
        # Snapshot every living-ffn buffer across all blocks.
        before = {}
        for bi, block in enumerate(model.blocks):
            ffn = getattr(block, "living_ffn", None)
            if ffn is None:
                continue
            for name, buf in ffn.named_buffers():
                before[(bi, name)] = buf.detach().clone()

        tokens = torch.randint(0, VOCAB, (1, SEQ))
        _ = encode_state(model, text_tokens=tokens, pool=True)

        changed = []
        max_delta = 0.0
        for bi, block in enumerate(model.blocks):
            ffn = getattr(block, "living_ffn", None)
            if ffn is None:
                continue
            for name, buf in ffn.named_buffers():
                key = (bi, name)
                if key not in before:
                    continue
                d = float((buf.detach() - before[key]).abs().max().item())
                if d > 0.0:
                    changed.append((f"block{bi}.{name}", d))
                    max_delta = max(max_delta, d)

        confirmed = len(changed) > 0
        top = sorted(changed, key=lambda x: -x[1])[:6]
        detail = (
            f"{len(changed)} living buffer(s) changed during a single "
            f"encode_state() call; max |delta| = {max_delta:.3e}. "
            f"top: {top}. "
            + (
                "Computing s_t mutates the substrate as a side effect."
                if confirmed else
                "encode_state() left the living buffers untouched."
            )
        )
        verdicts.append(Verdict(
            "F7a: encode_state()/model.encode() self-modifies the living weights",
            confirmed, detail,
        ))
    except Exception:
        verdicts.append(Verdict(
            "F7a: probe raised", False,
            "ERROR:\n" + traceback.format_exc(),
        ))
    return verdicts


# ---------------------------------------------------------------------------
# F7a follow-up -- does encode()'s self-modification perturb generation?
# ---------------------------------------------------------------------------


def _snapshot_living(model) -> dict:
    snap = {}
    for bi, block in enumerate(model.blocks):
        ffn = getattr(block, "living_ffn", None)
        if ffn is None:
            continue
        for name, buf in ffn.named_buffers():
            snap[(bi, name)] = buf.detach().clone()
    return snap


def _restore_living(model, snap) -> None:
    for bi, block in enumerate(model.blocks):
        ffn = getattr(block, "living_ffn", None)
        if ffn is None:
            continue
        for name, buf in ffn.named_buffers():
            if (bi, name) in snap:
                buf.copy_(snap[(bi, name)])


def probe_f7a_interference() -> list[Verdict]:
    """4.7's requested follow-up: encode_state mutates the substrate (F7a
    confirmed) -- but does that mutation measurably change what the
    *subsequent generation pass* produces? The seam runs encode_state
    BEFORE _generate_inner_speech, so generation starts from an
    encode-perturbed state. Compare a fixed eval forward from a clean
    state vs. the same forward after an intervening encode_state.
    """
    verdicts: list[Verdict] = []
    model = _build_model()
    try:
        torch.manual_seed(11)
        x_eval = torch.randint(0, VOCAB, (1, SEQ))
        prompt = torch.randint(0, VOCAB, (1, SEQ))

        s0 = _snapshot_living(model)

        _restore_living(model, s0)
        with torch.no_grad():
            base_logits = model(x_eval).detach().clone()

        _restore_living(model, s0)
        with torch.no_grad():
            encode_state(model, text_tokens=prompt, pool=True)  # mutates S0 -> S1
            after_logits = model(x_eval).detach().clone()

        rel = float(
            (after_logits - base_logits).norm().item()
            / (base_logits.norm().item() + 1e-9)
        )
        base_tok = int(base_logits[0, -1].argmax().item())
        after_tok = int(after_logits[0, -1].argmax().item())
        tok_changed = base_tok != after_tok

        confirmed = bool(rel > 0.05 or tok_changed)
        detail = (
            f"fixed eval forward, clean state vs after an intervening "
            f"encode_state: rel L2 logit delta = {rel:.3f}; last-token argmax "
            f"{'CHANGED' if tok_changed else 'unchanged'} "
            f"({base_tok} -> {after_tok}). "
            + (
                "encode's self-modification measurably perturbs the subsequent "
                "generation forward -- the double-plasticity is not free; the "
                "perception path should run under a no-self-modify guard or the "
                "interference accepted explicitly."
                if confirmed else
                "encode's mutation does not measurably move generation -- the "
                "'encode is the perception path' framing is load-bearing-but-fine."
            )
        )
        verdicts.append(Verdict(
            "F7a-followup: encode_state() self-modification interferes with the "
            "subsequent generation pass",
            confirmed, detail,
        ))
    except Exception:
        verdicts.append(Verdict(
            "F7a-followup: probe raised", False,
            "ERROR:\n" + traceback.format_exc(),
        ))
    return verdicts


# ---------------------------------------------------------------------------
# F7b -- train s_t (context-fraction) vs actor s_t (full encode) distribution
# ---------------------------------------------------------------------------


def probe_f7b() -> list[Verdict]:
    verdicts: list[Verdict] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        trainer = _build_trainer(Path(tmp))
        model = trainer.loss_module.online_encoder
        v_head = trainer.v_head
        try:
            torch.manual_seed(3)
            n = 64
            ctx_len = max(1, int(round(SEQ * CONTEXT_FRACTION)))
            train_pool, actor_pool = [], []
            paired_dist = []
            with torch.no_grad():
                for _ in range(n):
                    tokens = torch.randint(0, VOCAB, (1, SEQ))
                    # training-time s_t: encode the CONTEXT SLICE only, mean-pool
                    # (mirrors raw["online_context_latents"].mean(1)).
                    s_train = model.encode(
                        text_tokens=tokens[:, :ctx_len], causal=False,
                    )["latents"].mean(dim=1)
                    # actor-time s_t: encode_state over the FULL sequence.
                    s_actor = encode_state(model, text_tokens=tokens, pool=True)
                    train_pool.append(s_train)
                    actor_pool.append(s_actor)
                    paired_dist.append(
                        float((s_train - s_actor).norm().item())
                    )

                train = torch.cat(train_pool, dim=0)   # [n, D]
                actor = torch.cat(actor_pool, dim=0)
                # Per-distribution scale and the gap between the two.
                train_norm = float(train.norm(dim=1).mean().item())
                actor_norm = float(actor.norm(dim=1).mean().item())
                mean_pair_dist = sum(paired_dist) / len(paired_dist)
                rel_gap = mean_pair_dist / (0.5 * (train_norm + actor_norm) + 1e-9)
                # Per-dim mean shift between the two pools.
                mean_shift = float(
                    (train.mean(0) - actor.mean(0)).norm().item()
                )
                # The bite: does the V-head respond differently?
                v_train = v_head(train)
                v_actor = v_head(actor)
                v_train_m, v_train_s = float(v_train.mean()), float(v_train.std())
                v_actor_m, v_actor_s = float(v_actor.mean()), float(v_actor.std())
                v_shift = abs(v_train_m - v_actor_m)
                v_scale = 0.5 * (abs(v_train_m) + abs(v_actor_m)) + 1e-9
                v_rel = v_shift / v_scale

            # Confirmed if the two s_t pools are materially apart in input
            # space (rel gap) AND that propagates to a material V-head shift.
            confirmed = bool(rel_gap > 0.10 and v_rel > 0.10)
            detail = (
                f"paired ||s_train - s_actor|| mean = {mean_pair_dist:.4f} "
                f"(train |s|={train_norm:.4f}, actor |s|={actor_norm:.4f}, "
                f"rel gap = {rel_gap:.3f}); per-dim mean shift = {mean_shift:.4f}. "
                f"V-head: train mean={v_train_m:.4f}(sd {v_train_s:.4f}) vs "
                f"actor mean={v_actor_m:.4f}(sd {v_actor_s:.4f}); "
                f"rel V shift = {v_rel:.3f}. "
                + (
                    "Actor s_t is materially off the training distribution and "
                    "the V-head sees it -- bootstrap is OOD."
                    if confirmed else
                    "Distributions are close enough that the V-head response is "
                    "comparable (SIGReg likely keeping the marginals aligned)."
                )
            )
            verdicts.append(Verdict(
                "F7b: actor s_t (full encode) is OOD vs training s_t "
                "(context-fraction encode)",
                confirmed, detail,
            ))
        except Exception:
            verdicts.append(Verdict(
                "F7b: probe raised", False,
                "ERROR:\n" + traceback.format_exc(),
            ))
    return verdicts


def main() -> int:
    all_v = []
    all_v += report("PROBE F4: transition not self-contained (vs snapshot fix)", probe_f4())
    all_v += report("PROBE F7a: encode mutates substrate", probe_f7a())
    all_v += report("PROBE F7a-followup: encode interferes with generation", probe_f7a_interference())
    all_v += report("PROBE F7b: s_t train/actor distribution gap", probe_f7b())
    confirmed = sum(1 for v in all_v if v.confirmed)
    print("\n" + "=" * 62)
    print(f"SEAM REVIEW PROBES: {confirmed}/{len(all_v)} concerns CONFIRMED")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
