"""Two-arm JEPA pilot driver — Experiment 1, JEPA edition (2026-07-15).

The falsification-critical run, merged with the M8 256d de-risking pilot
(critical-path item 1). One instrumented sweep answers three
pre-registered questions (docs/research/2026-07-15_falsification-
preregistration.md; protocol living-weights-experiments.md, JEPA
edition):

  (a) the pilot's collapse-kill thresholds (pilot-set machinery derives
      them per-run; this sweep sanity-checks the derived values);
  (b) does representation collapse behave differently when the weights
      self-modify — the dead arm is the direct control;
  (c) matched capacity: does the living arm sit above the dead arm's
      effective-capacity curve on held-out latent prediction + probes?

Arms and stages (5 seeds per condition — Brian, 2026-07-15):

  stage 1   living@256 x5  +  dead@256 x5     the matched point
  stage 2   dead@192 x5  +  dead@384 x5       the curve's shape
  stage 3   dead@512 x5                       the upper bracket

Stage 1 decides half the outcomes alone: if the living arm loses or
ties at the matched point, KF2-strong dies and stages 2-3 are
unnecessary. Bracket {192,256,384,512} pends Brian's S1 ratification.

Per run: text-only JEPA (round-1 scope), leakage-gapped 2% holdout,
end-of-epoch held-out eval (built into JEPATrainer), then a final
held-out eval + next-token linear probe WITH its shuffled-label floor.
Results land in <run_dir>/pilot_result.json — the completion marker, so
the sweep is resumable (completed runs skipped; a failed run stops the
queue loudly: a condition with fewer seeds than its siblings corrupts
the variance estimate).

Device: DirectML -> CUDA -> CPU (the train_pc pick). NOTE: the JEPA path
has only ever run on CPU (the smokes); the first pilot run doubles as
the device shakeout — watch the first hundred steps.

Usage:
  python scripts/jepa_pilot_driver.py --stage 1 --dry-run
  python scripts/jepa_pilot_driver.py --stage 1
  python scripts/jepa_pilot_driver.py --aggregate
Smoke (CPU, minutes):
  python scripts/jepa_pilot_driver.py --stage 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
# The driver imports luthi in-process (unlike the retired LM driver,
# which shelled out); make it runnable from anywhere.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUT_ROOT = REPO_ROOT / "runs" / "jepa_pilot"

SEEDS = (42, 43, 44, 45, 46)

# (arm, d_model) conditions per stage. Bracket amended by Brian
# 2026-07-16: a SINGLE overshoot point (dead@512, ~4x the living FFN's
# nominal weight count) replaces the {192, 384, 512} curve; stage 3 is
# reserved for the 384 fallback the pre-registered read requires if
# dead@512 wins or ties (see the pre-registration's bracket entry).
STAGES: dict[int, list[tuple[str, int]]] = {
    1: [("living", 256), ("dead", 256)],
    2: [("dead", 512)],
    3: [("dead", 384)],  # fallback only -- run on an inconclusive stage 2
    # Run 2 of the staged-configuration ladder (Brian, 2026-07-17): the
    # FULL living configuration -- backward pass ON (DNR 9b: the
    # task-salience -> plasticity channel) + consolidation ON (episodes
    # become structure). Stage-1's "living" arm ran the MINIMAL config
    # (both off, inherited from smoke defaults); the ladder turns
    # subsystems on stepwise so improvements stay attributable. Held for
    # run 3: plasticity taper, inverted-U gain, recall-gate tightening.
    4: [("living_full", 256)],
    # Run 3 (Brian, 2026-07-17 evening): the three held builds land --
    # plasticity taper (formative->mature, floor 0.2 from 50% progress),
    # inverted-U gain, recall gate 0.5 -> 0.7 -- AND the living arm moves
    # to 512 alongside the existing dead@512 control (no dead rerun: all
    # three changes are living-side only). Attribution caveat, recorded:
    # width and the three builds move together in this rung by Brian's
    # ruling; a bridge arm (living_full@512, no builds) can be run later
    # if attribution needs splitting.
    5: [("living_v3", 512)],
    # The BRIDGE arm (Brian, 2026-07-18): living_full's exact config at
    # 512 -- no taper, no gain, recall gate 0.5. Fills the attribution
    # cell run 3 confounded: vs living_v3@512 isolates the three builds
    # at fixed width; vs living_full@256 isolates width at fixed config.
    # Tracking arm -- no verdict force; reads frozen in the registry.
    6: [("living_full", 512)],
    # Run 5, the DATA-SCALING cells (Brian, 2026-07-18: "increase the
    # data by 4x" -- the param rule, data ~ width^2). Same 512 arms on a
    # 4x-token SUPERSET corpus (the 1x corpus + 382 more books, ~50.4M
    # tokens). Stage 7a = dead (the pure starvation test: does the dead
    # NMSE curve's worsening-with-width reverse with data?); stage 7b =
    # living_v3 (does the living advantage's magnitude recover?).
    7: [("dead_4x", 512)],
    8: [("living_v3_4x", 512)],
    # Run 6, the DEPTH rung -- AMENDED 2026-07-20 (Brian): the depth
    # family now BUNDLES the two cheap levers with the depth increase
    # (2 -> 4 blocks): cosine LR decay (the registered cosine rung,
    # folded in) and 2x SIGReg weight (the variance-floor lever from the
    # seed44 denominator-race finding). One-variable attribution vs the
    # d2 anchors is deliberately traded for speed; if the bundle moves
    # the picture, single-lever follow-ups split it (bridge precedent).
    # First JEPA-era outing of the muPC depth machinery (exponent 0.25).
    # The depth-only registration is superseded -- see the 2026-07-20
    # amendment in the falsification pre-registration doc.
    9: [("living_v4_4x_d4", 512)],
    # RUN v5 (Brian's 2026-07-21 resequencing): precision awakening as
    # its own family -- v4 + relative_trust, single variable. The
    # dormant-machinery bundle (sparse/iPC/attractor/lambda) is v6.
    10: [("living_v5_4x_d4", 512)],
    # Seed44 ROBUSTNESS RERUN (Brian's 2026-07-26 ruling): identical
    # config and data order as stage 10's seed44; GPU float
    # nondeterminism supplies the only perturbation. Distinguishes a
    # robust trigger (the ~58650 trust event recurs) from a knife-edge
    # one (it does not). Distinct arm name so run dirs and verdict
    # filters keep it out of the registered v5 family (never-pool).
    # Unregistered descriptive probe -- no frozen prediction. Replaces
    # the dead_4x_d4 control in the SCHEDULE only; that control remains
    # a registered obligation (deferred; no depth claims until it runs).
    11: [("living_v5_4x_d4_rerun", 512)],
    # VALIDATION PROBE (2026-07-27, not a registered family): v5's exact
    # configuration plus the episode-store admission fix and the homeostatic
    # band. Short run, one seed. Its only purpose is to check the four
    # predictions in docs/research/2026-07-27_episode-store-frozen-defect.md
    # against real training before v6 commits to a multi-day family -- the
    # fix is unit-tested but has never run at scale, and the defect it
    # repairs was invisible to every counter for five whole families.
    # Distinct arm name so its artifacts can never pool with a family.
    12: [("probe_storefix", 512)],
    # SURPRISE-DRIVE PROBE (2026-07-29, not a registered family): stage 12's
    # configuration plus drive_mode="surprise". Same purpose and same
    # discipline -- the drive fix is unit-tested (20 tests) but has never run on
    # real corpus data, and the thing it repairs (a self-extinguishing drive)
    # was invisible for five families because "quiet because familiar" and
    # "quiet because broken" were not separable. Read drive_duty first.
    13: [("probe_surprise", 512)],
    # SURPRISE DRIVE AT DEPTH 8 (2026-07-29, Brian's call; also the depth v6
    # was registered to start at). Full length -- the extinction question is
    # what this run exists to answer and 4,000 steps provably cannot.
    14: [("probe_surprise_d8", 512)],
    # SIGREG PROJECTION TEST (2026-07-30): stage 14 with sigreg_projection
    # "none" instead of "linear". One variable. Readout is offset dominance,
    # NOT capability -- the carried-over clip independently kills capability.
    15: [("probe_surprise_d8_noproj", 512)],
    # muPC TEST (2026-07-30): stage 14 with mu_pc_enabled=False. One variable.
    # Primary readout is within-batch pairwise cosine measured post-hoc, NOT
    # capability -- the carried-over clip independently kills capability.
    16: [("probe_surprise_d8_nomupc", 512)],
}

# Per-arm model configuration -- single source of truth, shared with
# pilot_verdict.py's checkpoint-rebuild path (the eval must reconstruct
# the arm EXACTLY, including the recall gate, which is live at eval).
ARM_CONFIGS: dict[str, dict] = {
    "dead":        {"dead_ffn": True},
    "living":      {},  # minimal: smoke defaults (BP off, consolidation off)
    "living_full": {"backward_pass_enabled": True,
                    "consolidation_enabled": True},
    "living_v3":   {"backward_pass_enabled": True,
                    "consolidation_enabled": True,
                    "learning_gain_enabled": True,
                    "episode_recall_threshold": 0.7},
}
# Trainer-side (non-model) per-arm settings.
ARM_TAPER: dict[str, bool] = {"living_v3": True, "living_v3_4x": True}

# Data-scaled arms: model config identical to their base arm; the corpus
# is the 4x superset filelist. Registered as distinct arm names so run
# dirs, verdict filters, and the never-pool rules all keep them separate.
ARM_CONFIGS["dead_4x"] = dict(ARM_CONFIGS["dead"])
ARM_CONFIGS["living_v3_4x"] = dict(ARM_CONFIGS["living_v3"])
# The depth arm (run 6): v3's living config at 4 blocks with muPC depth
# scaling enabled -- its first outing under JEPA. n_blocks/mu_pc_* are
# model kwargs, so they ride in ARM_CONFIGS and override the driver's
# --n-blocks default via the arm-config merge in _run_one.
ARM_CONFIGS["living_v3_4x_d4"] = dict(
    ARM_CONFIGS["living_v3"],
    n_blocks=4,
    mu_pc_enabled=True,
    mu_pc_exponent=0.25,
)
ARM_TAPER["living_v3_4x_d4"] = True
# v4 (Brian's 2026-07-20 bundle ruling): the depth arm as actually run --
# same MODEL config as living_v3_4x_d4 (the sigreg weight and LR schedule
# are trainer/loss-side, not model kwargs; see ARM_SIGREG / ARM_COSINE).
# Registered as a distinct arm name so run dirs, verdict filters, and the
# never-pool rules keep it separate from any pure-depth run.
ARM_CONFIGS["living_v4_4x_d4"] = dict(ARM_CONFIGS["living_v3_4x_d4"])
ARM_TAPER["living_v4_4x_d4"] = True
# v5 (Brian's 2026-07-21 resequencing): the precision awakening as its
# own family -- v4's exact config + relative trust (ratio-to-median
# weighting, numerics-only eps, freed ledger). ONE change vs v4; the
# dormant-machinery bundle became v6. Registered in the pre-reg doc's
# corrected precision entries.
ARM_CONFIGS["living_v5_4x_d4"] = dict(
    ARM_CONFIGS["living_v4_4x_d4"], relative_trust=True,
)
ARM_TAPER["living_v5_4x_d4"] = True
ARM_FILELIST: dict[str, str] = {
    "dead_4x": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v3_4x": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v3_4x_d4": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v4_4x_d4": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
    "living_v5_4x_d4": str(REPO_ROOT / "corpus_build" / "gutenberg_4x_filelist.txt"),
}
# Loss-side per-arm setting: SIGReg weight. v4 doubles the default 0.1 --
# the variance-floor lever. Rationale (2026-07-20): living spaces settle
# at per-dim std ~0.3 against SIGReg's implicit unit target, and the
# seed44 denominator race showed over-quieting destabilizes NMSE across
# seeds. A stronger pull toward the isotropic target should hold the
# space louder; registered prediction in the pre-reg amendment.
ARM_SIGREG: dict[str, float] = {"living_v4_4x_d4": 0.2, "living_v5_4x_d4": 0.2}
# Trainer-side per-arm setting: cosine LR decay to a 10% floor (the
# registered cosine rung, folded into v4 by the same ruling).
ARM_COSINE: dict[str, bool] = {"living_v4_4x_d4": True, "living_v5_4x_d4": True}
# Rerun alias (stage 11): byte-identical configuration to the v5 arm
# under a distinct name, so the rerun's artifacts can never pool with
# the registered family by accident.
ARM_CONFIGS["living_v5_4x_d4_rerun"] = dict(ARM_CONFIGS["living_v5_4x_d4"])
ARM_TAPER["living_v5_4x_d4_rerun"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["living_v5_4x_d4_rerun"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["living_v5_4x_d4_rerun"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["living_v5_4x_d4_rerun"] = ARM_COSINE["living_v5_4x_d4"]
# Validation probe: v5 + the 2026-07-27 store fix + the homeostatic band.
# Identical to v5 in every other respect, so any change in store behaviour is
# attributable to the two flags and comparable to the measured v5 baselines.
ARM_CONFIGS["probe_storefix"] = dict(
    ARM_CONFIGS["living_v5_4x_d4"],
    adaptive_episodes=True,
    adaptive_recall=True,
    homeostatic_band_enabled=True,
)
ARM_TAPER["probe_storefix"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_storefix"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["probe_storefix"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["probe_storefix"] = ARM_COSINE["living_v5_4x_d4"]
# Surprise-drive probe (2026-07-29). probe_storefix + drive_mode="surprise".
# ONE change against probe_storefix, so the drive is attributable: the store
# fix, the band, and the fixed objective are all held constant and already
# measured (probe_storefix_512d_seed45: NMSE 0.597, probe lift 4.67x over its
# own shuffled floor, update_ema 2.6e-6..8.3e-6).
#
# `relative_trust` is already True here via the v5 base, which is what makes
# surprise mode legal at all -- absolute precision weighting saturates the
# +/-1 clamp at 100% and discards the drive magnitude (measured on the real
# code path). That dependency is enforced by a raise in the layer, not
# supplied silently, so this pairing is declared here where it can be
# attributed rather than inherited from a default.
#
# What to read first in the log: substrate_blocks[*].drive_duty. It is 0.0000
# on stationary input and on i.i.d. draws from a fixed distribution, and rises
# only at a genuine shift in the error scale (unit-tested). So on real corpus
# data a nonzero duty means the drive is finding structure the forecast did not
# expect, and a flat zero means it is not -- the discriminator between "quiet
# because familiar" and "quiet because broken" that the first five families
# could not make.
ARM_CONFIGS["probe_surprise"] = dict(
    ARM_CONFIGS["probe_storefix"],
    drive_mode="surprise",
)
ARM_TAPER["probe_surprise"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_surprise"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["probe_surprise"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["probe_surprise"] = ARM_COSINE["living_v5_4x_d4"]
# The 8-block depth arm (Brian's call 2026-07-29, and the depth v6 was already
# registered to start at): probe_surprise at n_blocks=8. Everything else held
# identical to the 4-block probe so depth is the only difference against it.
#
# muPC note: mu_pc_exponent stays at 0.25, inherited from the d4 arms. muPC
# exists precisely so depth does not need per-depth rate retuning, so 0.25 is
# the right thing to carry forward rather than re-tune by hand -- but this is
# the first time that claim gets tested at 8 blocks in this project, and it is
# a claim, not a guarantee. If the deep blocks show a systematically different
# firing rate or a rate/stability problem the shallow ones do not, the exponent
# is the first suspect.
ARM_CONFIGS["probe_surprise_d8"] = dict(
    ARM_CONFIGS["probe_surprise"],
    n_blocks=8,
)
# Trainer-side (non-model) per-arm gradient clipping. Depth 8 diverged at step
# ~2250 without it: grad_norm median 1065 vs 28.4 at depth 4, ~37x larger, at an
# unchanged 3e-4 learning rate, and there was no clipping in the runner at all.
#
# Clip value 1000 is chosen against the measured distribution rather than the
# conventional 1.0 -- gradients here are O(1e3) because the loss itself is
# O(1e2) (SIGReg contributes hundreds), so clipping at 1.0 would reduce the
# effective step ~1000x and stall learning. 1000 sits at roughly the depth-8
# median, so typical steps pass close to untouched while the excursions that
# preceded divergence (2158, 2941, 5555, 8645) are bounded. Deliberately
# aggressive for a stability probe: over-damping shows up as poor learning,
# which is a diagnosable failure, where divergence is not.
#
# Depth-4 arms keep grad_clip_norm=0.0 (off), so every completed family stays
# bit-identical and comparable.
ARM_GRAD_CLIP: dict[str, float] = {
    "probe_surprise_d8": 1000.0,
    "probe_surprise_d8_noproj": 1000.0,
}
# Loss-side per-arm setting: the SIGReg projection head.
#
# `probe_surprise_d8_noproj` is `probe_surprise_d8` with sigreg_projection
# switched from "linear" to "none", and NOTHING else changed -- same 8 blocks,
# same clip of 1000, same seed discipline. It tests one hypothesis: that the
# per-modality nn.Linear head's BIAS absorbs the batch-mean offset, presenting
# SIGReg with centered latents while the trunk keeps the offset. That would be
# structurally the same defect as the BatchNorm removed on 2026-07-28 -- a
# learnable layer standing between SIGReg and the quantity it exists to
# constrain. "none" is nn.Identity(), so SIGReg sees trunk latents directly.
#
# The clip of 1000 is deliberately CARRIED OVER rather than fixed first, so
# this run differs from probe_surprise_d8_512d_seed96 by exactly one thing. It
# is known to be too aggressive (43% of steps clipped) and to kill capability
# on its own -- see the pre-registered readout note in
# docs/research/2026-07-30_sigreg-projection-hypothesis.md for why capability
# metrics therefore CANNOT be read from this run either way.
ARM_SIGREG_PROJ: dict[str, str] = {"probe_surprise_d8_noproj": "none"}
# `probe_surprise_d8_nomupc` is `probe_surprise_d8` with mu_pc_enabled False and
# NOTHING else changed -- same 8 blocks, same clip of 1000, same "linear"
# projection (the "none" variant was refuted 2026-07-30 and made prediction
# 7.3x worse, so it is not carried forward).
#
# Disabling muPC changes two things at once, deliberately: residual_scale goes
# 1/(8**0.25) = 0.5946 -> 1.0, and the depth-scaled init is skipped. They are
# tested together because a clean result implicates muPC as a whole and a null
# result exonerates it as a whole; separating them costs a second run and only
# matters if the first one is positive. Init has already been shown to wash out
# by step 3000 (block-0 q_proj std 0.0322 at d4 vs 0.0325 at d8), so the
# residual scale is the live half.
ARM_CONFIGS["probe_surprise_d8_nomupc"] = dict(
    ARM_CONFIGS["probe_surprise_d8"],
    mu_pc_enabled=False,
)
ARM_GRAD_CLIP["probe_surprise_d8_nomupc"] = 1000.0
ARM_TAPER["probe_surprise_d8_nomupc"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_surprise_d8_nomupc"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["probe_surprise_d8_nomupc"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["probe_surprise_d8_nomupc"] = ARM_COSINE["living_v5_4x_d4"]
ARM_CONFIGS["probe_surprise_d8_noproj"] = dict(ARM_CONFIGS["probe_surprise_d8"])
ARM_TAPER["probe_surprise_d8_noproj"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_surprise_d8_noproj"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["probe_surprise_d8_noproj"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["probe_surprise_d8_noproj"] = ARM_COSINE["living_v5_4x_d4"]
ARM_TAPER["probe_surprise_d8"] = ARM_TAPER["living_v5_4x_d4"]
ARM_FILELIST["probe_surprise_d8"] = ARM_FILELIST["living_v5_4x_d4"]
ARM_SIGREG["probe_surprise_d8"] = ARM_SIGREG["living_v5_4x_d4"]
ARM_COSINE["probe_surprise_d8"] = ARM_COSINE["living_v5_4x_d4"]


def _device() -> torch.device:
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _DeviceLoader:
    """Wraps a MultimodalDataLoader; delivers device-local batches."""

    def __init__(self, inner, device: torch.device):
        self._inner = inner
        self._device = device

    def _move(self, batch: dict) -> dict:
        return {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def next_batch(self, modality: str) -> dict:
        return self._move(self._inner.next_batch(modality))

    def batch_token_count(self, modality: str, batch: dict) -> int:
        return self._inner.batch_token_count(modality, batch)

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._inner.load_state_dict(state)

    def holdout_batch_count(self, modality: str, batch_size: int) -> int:
        return self._inner.holdout_batch_count(modality, batch_size)

    def holdout_batches(self, modality: str, batch_size: int):
        for batch in self._inner.holdout_batches(modality, batch_size):
            yield self._move(batch)


def _run_name(arm: str, d_model: int, seed: int) -> str:
    return f"{arm}_{d_model}d_seed{seed}"


def _result_path(arm: str, d_model: int, seed: int) -> Path:
    return OUTPUT_ROOT / _run_name(arm, d_model, seed) / "pilot_result.json"


def _run_one(arm: str, d_model: int, seed: int, args) -> dict:
    from luthi.tokenizer import BPETokenizer
    from luthi.v2.eval_heldout import (
        fit_next_token_probe,
        heldout_latent_prediction,
        probe_accuracy,
    )
    from luthi.v2.jepa_loss import JEPALoss, SIGREG_LAMBD
    from luthi.v2.jepa_runner import (
        CheckpointConfig,
        EpochConfig,
        JEPATrainer,
        KillCriteriaConfig,
        LoggingConfig,
        LRScheduleConfig,
        ModalitySampler,
        RunnerConfig,
        SamplerConfig,
        TaperConfig,
    )
    from luthi.v2.multimodal_data import (
        MultimodalDataLoaderImpl,
        TextDataset,
        TextDatasetConfig,
    )
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    device = _device()
    run_dir = OUTPUT_ROOT / _run_name(arm, d_model, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Data source: per-arm filelist (the 4x superset) when registered,
    # else the default 1x directory. The 4x corpus's holdout is its own
    # tail (a DIFFERENT test set than the 1x tail) -- cross-corpus
    # comparisons are directional only; the registry carries the caveat.
    if arm in ARM_FILELIST:
        filelist = Path(ARM_FILELIST[arm])
        source_paths = [
            line.strip() for line in
            filelist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        source_paths = [args.data_dir]
    text_ds = TextDataset(TextDatasetConfig(
        source_paths=source_paths,
        tokenizer_path=Path(args.tokenizer),
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        base_seed=seed,
        holdout_fraction=args.holdout_fraction,
    ))
    loader = _DeviceLoader(MultimodalDataLoaderImpl(text=text_ds), device)

    # Model kwargs: driver defaults, overridden by the arm's declared
    # config (ARM_CONFIGS is the single source of truth, shared with the
    # verdict rebuild). The depth arm carries its own n_blocks/mu_pc_*
    # through this merge; arms without a key keep the smoke default.
    model_kwargs = dict(
        vocab_size=text_ds.vocab_size(),
        d_model=d_model,
        n_blocks=args.n_blocks,
        n_heads=4,
        ffn_expansion=1,
        max_seq_len=args.seq_len,
        backward_pass_enabled=False,
    )
    model_kwargs.update(ARM_CONFIGS[arm])
    # Silent-fallback guard (2026-07-21): the C++ extension rebuilds
    # between families (the running trainer locks the .pyd, so source
    # changes compile at the NEXT process start). If that rebuild fails,
    # pc_ops silently falls back to pure Python -- ~50x slower, which
    # would turn a 9h seed into weeks, unattended. Refuse loudly instead.
    if model_kwargs.get("relative_trust"):
        from luthi.v2.pc_ops import is_cpp_available
        if not is_cpp_available():
            raise SystemExit(
                f"[jepa-pilot] {arm}: relative_trust requires the C++ "
                "pc_ops extension, which failed to load in this process. "
                "Fix the build (python -c \"from luthi.v2 import pc_ops; "
                "print(pc_ops.is_cpp_available())\") before running this arm."
            )
    model = MultimodalPredictiveCodingLM(**model_kwargs).to(device)
    loss_module = JEPALoss(
        online_encoder=model,
        sigreg_lambd=ARM_SIGREG.get(arm, SIGREG_LAMBD),
        sigreg_projection=ARM_SIGREG_PROJ.get(arm, "linear"),
    ).to(device)

    # Cosine LR needs the planned run length. tokens_per_pass is
    # n_sequences * seq_len (exact), so steps/epoch falls out directly;
    # epoch-boundary partial batches can shift the true total by a step
    # or two, which cosine absorbs (progress past 1.0 holds the floor).
    steps_per_epoch = math.ceil(
        text_ds.tokens_per_pass() / args.seq_len / args.batch_size
    )
    lr_schedule = LRScheduleConfig(
        enabled=ARM_COSINE.get(arm, False),
        min_lr_ratio=0.1,
        total_steps=args.epochs * steps_per_epoch,
    )

    sampler_cfg = SamplerConfig(
        corpus_sizes_tokens={"text": text_ds.tokens_per_pass()}, alpha=0.7,
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    trainer = JEPATrainer(
        loss_module=loss_module,
        optimizer=optim.AdamW(
            [p for p in loss_module.parameters() if p.requires_grad],
            lr=args.lr,
        ),
        sampler=ModalitySampler(sampler_cfg, generator=gen),
        data_loader=loader,
        config=RunnerConfig(
            sampler=sampler_cfg,
            checkpoint=CheckpointConfig(rolling_slots=3),
            logging=LoggingConfig(heldout_eval_batches=args.heldout_batches),
            kill_criteria=KillCriteriaConfig(
                warmup_batches=args.kill_warmup,
                # Pilot-derived thresholds (calibration pass 1, 2026-07-16
                # -- docs/research/2026-07-16_jepa-pilot-calibration-pass.md).
                # The static defaults killed 10/10 healthy runs: kill-1's
                # init-window baseline fired while effective rank was
                # RISING; kill-6's 25% band around a transiently-latched
                # running min fired at the substrate's healthiest moment.
                stationary_deviation_pct=0.85,
                substrate_health_degradation_pct=1.0,
                trending_smoothing_window=9,
                substrate_health_window=10,
            ),
            epoch=EpochConfig(
                max_epochs=args.epochs,
                abort_continue_at_epoch_1=False,  # sweep runs are unattended
                max_batches_per_epoch=args.max_batches_per_epoch,
            ),
            taper=TaperConfig(enabled=ARM_TAPER.get(arm, False)),
            lr_schedule=lr_schedule,
            grad_clip_norm=ARM_GRAD_CLIP.get(arm, 0.0),
        ),
        run_dir=run_dir,
    )

    started = time.time()
    # Mid-seed resilience (2026-07-20, Brian's terminal-close question):
    # an interrupted seed continues from its latest rolling checkpoint
    # (<=15 min lost) instead of restarting from zero. resume_from_latest
    # falls back through older slots if the newest is a partial write.
    ckpt_dir = run_dir / "checkpoints"
    if any(ckpt_dir.glob("ckpt_*.pt")):
        loaded = trainer.resume_from_latest()
        print(f"[jepa-pilot] {_run_name(arm, d_model, seed)}: resumed "
              f"mid-seed from {loaded.name} (step {trainer.global_step})")
    outcome = trainer.run()

    heldout = heldout_latent_prediction(
        loss_module,
        loader.holdout_batches("text", 8),
        "text",
        max_batches=args.heldout_batches,
    )
    train_batches = [loader.next_batch("text") for _ in range(args.probe_batches)]
    probe = fit_next_token_probe(
        loss_module, train_batches,
        vocab_size=text_ds.vocab_size(),
        max_batches=args.probe_batches,
    )
    heldout_list = list(loader.holdout_batches("text", 8))
    probe_real = probe_accuracy(loss_module, probe, heldout_list)
    probe_floor = probe_accuracy(
        loss_module, probe, heldout_list, shuffled_label_floor=True,
    )

    result = {
        "arm": arm,
        "d_model": d_model,
        "seed": seed,
        "outcome": outcome,
        # Collapse-admissibility (protocol section 1): a killed run's
        # numbers are reported but flagged inadmissible for comparison.
        "admissible": outcome == "completed",
        "heldout": heldout,
        "probe": probe_real,
        "probe_shuffled_floor": probe_floor,
        "wall_clock_seconds": time.time() - started,
        "config": {
            # Effective model shape (the depth arm overrides n_blocks
            # via ARM_CONFIGS -- record what actually trained).
            "n_blocks": model_kwargs["n_blocks"], "epochs": args.epochs,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "stride": args.stride, "lr": args.lr,
            "holdout_fraction": args.holdout_fraction,
            "data_dir": str(args.data_dir),
            # Honest data provenance: the verdict rebuild must reload
            # the SAME corpus (the holdout is corpus-derived).
            "file_list": ARM_FILELIST.get(arm),
            # v4 bundle provenance (2026-07-20): loss/trainer-side
            # settings that don't ride in ARM_CONFIGS model kwargs.
            "sigreg_lambd": ARM_SIGREG.get(arm, SIGREG_LAMBD),
            "cosine_lr": ARM_COSINE.get(arm, False),
            "lr_total_steps": lr_schedule.total_steps,
        },
    }
    _result_path(arm, d_model, seed).write_text(json.dumps(result, indent=2))
    return result


def run(stages: list[int], args) -> int:
    # Explicit --seeds overrides the prefix-of-SEEDS selection (added for
    # the stage-11 seed44 rerun, which must run seed 44 alone).
    run_seeds = (
        [int(x) for x in args.seeds.split(",")]
        if args.seeds else list(SEEDS[: args.n_seeds])
    )
    plan = [
        (arm, d, s)
        for stage in stages
        for arm, d in STAGES[stage]
        for s in run_seeds
    ]
    done = [c for c in plan if _result_path(*c).exists()]
    todo = [c for c in plan if not _result_path(*c).exists()]
    print(f"[jepa-pilot] plan: {len(plan)} runs "
          f"({len(done)} complete, {len(todo)} to run)")
    for arm, d_model, seed in todo:
        name = _run_name(arm, d_model, seed)
        if args.dry_run:
            print(f"  DRY-RUN: {name} (n_blocks={args.n_blocks}, "
                  f"epochs={args.epochs}, data={args.data_dir})")
            continue
        print(f"[jepa-pilot] starting {name}")
        try:
            result = _run_one(arm, d_model, seed, args)
        except Exception as e:  # noqa: BLE001 -- stop the queue loudly
            print(f"[jepa-pilot] FAILED: {name}: {type(e).__name__}: {e}")
            raise
        print(f"[jepa-pilot] {name}: outcome={result['outcome']} "
              f"heldout_l_pred={result['heldout']['l_pred_mean']:.6f} "
              f"probe_top1={result['probe']['top1']:.4f} "
              f"({result['wall_clock_seconds']/3600:.2f}h)")
    return 0


def aggregate() -> int:
    """Per-condition summary. Prints the ingredients; the curve-level
    verdict (effective-capacity placement) is the analysis doc's job."""
    conditions: dict[str, list[dict]] = {}
    for path in sorted(OUTPUT_ROOT.glob("*/pilot_result.json")):
        r = json.loads(path.read_text())
        conditions.setdefault(f"{r['arm']}_{r['d_model']}d", []).append(r)
    if not conditions:
        print("[aggregate] no completed runs")
        return 1
    summary = {}
    hdr = f"{'condition':<14} {'n':>2} {'adm':>3} {'l_pred mean':>12} {'std':>9} {'probe_top1':>10}"
    print(hdr)
    for cond, runs in sorted(conditions.items()):
        adm = [r for r in runs if r["admissible"]]
        vals = [r["heldout"]["l_pred_mean"] for r in adm]
        probes = [r["probe"]["top1"] for r in adm]
        mean = statistics.mean(vals) if vals else float("nan")
        std = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        p1 = statistics.mean(probes) if probes else float("nan")
        summary[cond] = {
            "n_total": len(runs), "n_admissible": len(adm),
            "l_pred_mean": mean, "l_pred_std": std,
            "probe_top1_mean": p1,
            "inadmissible": [
                {"seed": r["seed"], "outcome": r["outcome"]}
                for r in runs if not r["admissible"]
            ],
        }
        print(f"{cond:<14} {len(runs):>2} {len(adm):>3} {mean:>12.6f} "
              f"{std:>9.6f} {p1:>10.4f}")
        if len(adm) < len(runs):
            print(f"  NOTE: {len(runs)-len(adm)} inadmissible run(s) "
                  f"(killed) -- reported, excluded from means")
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[aggregate] written to {OUTPUT_ROOT / 'summary.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", type=str, default=None, help="1, 2, 3, or 'all'")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU shakeout: 64d/1-block, capped steps.")
    p.add_argument("--data_dir", type=str,
                   default=str(REPO_ROOT / "corpus_build" / "gutenberg_100"))
    p.add_argument("--tokenizer", type=str,
                   default=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"))
    p.add_argument("--n-blocks", dest="n_blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout-fraction", dest="holdout_fraction",
                   type=float, default=0.02)
    p.add_argument("--heldout-batches", dest="heldout_batches",
                   type=int, default=50)
    p.add_argument("--probe-batches", dest="probe_batches",
                   type=int, default=32)
    p.add_argument("--kill-warmup", dest="kill_warmup",
                   type=int, default=5000)
    p.add_argument("--max-batches-per-epoch", dest="max_batches_per_epoch",
                   type=int, default=-1)
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=len(SEEDS))
    p.add_argument("--seeds", type=str, default=None,
                   help="Explicit comma-separated seed list (overrides "
                        "--n-seeds), e.g. '44'.")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_batches_per_epoch = 20
        args.heldout_batches = 3
        args.probe_batches = 4
        args.n_seeds = 1
        args.kill_warmup = 10**9
        args.batch_size = 4
        args.seq_len = 64
        # Smoke shrinks the conditions too: one living + one dead at 64d.
        STAGES[1] = [("living", 64), ("dead", 64)]
        print("[jepa-pilot] SMOKE MODE: 64d, 20 steps, 1 seed, CPU-ok")

    if args.aggregate:
        return aggregate()
    if args.stage is None:
        p.error("--stage required unless --aggregate")
    stages = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    return run(stages, args)


if __name__ == "__main__":
    raise SystemExit(main())
