"""JEPA runner for M8 multimodal training.

**STATUS: Gate-1 WIP runnable skeleton, NOT production-ready.** Must-fix
items before the baseline production run (per 4.8 review 2026-06-06):

1. **Per-modality cadence for diagnostics + kill criteria.** Light/deep
   diagnostics currently fire on the global step counter, so rare
   modalities (audio at alpha=0.7 is ~0.9% of steps) get instrumented
   ~once per 11K steps -- kill-1/3/5 windows would evaluate ~once per
   40K steps for audio and could miss a real collapse. Fix: per-modality
   step counters; instrument every N steps of that modality.
2. **Kill-7 cadence + scale.** _smoothed_loss_buf is appended every 100
   global steps (not every step) and mixes per-modality losses of
   different scales -- the descent test is currently a noisy text-only
   proxy. Fix: per-modality smoothed-loss tracking, appended every step.
3. **Activate kill-2 (effective rank) and kill-4 (LID) once thresholds
   are pilot-set from M8's warmup window.** Currently computed but not
   armed.
4. **Wire kill-6 (substrate override on pred_frob/err_acc) via
   MultimodalPredictiveCodingLM.aliveness_report()**; not called yet.
5. **Predictor-trivial cosine in light metrics** -- cheap if loss returns
   predicted_target.detach().
6. **Pilot-set threshold derivation** from warmup window (currently
   static config fallbacks).

Implemented (Gate-1 sufficient):
- Temperature-weighted modality sampling (alpha as config; M8 baseline 0.7).
- Per-modality train step: one modality per step; substrate sees text/audio/
  vision interleaved across steps with frequencies set by the sampler.
- Per-modality v0.5 §5 diagnostics (light every 100 global steps, deep
  every 1000) -- but see must-fix #1.
- Time-based checkpointing (~15 min wall-clock; rolling 3 slots) with
  fsync on the data fd before rename (B6 power-loss durability).
- resume_from_latest with fallback to older slots if newest fails to load.
- Coverage-anchored epochs: epoch ends when every modality has consumed
  at least one corpus-worth of tokens.
- Multi-epoch policy with abort/continue decision at end of the first
  epoch via marker files in the run directory.
- Run config archival at launch (Gate 5).
- Data loader state_dict/load_state_dict are part of the Protocol; the
  pipeline (#8) is built resumable by construction.

Spec: docs/research/2026-06-06_m8-brief-v0.5.md
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from luthi.v2.jepa_loss import JEPALoss, MODALITIES
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    """Temperature-weighted modality sampling.

    weights[m] = corpus_size_tokens[m] ** alpha
    p[m] = weights[m] / sum(weights)

    M8 baseline: alpha = 0.7 (Brian's call 2026-06-06, see v0.5 §3).
    At alpha=0.7 over corpora ~2.2B/23M/2M (text/vision/audio):
    per-batch share ~93/6/0.9%, audio/vision get ~6-9 passes per text-pass.
    """

    corpus_sizes_tokens: dict[str, int]
    alpha: float = 0.7


@dataclass
class CheckpointConfig:
    interval_seconds: int = 15 * 60  # ~15 min wall-clock (v0.5 §4)
    rolling_slots: int = 3


@dataclass
class LoggingConfig:
    light_interval_batches: int = 100  # v0.5 §5: per-100-batch metrics
    deep_interval_batches: int = 1000  # v0.5 §5: deep metrics


@dataclass
class KillCriteriaConfig:
    """v0.5 §7 thresholds with the pilot-set derivation per 4.8 review
    2026-06-08. Three metric classes, three handling rules (the
    classification 4.8 worked out before implementation):

    - **Stationary** (online_std_p5, mean_abs_off_diag_correlation):
      pilot-set baseline = median of first ``pilot_set_n`` observations;
      kill when current strays by ``stationary_deviation_pct`` from
      baseline (down for std, up for correlation). Before pilot-set
      completes, falls back to the absolute floors below.
    - **Trending substrate-health** (pred_frob, err_acc): running-best
      anchor over a rolling-median window
      (``trending_smoothing_window``), so a single outlier reading
      can't latch the anchor. Kill activates after
      ``trending_warmup_n`` observations so early settling doesn't
      false-fire. Kill on sustained reversal toward unhealthy of
      ``substrate_health_degradation_pct``.
    - **Absolute** (encoder_asymmetry_cosine_mean,
      predictor_trivial_cosine_mean): fixed 0.99. No pilot, no
      baseline -- near-identical encoders is collapse regardless of
      training history.
    """

    warmup_batches: int = 5000  # observe-only window before any kill activates
    # Stationary fallback thresholds (used when pilot-set hasn't
    # completed for the modality OR config-override isn't supplied).
    std_collapse_threshold: float = 0.1
    correlation_collapse_threshold: float = 0.95
    # Absolute (no baselining ever).
    cosine_collapse_threshold: float = 0.99
    # Trending degradation parameters (kill-6).
    substrate_health_degradation_pct: float = 0.25
    substrate_health_window: int = 5  # sustained-checkpoints requirement
    # Kill-2 (dimensional collapse): fires on a sustained drop in
    # effective_rank below running_max * (1 - this_pct). v0.5 §7.2
    # specifies 50% of healthy baseline.
    dimensional_collapse_threshold_pct: float = 0.5
    # Pilot-set derivation -- stationary path.
    pilot_set_n: int = 10  # observations before stationary baseline is set
    stationary_deviation_pct: float = 0.5  # half the baseline triggers kill
    # Pilot-set derivation -- trending path. Per 4.8 review 2026-06-08,
    # light-cadence (pred_frob, err_acc) and deep-cadence
    # (effective_rank) trending metrics get separate warmup and
    # smoothing parameters: each deep observation covers ~10x the
    # training-progress of a light observation (deep_interval is
    # typically 10x light_interval), so the deep warmup needs to be
    # smaller in count so kill-2 actually arms on rare modalities
    # within a realistic run. Effective_rank on a fixed probe batch is
    # also lower-noise than the light metrics, so the deep smoothing
    # window can be smaller too.
    trending_smoothing_window: int = 3  # rolling-median window (light cadence)
    trending_warmup_n: int = 5  # observations before trending kill activates (light)
    trending_smoothing_window_deep: int = 2  # deep cadence (effective_rank)
    trending_warmup_n_deep: int = 2  # deep cadence (effective_rank)
    # Criterion 7 (global, not per-modality).
    loss_descent_window: int = 5000
    # Sustained-trigger requirements (consecutive checkpoint counts).
    collapse_sustained_checkpoints: int = 3
    dimensional_sustained_checkpoints: int = 5
    # Reserved for a future cross-run diagnostic (4.8 review 2026-06-08,
    # override-removal sweep): the pilot path / running-best is the
    # universal kill anchor at all widths -- it's width-independent and
    # self-referential ("is M8 degrading from its own peak?"). These
    # fields are no longer consulted by the kill helpers; they're
    # retained to receive M7-literal values for an eventual logged-but-
    # not-killing comparison ("is M8 as healthy as M7 was?"). Setting
    # them today has no effect on the kill criteria.
    substrate_health_baselines: Optional[dict[str, dict[str, float]]] = None
    stationary_baselines: Optional[dict[str, dict[str, float]]] = None


@dataclass
class EpochConfig:
    """Coverage-anchored epochs and multi-epoch policy (v0.5 §3, §10.4)."""

    max_epochs: int = 3
    # If True, abort/continue decision at end of epoch 1 by Brian; runner
    # writes the decision marker and waits for confirmation via the
    # presence of a `continue.marker` file in the run directory.
    abort_continue_at_epoch_1: bool = True
    # Safety bound: stop the epoch even if coverage anchor not yet met.
    # Useful for runaway scenarios; -1 disables.
    max_batches_per_epoch: int = -1


@dataclass
class RunnerConfig:
    sampler: SamplerConfig
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    kill_criteria: KillCriteriaConfig = field(default_factory=KillCriteriaConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class ModalitySampler:
    """Independently samples one modality per training step.

    Weights are derived from corpus token counts via temperature alpha:
    weight[m] = corpus_sizes_tokens[m] ** alpha.

    At alpha=0.7 with M8's corpora (~2.2B/23M/2M tokens for
    text/vision/audio), per-batch probabilities are ~0.93/0.06/0.009.
    """

    def __init__(self, config: SamplerConfig, generator: Optional[torch.Generator] = None):
        sizes = config.corpus_sizes_tokens
        if not sizes:
            raise ValueError("corpus_sizes_tokens cannot be empty")
        unknown = [m for m in sizes if m not in MODALITIES]
        if unknown:
            raise ValueError(
                f"corpus_sizes_tokens has unknown modalities: {unknown}; "
                f"expected subset of {MODALITIES}"
            )
        # Only modalities present in sizes participate in sampling. Text-
        # only smoke / single-modality runs configure sizes with just
        # {"text": ...}; absent modalities are silently zero-probability.
        present = tuple(m for m in MODALITIES if m in sizes)
        weights = torch.tensor(
            [sizes[m] ** config.alpha for m in present],
            dtype=torch.float64,
        )
        self.probs = weights / weights.sum()
        self.modalities = present
        self.generator = generator
        self.alpha = config.alpha
        self.corpus_sizes_tokens = dict(sizes)

    def sample(self) -> str:
        idx = torch.multinomial(self.probs, num_samples=1, generator=self.generator).item()
        return self.modalities[idx]

    def per_modality_probs(self) -> dict[str, float]:
        return {m: float(p) for m, p in zip(self.modalities, self.probs)}


# ---------------------------------------------------------------------------
# Data loader contract (implementation lives in multimodal_data.py)
# ---------------------------------------------------------------------------


class MultimodalDataLoader(Protocol):
    """Contract for the v2 multimodal data pipeline.

    The runner picks a modality via the sampler, then calls
    next_batch(modality) on the loader. The loader returns the
    modality_inputs dict that JEPALoss.compute_modality_loss expects.

    For each modality, the loader maintains its own iteration state and
    shuffled without-replacement sampling within an epoch (v0.5 §3).

    The state_dict / load_state_dict interface is part of the contract
    (per 4.8 review 2026-06-06): on resume the loader must restore its
    per-modality shuffle position so the without-replacement guarantee
    holds across crashes. Without this, the data stream becomes non-
    reproducible (data re-served or skipped). Build the pipeline
    resumable by construction.
    """

    def next_batch(self, modality: str) -> dict:
        """Returns the modality_inputs dict for one batch of `modality`.

        For "text": {"text_tokens": Tensor[B, seq_len]}
        For "audio": {"audio_waveform": Tensor[B, samples]} or
                     {"audio_tokens": Tensor[B, L, D]}
        For "vision": {"image": Tensor[B, 3, H, W]} or
                      {"vision_tokens": Tensor[B, L, D]}
        """
        ...

    def batch_token_count(self, modality: str, batch: dict) -> int:
        """How many tokens this batch contributes to coverage. Loader
        knows the per-batch token count (depends on batch_size, seq_len,
        and any padding semantics)."""
        ...

    def state_dict(self) -> dict:
        """Serialize per-modality iteration state (shuffle indices, RNG,
        epoch position) so resume restores the without-replacement order."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore per-modality iteration state from a prior state_dict()
        snapshot. Called by JEPATrainer.resume(). After load_state_dict,
        the next next_batch(modality) call must continue the same
        without-replacement sequence."""
        ...


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _percentile(t: torch.Tensor, q: float) -> float:
    """Single-percentile helper; q in [0, 1]."""
    if t.numel() == 0:
        return float("nan")
    return float(torch.quantile(t.float().flatten(), q).item())


def _light_collapse_metrics(
    online_context_latents: torch.Tensor,
    target_latents: torch.Tensor,
    predicted_target: Optional[torch.Tensor],
    ctx_len: int,
    online_std: torch.Tensor,
    l_sigreg: float,
) -> dict:
    """Per-modality light metrics. LeJEPA refactor 2026-06-09: dropped
    target_std (still produced by the loss but not as a distinct kill
    metric) and encoder_asymmetry_cosine (no asymmetry without an EMA
    target). Kept online_std, mean-abs off-diag correlation,
    predictor_trivial_cosine, and added the SIGReg loss value as a
    direct per-modality collapse signal (rising SIGReg = drifting off
    isotropic Gaussian, the regularizer's anti-collapse target).

    online_context_latents: [B, ctx_len, D] online encoder's
        context-only output.
    target_latents: [B, seq_len, D] full-sequence online encoder
        output (gradients flow; same encoder, no EMA twin).
    predicted_target: [B, tgt_len, D] predictor's output. The
        predictor-trivial cosine compares this against
        target_latents[:, ctx_len:].
    l_sigreg: scalar SIGReg statistic from this loss step.
    """
    metrics: dict = {}

    # Per-dim std summary (online context-only).
    online_std_sorted = online_std.detach().float().flatten()
    metrics["online_std_p5"] = _percentile(online_std_sorted, 0.05)
    metrics["online_std_p50"] = _percentile(online_std_sorted, 0.50)
    metrics["online_std_p95"] = _percentile(online_std_sorted, 0.95)
    metrics["online_std_below_0.1"] = int((online_std_sorted < 0.1).sum().item())
    metrics["online_std_below_0.5"] = int((online_std_sorted < 0.5).sum().item())

    # Off-diagonal correlation (mean abs).
    flat = online_context_latents.detach().float().reshape(-1, online_context_latents.shape[-1])
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    std = flat_centered.std(dim=0, unbiased=False).clamp(min=1e-8)
    flat_norm = flat_centered / std
    n = flat_norm.shape[0]
    corr = (flat_norm.t() @ flat_norm) / max(n - 1, 1)
    off_diag = corr - torch.diag(torch.diag(corr))
    d = corr.shape[0]
    metrics["mean_abs_off_diag_correlation"] = float(off_diag.abs().sum().item()) / max(d * (d - 1), 1)

    # SIGReg statistic -- direct per-modality collapse signal. Rising
    # SIGReg = encoder output drifting away from isotropic Gaussian,
    # the regularizer's target. The kill criterion based on this is the
    # natural LeJEPA-era replacement for the EMA-asymmetry kill-5.
    metrics["sigreg"] = float(l_sigreg)

    # Predictor-trivial cosine: kill-5's surviving axis. Predicted
    # vs target-block (same encoder, target positions). Still a
    # meaningful degeneracy signal in the LeJEPA refactor -- predictor
    # collapsing to identity / target-copy is the failure mode.
    if predicted_target is not None:
        target_block = target_latents[:, ctx_len:, :].detach().float()
        cos_pred = F.cosine_similarity(
            predicted_target.detach().float(),
            target_block,
            dim=-1,
        )  # [B, tgt_len]
        metrics["predictor_trivial_cosine_mean"] = float(cos_pred.mean().item())
        metrics["predictor_trivial_cosine_std"] = float(cos_pred.std().item())
        # Predictor-output std as an independent collapse signal.
        pred_std = predicted_target.detach().float().std(dim=(0, 1))
        metrics["predictor_output_std_p50"] = _percentile(pred_std, 0.50)

    return metrics


def _substrate_health_metrics(model: MultimodalPredictiveCodingLM) -> dict:
    """Per-step substrate-health aggregates (v0.5 §7.6 / kill-6 source).

    Calls ``model.aliveness_report()`` to get per-block PC layer state,
    then aggregates to single scalars matching the M7 launch.log
    conventions:

    - ``pred_frob`` = mean of ``prediction_norm`` across blocks. M7
      baseline at 1024d climbed 4.02 -> 4.59 over the 47h run; healthy
      trajectory is monotonically increasing (substrate building
      predictive structure). Kill-6: degraded = below baseline by
      ``substrate_health_degradation_pct``.
    - ``err_acc`` = mean of ``error_acc_mean`` across blocks. M7
      baseline at 1024d descended 0.015 -> 0.003; healthy trajectory is
      decreasing (substrate learning to predict its own input).
      Kill-6: degraded = above baseline by the same fraction.
    """
    aliveness = model.aliveness_report()
    pred_norms = [
        a["prediction_norm"] for a in aliveness if "prediction_norm" in a
    ]
    err_accs = [
        a["error_acc_mean"] for a in aliveness if "error_acc_mean" in a
    ]
    return {
        "pred_frob": (
            float(sum(pred_norms) / len(pred_norms))
            if pred_norms else float("nan")
        ),
        "err_acc": (
            float(sum(err_accs) / len(err_accs))
            if err_accs else float("nan")
        ),
    }


def _deep_collapse_metrics(online_context_latents: torch.Tensor) -> dict:
    """v0.5 §5 deep metrics, per modality.

    SVD spectrum + effective rank + stable rank. LID is intentionally
    omitted from this first cut: the MLE/Fisher-Rao estimator is more
    involved and the simpler rank measures already give us dimensional-
    collapse signal. LID landing is a follow-up; flagged.
    """
    metrics: dict = {}
    flat = online_context_latents.detach().float().reshape(-1, online_context_latents.shape[-1])
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    n = flat_centered.shape[0]
    cov = (flat_centered.t() @ flat_centered) / max(n - 1, 1)
    sing_vals = torch.linalg.svdvals(cov).clamp(min=1e-12)
    log_sv = torch.log(sing_vals)

    # Spectral entropy -> effective rank (exp of entropy).
    p = sing_vals / sing_vals.sum()
    spectral_entropy = float(-(p * torch.log(p.clamp(min=1e-12))).sum().item())
    metrics["effective_rank"] = float(math.exp(spectral_entropy))

    # Stable rank: ||C||_F^2 / ||C||_2^2.
    metrics["stable_rank"] = float((sing_vals.pow(2).sum() / sing_vals.max().pow(2)).item())

    # Cumulative-variance indices (where do we cross 90% / 99% of variance?).
    cumsum = sing_vals.cumsum(0) / sing_vals.sum()
    metrics["sv_index_at_90pct"] = int((cumsum >= 0.90).nonzero()[0].item()) + 1
    metrics["sv_index_at_99pct"] = int((cumsum >= 0.99).nonzero()[0].item()) + 1

    # Log-spectrum head/tail summary.
    metrics["log_sv_max"] = float(log_sv.max().item())
    metrics["log_sv_min"] = float(log_sv.min().item())
    metrics["log_sv_range"] = float((log_sv.max() - log_sv.min()).item())

    return metrics


# ---------------------------------------------------------------------------
# Kill-criteria state
# ---------------------------------------------------------------------------


class _PerModalityHistory:
    """Rolling per-modality metric history for kill-criteria evaluation.

    Persistence (state_dict/load_state_dict) added 2026-06-08 to close
    the gap 4.8 flagged in the runner-review: a 15-min checkpoint that
    lands mid-pilot for a rare modality must not silently restart the
    history accumulation.
    """

    def __init__(self, sustained_count: int):
        self._history: dict[str, dict[str, deque]] = {
            m: {} for m in MODALITIES
        }
        self.sustained_count = sustained_count

    def push(self, modality: str, metrics: dict) -> None:
        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            buf = self._history[modality].setdefault(
                key, deque(maxlen=self.sustained_count * 4),
            )
            buf.append(val)

    def recent(self, modality: str, key: str, n: int) -> list[float]:
        buf = self._history.get(modality, {}).get(key)
        if buf is None:
            return []
        return list(buf)[-n:]

    def state_dict(self) -> dict:
        return {
            "sustained_count": self.sustained_count,
            "history": {
                m: {k: list(buf) for k, buf in d.items()}
                for m, d in self._history.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.sustained_count = int(state["sustained_count"])
        cap = self.sustained_count * 4
        self._history = {}
        for m, d in state.get("history", {}).items():
            self._history[m] = {
                k: deque(vals, maxlen=cap) for k, vals in d.items()
            }
        for m in MODALITIES:
            self._history.setdefault(m, {})


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class JEPATrainer:
    """Orchestrates M8 multimodal JEPA training.

    Per-step: sampler chooses one modality, data loader produces a batch,
    loss is computed, backward, optimizer step, EMA target update.

    Per-100-batches: light §5 diagnostics; per-1000-batches: deep §5.

    Time-based checkpointing on a wall-clock interval (default ~15 min).
    Coverage-anchored epochs (each modality must consume >= 1 corpus).
    Kill criteria activate after warmup_batches with pilot-set thresholds.
    """

    def __init__(
        self,
        loss_module: JEPALoss,
        optimizer: Optimizer,
        sampler: ModalitySampler,
        data_loader: MultimodalDataLoader,
        config: RunnerConfig,
        run_dir: Path,
    ):
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.sampler = sampler
        self.data_loader = data_loader
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # State.
        self.global_step = 0
        # Per-modality step counter (4.8 review 2026-06-06 item A). The
        # cadence for diagnostics, kill-criteria history pushes, and the
        # kill-criteria warmup all key off the modality's *own* step
        # count so rare-modality kill windows mean "consecutive
        # observations of that modality" rather than "consecutive
        # observations of anything." Without this, audio at ~0.9% of
        # global steps would have kill windows that fire once per ~40K
        # global steps -- effectively dead detection on the very axis
        # the per-modality design exists to protect.
        self.modality_step: dict[str, int] = {m: 0 for m in MODALITIES}
        self.epoch = 0
        self.tokens_consumed: dict[str, int] = {m: 0 for m in MODALITIES}
        self.epoch_token_baseline: dict[str, int] = {m: 0 for m in MODALITIES}
        self.run_start_time = time.monotonic()
        self.last_checkpoint_time = self.run_start_time

        # Log files.
        self.metric_log_path = self.run_dir / "training_log.jsonl"
        self.human_log_path = self.run_dir / "training.log"

        # Kill-criteria history.
        self.history = _PerModalityHistory(
            sustained_count=config.kill_criteria.dimensional_sustained_checkpoints,
        )
        # Smoothed loss for criterion 7.
        self._smoothed_loss_buf: deque[float] = deque(
            maxlen=config.kill_criteria.loss_descent_window,
        )

        # Pilot-set state per 4.8 review 2026-06-08 (#9 item, pilot-set
        # threshold derivation):
        #
        # Stationary metrics (online_std_p5, mean_abs_off_diag_correlation):
        # collect first pilot_set_n observations, then derive median as
        # baseline. Kill condition compares current against baseline.
        #
        # Trending metrics (pred_frob, err_acc): maintain a rolling-median
        # smoothing buffer (outlier-robust per 4.8); update the running-
        # best (max for pred_frob, min for err_acc) from smoothed values
        # only. Trending kill activates after trending_warmup_n
        # observations so early settling doesn't false-fire.
        kc = config.kill_criteria
        self._pilot_observations: dict[str, dict[str, list[float]]] = {
            m: {} for m in MODALITIES
        }
        self._stationary_baselines: dict[str, dict[str, float]] = {
            m: {} for m in MODALITIES
        }
        self._trending_smoothing_buf: dict[str, dict[str, deque]] = {
            m: {} for m in MODALITIES
        }
        self._running_best: dict[str, dict[str, float]] = {
            m: {} for m in MODALITIES
        }
        self._trending_obs_counts: dict[str, dict[str, int]] = {
            m: {} for m in MODALITIES
        }

        # Archive run config (Gate 5).
        self._archive_run_config()

    # -- Run config archival --

    def _archive_run_config(self) -> None:
        """Serialize all hyperparameters at launch (v0.5 §6 Gate 5)."""
        config_path = self.run_dir / "run_config.json"
        config_dict = asdict(self.config)
        # Add per-modality sampler probabilities (auditable).
        config_dict["sampler_probabilities"] = self.sampler.per_modality_probs()
        # Add loss-module hyperparameters that aren't in RunnerConfig.
        # LeJEPA refactor 2026-06-09: SIGReg replaces the VICReg + EMA
        # parameter block. Only sigreg_lambd, SIGReg's own params, and
        # the masking context_fraction are loss-side knobs now.
        config_dict["loss"] = {
            "sigreg_lambd": self.loss_module.sigreg_lambd,
            "sigreg_knots": int(self.loss_module.sigreg.t.numel()),
            "sigreg_num_proj": int(self.loss_module.sigreg.num_proj),
            "context_fraction": self.loss_module.context_fraction,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        logger.info("Archived run config to %s", config_path)

    # -- Train step --

    def train_step(self, modality: str, batch: dict) -> dict:
        """One per-modality training step.

        Returns: dict with "loss" (float), "modality", and the loss module's
        raw result for downstream diagnostics.
        """
        # Ensure correct train/eval state: JEPALoss.train() override keeps
        # target encoder in eval regardless of the runner's mode toggles
        # (B6 / Blocker 2 fix in jepa_loss.py).
        self.loss_module.train()
        # The model that the loss wraps must be in training mode for the
        # PC top-down sweep to run on the online encoder.
        self.loss_module.online_encoder.train()

        result = self.loss_module.compute_modality_loss(modality, batch)
        loss: torch.Tensor = result["loss"]

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # LeJEPA refactor 2026-06-09: no EMA target encoder to update.
        # SIGReg + projection-head BN handles anti-collapse via a direct
        # per-batch regularizer instead of the asymmetric EMA twin.

        # Free per-layer forward-pass snapshots (PredictiveCodingLM
        # convention -- audit fix 2026-05-11).
        self.loss_module.online_encoder.clear_forward_cache()

        loss_value = float(loss.detach().item())

        # Append to the smoothed-loss buffer every step (4.8 review
        # 2026-06-06 item D). Previously this lived in
        # _compute_and_log_diagnostics, which fires every N steps, so
        # loss_descent_window=5000 actually measured 5000*N steps and
        # the config name lied. Mixed modalities here is intentional --
        # criterion 7 is about total objective trainability per §7.7;
        # per-modality collapse is caught by criteria 1-5.
        self._smoothed_loss_buf.append(loss_value)

        # Per-modality step counter advanced here (4.8 review 2026-06-06
        # item A) so callers don't need to remember to do it; the
        # post-train_step value is what cadence checks should read.
        self.global_step += 1
        self.modality_step[modality] += 1

        return {
            "loss": loss_value,
            "modality": modality,
            "raw": result,
        }

    # -- Diagnostics --

    def _compute_and_log_diagnostics(
        self,
        step_out: dict,
        light: bool,
        deep: bool,
    ) -> dict:
        modality = step_out["modality"]
        raw = step_out["raw"]

        record: dict = {
            "step": self.global_step,
            "modality": modality,
            "loss": step_out["loss"],
            "l_pred": float(raw["l_pred"].item()),
            # LeJEPA refactor 2026-06-09: l_var / l_cov replaced by
            # l_sigreg. SIGReg is the single anti-collapse term that
            # subsumes both variance and covariance regularization.
            "l_sigreg": float(raw["l_sigreg"].item()),
            "tokens_consumed": dict(self.tokens_consumed),
            "elapsed_seconds": time.monotonic() - self.run_start_time,
        }

        # Compute light + substrate + deep metrics independently first,
        # then push to history and advance pilot state once. This makes
        # the pilot-state update correct regardless of whether
        # deep ⊂ light (the latent fragility 4.8 flagged 2026-06-08:
        # the old code called _advance_pilot_state inside if light: and
        # passed record.get("deep"), which would always be None because
        # the deep branch hadn't run yet -- effective_rank was being
        # silently dropped on deep-only steps AND co-fire steps).
        light_m: Optional[dict] = None
        substrate_m: Optional[dict] = None
        deep_m: Optional[dict] = None

        if light:
            light_m = _light_collapse_metrics(
                online_context_latents=raw["online_context_latents"],
                target_latents=raw["target_latents"],
                predicted_target=raw.get("predicted_target"),
                ctx_len=raw["ctx_len"],
                online_std=raw["online_std"],
                # LeJEPA refactor: SIGReg loss value is the per-modality
                # collapse signal (no more target-encoder std).
                l_sigreg=float(raw["l_sigreg"].item()),
            )
            record["light"] = light_m
            self.history.push(modality, light_m)

            # Kill-6 source (v0.5 §7.6): substrate health from
            # aliveness_report. Computed every light firing so the same
            # per-modality cadence governs kill-6 as the collapse criteria.
            substrate_m = _substrate_health_metrics(
                self.loss_module.online_encoder,
            )
            record["substrate"] = substrate_m
            self.history.push(modality, substrate_m)

        if deep:
            deep_m = _deep_collapse_metrics(raw["online_context_latents"])
            record["deep"] = deep_m
            self.history.push(modality, deep_m)

        # Pilot-set state advancement -- runs whenever at least one
        # metric block was computed this firing. Empty dicts for absent
        # metrics let _advance_pilot_state skip them cleanly.
        if light_m is not None or deep_m is not None:
            self._advance_pilot_state(
                modality,
                light_m=light_m or {},
                substrate_m=substrate_m or {},
                deep_m=deep_m,
            )

        # Note: _smoothed_loss_buf.append is now in train_step (4.8 item D
        # fix 2026-06-06) so it runs every step rather than every N steps.

        # Persist to JSONL.
        with open(self.metric_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def _human_log_line(self, record: dict) -> None:
        elapsed_h = record["elapsed_seconds"] / 3600.0
        light = record.get("light", {})
        msg = (
            f"[step {record['step']:>7}] mod={record['modality']:<6} "
            f"loss={record['loss']:.4f} "
            f"L_pred={record['l_pred']:.4f} "
            f"L_sigreg={record['l_sigreg']:.4f} "
            f"std_p5={light.get('online_std_p5', float('nan')):.4f} "
            f"cos_pred={light.get('predictor_trivial_cosine_mean', float('nan')):.4f} "
            f"elapsed={elapsed_h:.2f}h"
        )
        logger.info(msg)
        with open(self.human_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # -- Pilot-set state advancement (per 4.8 review 2026-06-08) --

    # Metric classification (the three classes 4.8 worked out).
    _STATIONARY_METRICS = (
        "online_std_p5",
        "mean_abs_off_diag_correlation",
    )
    # Direction = "max" means running max is the healthy anchor (kill on
    # drop below); "min" means running min (kill on rise above).
    # Cadence = "light" updates every light firing; "deep" updates every
    # deep firing (effective_rank lives in the heavier deep metrics).
    # Cadence determines which smoothing window + warmup count applies
    # (per 4.8 review 2026-06-08).
    _TRENDING_METRICS: dict[str, dict[str, str]] = {
        "pred_frob": {"direction": "max", "cadence": "light"},
        "err_acc": {"direction": "min", "cadence": "light"},
        # effective_rank: dimensional collapse signal (v0.5 §7.2 / kill-2).
        # Healthy trajectory rises as the substrate spans more dimensions;
        # a sustained drop below the running max = dimensional collapse.
        "effective_rank": {"direction": "max", "cadence": "deep"},
    }

    def _advance_pilot_state(
        self,
        modality: str,
        light_m: dict,
        substrate_m: dict,
        deep_m: Optional[dict] = None,
    ) -> None:
        """Called once per diagnostic firing. Updates the stationary
        observation list (for median-of-first-N baselining) and the
        trending smoothing buffer + running-best (for outlier-robust
        anchor tracking).

        ``deep_m`` is supplied only on deep firings; effective_rank
        lives there. Trending observations for effective_rank
        therefore accumulate at the deep cadence (every
        ``deep_interval_batches`` modality steps), not the light
        cadence -- correct, since deep metrics are heavier and fire
        less often.
        """
        # Stationary: collect raw observations until pilot_set_n hit,
        # then derive median baseline once.
        for metric in self._STATIONARY_METRICS:
            val = light_m.get(metric)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                continue
            self._observe_stationary(modality, metric, float(val))

        # Trending: source dict depends on the metric -- pred_frob /
        # err_acc come from substrate_m, effective_rank from deep_m.
        sources: dict[str, Optional[dict]] = {
            "pred_frob": substrate_m,
            "err_acc": substrate_m,
            "effective_rank": deep_m,
        }
        for metric, info in self._TRENDING_METRICS.items():
            source = sources.get(metric)
            if source is None:
                continue
            val = source.get(metric)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                continue
            self._observe_trending(
                modality, metric, float(val),
                direction=info["direction"], cadence=info["cadence"],
            )

    def _observe_stationary(
        self, modality: str, metric: str, value: float,
    ) -> None:
        # Once a baseline is set, no need to keep collecting (kill check
        # uses the derived baseline forever after).
        if metric in self._stationary_baselines[modality]:
            return
        obs = self._pilot_observations[modality].setdefault(metric, [])
        obs.append(value)
        if len(obs) >= self.config.kill_criteria.pilot_set_n:
            # Median of first pilot_set_n -- robust against startup noise.
            window = sorted(obs[: self.config.kill_criteria.pilot_set_n])
            self._stationary_baselines[modality][metric] = window[len(window) // 2]

    def _observe_trending(
        self,
        modality: str,
        metric: str,
        value: float,
        direction: str,
        cadence: str,
    ) -> None:
        # Cadence-specific smoothing window: light-cadence metrics
        # observe ~10x more frequently than deep-cadence ones at
        # production intervals, so the deep window can be smaller
        # (effective_rank on a fixed probe batch is also lower-noise).
        if cadence == "deep":
            smoothing_w = self.config.kill_criteria.trending_smoothing_window_deep
        else:
            smoothing_w = self.config.kill_criteria.trending_smoothing_window
        buf = self._trending_smoothing_buf[modality].get(metric)
        if buf is None or buf.maxlen != smoothing_w:
            # First observation OR cadence-config changed since the
            # buffer was built (handles config-tweaks between resumes).
            buf = deque(buf or [], maxlen=smoothing_w)
            self._trending_smoothing_buf[modality][metric] = buf
        buf.append(value)
        self._trending_obs_counts[modality][metric] = (
            self._trending_obs_counts[modality].get(metric, 0) + 1
        )
        if len(buf) < smoothing_w:
            return  # need a full window before we trust the smoothed value
        smoothed = sorted(buf)[len(buf) // 2]  # median over window
        current = self._running_best[modality].get(metric)
        if current is None:
            self._running_best[modality][metric] = smoothed
        elif direction == "max" and smoothed > current:
            self._running_best[modality][metric] = smoothed
        elif direction == "min" and smoothed < current:
            self._running_best[modality][metric] = smoothed

    def _get_stationary_baseline(
        self, modality: str, metric: str,
    ) -> Optional[float]:
        """Returns the pilot-derived baseline for a stationary metric, or
        None if pilot-set hasn't completed for this modality+metric.

        Per 4.8 review 2026-06-08 (override-removal sweep): the pilot
        path is now the canonical baseline at all widths, replacing the
        earlier config-override-wins precedence. KillCriteriaConfig.
        stationary_baselines is no longer consulted here; the field is
        retained for backward-compat and reserved for a future cross-
        run diagnostic ("are M8's healthy stationary values comparable
        to M7's?") logged alongside the pilot baseline but not driving
        the kill.
        """
        return self._stationary_baselines.get(modality, {}).get(metric)

    def _get_trending_anchor(
        self, modality: str, metric: str,
    ) -> Optional[float]:
        """Returns the smoothed running-best anchor for a trending
        metric. None means trending kill is inactive for this modality
        (warmup not yet met, or no observations yet).

        Per 4.8 review 2026-06-08 (override-removal sweep): running-best
        is the universal kill anchor at all widths -- it's width-
        independent and self-referential ("is M8 degrading from its own
        peak?"). The earlier KillCriteriaConfig.substrate_health_baselines
        override is no longer consulted; the field is retained for
        backward-compat and reserved for a future cross-run diagnostic
        ("is M8 as healthy as M7 was?") logged alongside the running-best
        but not driving the kill.

        Cadence-specific warmup: deep-cadence trending metrics
        (effective_rank) need fewer warmup observations because each
        deep observation covers more training-progress.
        """
        info = self._TRENDING_METRICS.get(metric)
        cadence = info["cadence"] if info else "light"
        if cadence == "deep":
            warmup_n = self.config.kill_criteria.trending_warmup_n_deep
        else:
            warmup_n = self.config.kill_criteria.trending_warmup_n
        if self._trending_obs_counts[modality].get(metric, 0) < warmup_n:
            return None
        return self._running_best.get(modality, {}).get(metric)

    # -- Kill criteria --

    def _check_kill_criteria(self, modality: str) -> Optional[str]:
        """Returns kill reason string if any criterion triggers for this
        modality's recent history, else None. Activates only after the
        modality's *own* warmup -- per 4.8 review 2026-06-06 item A, the
        warmup is per-modality so a rare modality's kill criteria activate
        once it has actually been observed enough times, not once enough
        global steps have happened (which at alpha=0.7 would let audio
        run for ~5.5M global steps before its kill window had 5 audio
        observations to evaluate against)."""
        if self.modality_step[modality] < self.config.kill_criteria.warmup_batches:
            return None

        cfg = self.config.kill_criteria

        # Criterion 1: complete collapse (online std 5th-pct under floor).
        # Pilot-set per 4.8 review 2026-06-08: use the derived baseline
        # (median of first pilot_set_n observations) when available;
        # otherwise fall back to the absolute std_collapse_threshold.
        recent_std = self.history.recent(
            modality, "online_std_p5", cfg.collapse_sustained_checkpoints,
        )
        pilot_std = self._get_stationary_baseline(modality, "online_std_p5")
        if pilot_std is not None:
            std_threshold = pilot_std * (1.0 - cfg.stationary_deviation_pct)
            threshold_source = (
                f"baseline {pilot_std:.4f} -- {cfg.stationary_deviation_pct:.0%}"
            )
        else:
            std_threshold = cfg.std_collapse_threshold
            threshold_source = f"absolute floor {std_threshold}"
        if len(recent_std) >= cfg.collapse_sustained_checkpoints and all(
            v < std_threshold for v in recent_std
        ):
            return (
                f"kill-1 (complete collapse) on {modality}: "
                f"std_p5 < {std_threshold:.4f} ({threshold_source}) for "
                f"{cfg.collapse_sustained_checkpoints} checkpoints"
            )

        # Criterion 2: dimensional collapse via effective rank (v0.5 §7.2).
        # Activated 2026-06-08 against the same trending machinery as
        # kill-6 (4.8 review note for kill-2 wiring): direction="max",
        # rolling-median smoothing buffer, warmup gate, running-best
        # anchor. Kill fires on sustained drop below
        # running_max * (1 - dimensional_collapse_threshold_pct).
        er_anchor = self._get_trending_anchor(modality, "effective_rank")
        if er_anchor is not None:
            er_threshold = er_anchor * (1.0 - cfg.dimensional_collapse_threshold_pct)
            recent_er = self.history.recent(
                modality, "effective_rank",
                cfg.dimensional_sustained_checkpoints,
            )
            if (
                len(recent_er) >= cfg.dimensional_sustained_checkpoints
                and all(v < er_threshold for v in recent_er)
            ):
                return (
                    f"kill-2 (dimensional collapse, effective rank) on "
                    f"{modality}: effective_rank < {er_threshold:.4f} "
                    f"(running max {er_anchor:.4f} -- "
                    f"{cfg.dimensional_collapse_threshold_pct:.0%}) for "
                    f"{cfg.dimensional_sustained_checkpoints} checkpoints"
                )

        # Criterion 3: dimensional collapse (off-diagonal correlation).
        # Pilot-set per 4.8 review 2026-06-08: threshold = baseline +
        # (1 - baseline) * stationary_deviation_pct (the absolute
        # version works correctly for the [0, 1]-bounded correlation).
        recent_corr = self.history.recent(
            modality,
            "mean_abs_off_diag_correlation",
            cfg.dimensional_sustained_checkpoints,
        )
        pilot_corr = self._get_stationary_baseline(
            modality, "mean_abs_off_diag_correlation",
        )
        if pilot_corr is not None:
            corr_threshold = pilot_corr + (1.0 - pilot_corr) * cfg.stationary_deviation_pct
            corr_source = (
                f"baseline {pilot_corr:.4f} + {cfg.stationary_deviation_pct:.0%} "
                f"of headroom"
            )
        else:
            corr_threshold = cfg.correlation_collapse_threshold
            corr_source = f"absolute ceiling {corr_threshold}"
        if len(recent_corr) >= cfg.dimensional_sustained_checkpoints and all(
            v > corr_threshold for v in recent_corr
        ):
            return (
                f"kill-3 (correlation collapse) on {modality}: "
                f"mean_abs_off_diag > {corr_threshold:.4f} ({corr_source}) for "
                f"{cfg.dimensional_sustained_checkpoints} checkpoints"
            )

        # Criterion 5: predictor-trivial cosine. LeJEPA refactor
        # 2026-06-09 removed the encoder-asymmetry axis (no EMA target,
        # so the "online vs target divergence" framing is moot). The
        # predictor-trivial axis survives unchanged: predictor learned
        # the identity / target-copy is the failure mode, distinct from
        # encoder distributional collapse (which SIGReg + kill-1/2/3
        # catch).
        recent_pred_cos = self.history.recent(
            modality,
            "predictor_trivial_cosine_mean",
            cfg.dimensional_sustained_checkpoints,
        )
        if len(recent_pred_cos) >= cfg.dimensional_sustained_checkpoints and all(
            v > cfg.cosine_collapse_threshold for v in recent_pred_cos
        ):
            return (
                f"kill-5 (predictor-trivial) on {modality}: "
                f"predicted-vs-target cosine > {cfg.cosine_collapse_threshold} for "
                f"{cfg.dimensional_sustained_checkpoints} checkpoints"
            )

        # Criterion 6: substrate override on pred_frob / err_acc
        # (v0.5 §7.6). Per 4.8 review 2026-06-08, the anchor is the
        # outlier-robust running best of smoothed observations (not the
        # static early baseline) -- the static early baseline would
        # silently miss a substrate that climbs to a new high and then
        # degrades back to its early value. _get_trending_anchor
        # returns the running-best (or config-override at 1024d) and
        # None until trending_warmup_n observations have accumulated.
        deg = cfg.substrate_health_degradation_pct

        pf_anchor = self._get_trending_anchor(modality, "pred_frob")
        if pf_anchor is not None:
            pf_threshold = pf_anchor * (1.0 - deg)
            recent_pf = self.history.recent(
                modality, "pred_frob", cfg.substrate_health_window,
            )
            if (
                len(recent_pf) >= cfg.substrate_health_window
                and all(v < pf_threshold for v in recent_pf)
            ):
                return (
                    f"kill-6 (substrate override, pred_frob) on "
                    f"{modality}: pred_frob < {pf_threshold:.4f} "
                    f"(running max {pf_anchor:.4f} -- "
                    f"{deg:.0%}) for "
                    f"{cfg.substrate_health_window} checkpoints"
                )

        ea_anchor = self._get_trending_anchor(modality, "err_acc")
        if ea_anchor is not None:
            ea_threshold = ea_anchor * (1.0 + deg)
            recent_ea = self.history.recent(
                modality, "err_acc", cfg.substrate_health_window,
            )
            if (
                len(recent_ea) >= cfg.substrate_health_window
                and all(v > ea_threshold for v in recent_ea)
            ):
                return (
                    f"kill-6 (substrate override, err_acc) on "
                    f"{modality}: err_acc > {ea_threshold:.4f} "
                    f"(running min {ea_anchor:.4f} + "
                    f"{deg:.0%}) for "
                    f"{cfg.substrate_health_window} checkpoints"
                )

        # Criterion 7 (smoothed total loss descent) is global, not per-
        # modality; check it after warmup once smoothed buffer is full.
        if len(self._smoothed_loss_buf) == cfg.loss_descent_window:
            first_half = list(self._smoothed_loss_buf)[: cfg.loss_descent_window // 2]
            second_half = list(self._smoothed_loss_buf)[cfg.loss_descent_window // 2 :]
            if (sum(second_half) / len(second_half)) >= (
                sum(first_half) / len(first_half)
            ):
                return (
                    f"kill-7 (objective unlearnable): smoothed loss did "
                    f"not descend over {cfg.loss_descent_window} steps"
                )

        return None

    # -- Coverage --

    def _update_coverage(self, modality: str, batch: dict) -> None:
        n_tokens = self.data_loader.batch_token_count(modality, batch)
        self.tokens_consumed[modality] += n_tokens

    def _epoch_done(self) -> bool:
        """Coverage anchor: every modality registered with the sampler has
        consumed >= 1 corpus-worth of tokens beyond its epoch baseline
        (v0.5 §3, F5 from 4.8 2026-06-06). Modalities absent from the
        sampler (e.g. text-only runs) are skipped -- they cannot be
        sampled, so they have no coverage to wait on.
        """
        for m in self.sampler.modalities:
            target = self.sampler.corpus_sizes_tokens[m]
            if self.tokens_consumed[m] - self.epoch_token_baseline[m] < target:
                return False
        return True

    def _start_new_epoch(self) -> None:
        self.epoch_token_baseline = dict(self.tokens_consumed)

    # -- Checkpointing --

    def _checkpoint_if_due(self) -> None:
        now = time.monotonic()
        if now - self.last_checkpoint_time < self.config.checkpoint.interval_seconds:
            return
        self._checkpoint(reason="interval")
        self.last_checkpoint_time = now

    def _checkpoint(self, reason: str) -> None:
        """Atomic, fsync'd rolling checkpoint with v0.5 §4 contents.

        Crash-durability path (per 4.8 review 2026-06-06):
        1. Write to tmp file.
        2. flush + fsync the data fd (forces it out of page cache).
        3. os.replace (atomic on NTFS / POSIX).
        4. On Windows, directory fsync is not portable; the NTFS metadata
           journal makes the rename durable on its own. On POSIX we'd
           additionally fsync the directory.
        Resume tries the newest checkpoint first and falls back to older
        slots on load failure.
        """
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        # Find next slot index.
        existing = sorted(ckpt_dir.glob("ckpt_*.pt"))
        next_idx = (
            (int(existing[-1].stem.split("_")[-1]) + 1) if existing else 0
        )
        slot_path = ckpt_dir / f"ckpt_{next_idx:08d}.pt"

        # Data-loader iteration state (C from 4.8 review: without-replacement
        # guarantee requires saving the loader's shuffle position).
        try:
            loader_state = self.data_loader.state_dict()
        except (AttributeError, NotImplementedError):
            loader_state = None
            logger.warning(
                "Data loader does not implement state_dict(); without-"
                "replacement guarantee will not survive resume.",
            )

        # Sampler RNG state, if the sampler uses a non-global generator.
        sampler_gen_state = (
            self.sampler.generator.get_state()
            if self.sampler.generator is not None
            else None
        )

        state = {
            "global_step": self.global_step,
            "modality_step": dict(self.modality_step),
            "epoch": self.epoch,
            "tokens_consumed": dict(self.tokens_consumed),
            "epoch_token_baseline": dict(self.epoch_token_baseline),
            # Kill-criteria history + pilot-set state (4.8 review
            # 2026-06-08: without these, a 15-min checkpoint that lands
            # mid-pilot silently restarts the pilot derivation for any
            # rare modality and loses progress).
            "kill_history": self.history.state_dict(),
            "pilot_observations": {
                m: {k: list(v) for k, v in d.items()}
                for m, d in self._pilot_observations.items()
            },
            "stationary_baselines": {
                m: dict(d) for m, d in self._stationary_baselines.items()
            },
            "trending_smoothing_buf": {
                m: {k: list(v) for k, v in d.items()}
                for m, d in self._trending_smoothing_buf.items()
            },
            "running_best": {
                m: dict(d) for m, d in self._running_best.items()
            },
            "trending_obs_counts": {
                m: dict(d) for m, d in self._trending_obs_counts.items()
            },
            "smoothed_loss_buf": list(self._smoothed_loss_buf),
            "online_state_dict": self.loss_module.online_encoder.state_dict(),
            "predictor_state_dict": self.loss_module.predictor.state_dict(),
            # LeJEPA refactor 2026-06-09: per-modality projection heads
            # (Linear -> BN) are part of the loss module's state.
            "projection_heads_state_dict": self.loss_module.projection_heads.state_dict(),
            "loss_module_buffers": {
                name: buf.detach().clone()
                for name, buf in self.loss_module.named_buffers()
                # online_encoder.* and predictor.* are saved above; SIGReg's
                # quadrature buffers and the action_token are reconstructed
                # at init from constants, so we exclude them too.
                if not name.startswith((
                    "online_encoder.",
                    "predictor.",
                    "projection_heads.",
                    "sigreg.",
                ))
                and name != "action_token"
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "sampler_generator_state": sampler_gen_state,
            "data_loader_state": loader_state,
            "reason": reason,
            "wall_clock_seconds": time.monotonic() - self.run_start_time,
        }

        # Atomic write: save -> flush -> fsync -> atomic rename.
        tmp_path = slot_path.with_suffix(".pt.tmp")
        with open(tmp_path, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, slot_path)

        # Enforce rolling cap.
        existing = sorted(ckpt_dir.glob("ckpt_*.pt"))
        excess = len(existing) - self.config.checkpoint.rolling_slots
        for old in existing[:excess]:
            try:
                old.unlink()
            except OSError:
                logger.warning("Failed to remove old checkpoint %s", old)

        logger.info(
            "Checkpoint written: %s (reason=%s, step=%d)",
            slot_path, reason, self.global_step,
        )

    def resume(self, ckpt_path: Path) -> None:
        """Load state from a specific checkpoint and continue from where
        it left off. Raises on load failure -- prefer resume_from_latest
        for production use, which falls back to older slots."""
        # weights_only=False is explicit (silences the FutureWarning):
        # our checkpoint contains optimizer state with Python objects
        # that pickle-loads must reconstruct -- not weights_only-safe.
        # Allowlisting via add_safe_globals is the long-term answer if
        # PyTorch flips the default; for now explicit-False matches the
        # current default and stays warning-quiet.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.global_step = state["global_step"]
        # modality_step added 2026-06-06 (item A). Older checkpoints
        # without it resume at zero per-modality counts; that's a
        # degraded resume (per-modality kill warmup restarts and
        # cadence shifts) but better than a crash.
        if "modality_step" in state:
            self.modality_step = dict(state["modality_step"])
        else:
            logger.warning(
                "Checkpoint missing modality_step; resuming at zero per-"
                "modality counts. Per-modality kill warmup will restart."
            )
            self.modality_step = {m: 0 for m in MODALITIES}
        self.epoch = state["epoch"]
        self.tokens_consumed = dict(state["tokens_consumed"])
        self.epoch_token_baseline = dict(state["epoch_token_baseline"])

        # Kill-criteria history + pilot-set state (added 2026-06-08).
        # Older checkpoints (pre-pilot-set) won't have these; that's a
        # degraded resume -- pilot derivation restarts but the run
        # continues.
        if "kill_history" in state:
            self.history.load_state_dict(state["kill_history"])
        if "pilot_observations" in state:
            self._pilot_observations = {
                m: {k: list(v) for k, v in d.items()}
                for m, d in state["pilot_observations"].items()
            }
            for m in MODALITIES:
                self._pilot_observations.setdefault(m, {})
        if "stationary_baselines" in state:
            self._stationary_baselines = {
                m: dict(d) for m, d in state["stationary_baselines"].items()
            }
            for m in MODALITIES:
                self._stationary_baselines.setdefault(m, {})
        if "trending_smoothing_buf" in state:
            smoothing_w = self.config.kill_criteria.trending_smoothing_window
            self._trending_smoothing_buf = {
                m: {k: deque(v, maxlen=smoothing_w) for k, v in d.items()}
                for m, d in state["trending_smoothing_buf"].items()
            }
            for m in MODALITIES:
                self._trending_smoothing_buf.setdefault(m, {})
        if "running_best" in state:
            self._running_best = {
                m: dict(d) for m, d in state["running_best"].items()
            }
            for m in MODALITIES:
                self._running_best.setdefault(m, {})
        if "trending_obs_counts" in state:
            self._trending_obs_counts = {
                m: dict(d) for m, d in state["trending_obs_counts"].items()
            }
            for m in MODALITIES:
                self._trending_obs_counts.setdefault(m, {})
        if "smoothed_loss_buf" in state:
            self._smoothed_loss_buf = deque(
                state["smoothed_loss_buf"],
                maxlen=self.config.kill_criteria.loss_descent_window,
            )

        self.loss_module.online_encoder.load_state_dict(state["online_state_dict"])
        self.loss_module.predictor.load_state_dict(state["predictor_state_dict"])
        # LeJEPA refactor 2026-06-09: projection heads added; older
        # checkpoints (pre-refactor) won't have them.
        if "projection_heads_state_dict" in state:
            self.loss_module.projection_heads.load_state_dict(
                state["projection_heads_state_dict"],
            )
        # Loss-module's miscellaneous buffers (none currently exposed
        # outside of online_encoder / predictor / projection_heads /
        # sigreg / action_token, all of which are handled separately).
        own_state = dict(self.loss_module.named_buffers())
        for name, val in state["loss_module_buffers"].items():
            if name in own_state:
                own_state[name].data.copy_(val)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["rng_state"])

        # Sampler generator state (C from 4.8 review).
        if state.get("sampler_generator_state") is not None and self.sampler.generator is not None:
            self.sampler.generator.set_state(state["sampler_generator_state"])

        # Data-loader iteration state (C from 4.8 review: without-replacement
        # guarantee). Missing loader_state is a degraded resume that
        # logs and continues; the next batch may re-serve or skip data.
        if state.get("data_loader_state") is not None:
            try:
                self.data_loader.load_state_dict(state["data_loader_state"])
            except (AttributeError, NotImplementedError) as e:
                logger.warning(
                    "Data loader cannot restore iteration state on resume: %s. "
                    "Without-replacement guarantee may be violated.",
                    e,
                )
        else:
            logger.warning(
                "Checkpoint has no data_loader_state; without-replacement "
                "guarantee not restorable.",
            )

        # Time anchors restart from now; do not roll wall-clock backward.
        self.run_start_time = time.monotonic() - state["wall_clock_seconds"]
        self.last_checkpoint_time = time.monotonic()
        logger.info(
            "Resumed from %s at step %d, epoch %d",
            ckpt_path, self.global_step, self.epoch,
        )

    def resume_from_latest(self, ckpt_dir: Optional[Path] = None) -> Path:
        """Try the newest checkpoint first; on load failure, fall back to
        older slots in order until one succeeds. Returns the path that
        loaded, or raises if no checkpoint is usable.

        Per 4.8 review 2026-06-06: a renamed-but-incomplete checkpoint
        from a power-loss event would be the newest by filename, but the
        bytes may be partial. Fallback survives that case.
        """
        if ckpt_dir is None:
            ckpt_dir = self.run_dir / "checkpoints"
        candidates = sorted(ckpt_dir.glob("ckpt_*.pt"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        last_exc: Optional[Exception] = None
        for ckpt_path in candidates:
            try:
                self.resume(ckpt_path)
                return ckpt_path
            except Exception as e:  # noqa: BLE001 -- explicit fallback
                logger.warning(
                    "Failed to load %s (%s); trying next-older slot",
                    ckpt_path, e,
                )
                last_exc = e
        raise RuntimeError(
            f"No usable checkpoint in {ckpt_dir}; last error: {last_exc}"
        )

    # -- Main loop --

    def run(self) -> str:
        """Train until max_epochs or a kill criterion fires.

        Returns one of: "completed", "killed:<reason>", "aborted".
        """
        while self.epoch < self.config.epoch.max_epochs:
            self._start_new_epoch()
            steps_this_epoch = 0
            logger.info("Starting epoch %d", self.epoch)

            while not self._epoch_done():
                modality = self.sampler.sample()
                try:
                    batch = self.data_loader.next_batch(modality)
                except StopIteration:
                    # Loader exhausted this modality. Continue sampling;
                    # the loader is responsible for re-shuffling (v0.5 §3).
                    continue

                step_out = self.train_step(modality, batch)
                # train_step has already advanced both self.global_step
                # and self.modality_step[modality] -- do NOT increment
                # again here (4.8 review 2026-06-06 item A).
                self._update_coverage(modality, batch)

                # Logging fires on this modality's *own* step count, so
                # rare modalities are instrumented on their own cadence
                # rather than on the global step counter (item A).
                m_step = self.modality_step[modality]
                light_due = (
                    m_step % self.config.logging.light_interval_batches == 0
                )
                deep_due = (
                    m_step % self.config.logging.deep_interval_batches == 0
                )
                if light_due or deep_due:
                    record = self._compute_and_log_diagnostics(
                        step_out, light=light_due, deep=deep_due,
                    )
                    self._human_log_line(record)

                # Checkpoint.
                self._checkpoint_if_due()

                # Kill check.
                kill_reason = self._check_kill_criteria(step_out["modality"])
                if kill_reason is not None:
                    logger.error("KILL CRITERION TRIGGERED: %s", kill_reason)
                    self._checkpoint(reason=f"kill:{kill_reason}")
                    return f"killed:{kill_reason}"

                steps_this_epoch += 1

                if (
                    self.config.epoch.max_batches_per_epoch > 0
                    and steps_this_epoch >= self.config.epoch.max_batches_per_epoch
                ):
                    logger.warning(
                        "Safety bound hit: max_batches_per_epoch=%d before coverage anchor",
                        self.config.epoch.max_batches_per_epoch,
                    )
                    break

            self._checkpoint(reason=f"end_of_epoch_{self.epoch}")
            logger.info(
                "End of epoch %d. Coverage: %s",
                self.epoch, self.tokens_consumed,
            )

            # Abort/continue gate at end of epoch 1 (v0.5 §3, §10.4).
            self.epoch += 1
            if (
                self.config.epoch.abort_continue_at_epoch_1
                and self.epoch == 1
            ):
                decision = self._abort_continue_decision()
                if decision == "abort":
                    logger.info("Abort decision at end of epoch 1; stopping.")
                    return "aborted"

        return "completed"

    def _abort_continue_decision(self) -> str:
        """Read Brian's abort/continue decision from a marker file.

        Looks for `continue.marker` or `abort.marker` in the run dir.
        Default: continue (no marker present means Brian hasn't decided
        to stop). The runner writes a `decision_pending.marker` so it's
        visible what state the run is in.
        """
        abrt = self.run_dir / "abort.marker"
        if abrt.exists():
            return "abort"
        # If neither marker exists, default to continue and write a
        # decision_pending.marker so Brian sees the gate happened.
        # Note: internally self.epoch counts the next epoch to be run;
        # the user-facing message uses "first epoch" to avoid the
        # 0-indexed vs 1-indexed confusion 4.8 flagged 2026-06-06.
        pending = self.run_dir / "decision_pending.marker"
        pending.write_text(
            f"End of first epoch reached at step {self.global_step}, "
            f"wall-clock {time.monotonic() - self.run_start_time:.0f}s. "
            f"To abort: touch abort.marker in this directory. "
            f"To continue (default): touch continue.marker, or leave both absent.\n"
        )
        return "continue"
