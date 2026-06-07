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
    """v0.5 §7 thresholds. Pilot-set thresholds derived from M8's own
    early healthy trajectory; warmup_batches determines when criteria
    activate. The first warmup_batches establish baselines (observe-only);
    after warmup the kill criteria are enforced."""

    warmup_batches: int = 5000  # observe-only for first 5K batches
    # Pilot-set thresholds, derived from warmup; defaults are fallbacks
    # in case warmup data is degenerate.
    std_collapse_threshold: float = 0.1  # criterion 1: 5th-pct std floor
    correlation_collapse_threshold: float = 0.95  # criterion 3
    cosine_collapse_threshold: float = 0.99  # criterion 5
    substrate_health_degradation_pct: float = 0.25  # criterion 6: 25%
    substrate_health_window: int = 5  # consecutive checkpoints
    loss_descent_window: int = 5000  # criterion 7: smoothed loss window
    # Sustained-trigger requirements (consecutive checkpoint counts).
    collapse_sustained_checkpoints: int = 3
    dimensional_sustained_checkpoints: int = 5


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
    target_std: torch.Tensor,
) -> dict:
    """v0.5 §5 light metrics, per modality. All computed on the latents
    this modality's loss step produced.

    online_context_latents: [B, ctx_len, D] - online encoder's
        context-only output. The encoder-asymmetry cosine compares this
        against target_latents[:, :ctx_len] (position-aligned, per 4.8
        2026-06-06).
    target_latents: [B, seq_len, D] - target encoder's full output.
    predicted_target: [B, tgt_len, D] - predictor's output (optional;
        the predictor-trivial cosine compares this against
        target_latents[:, ctx_len:]).
    """
    metrics: dict = {}

    # Per-dim std summary (online context-only).
    online_std_sorted = online_std.detach().float().flatten()
    metrics["online_std_p5"] = _percentile(online_std_sorted, 0.05)
    metrics["online_std_p50"] = _percentile(online_std_sorted, 0.50)
    metrics["online_std_p95"] = _percentile(online_std_sorted, 0.95)
    metrics["online_std_below_0.1"] = int((online_std_sorted < 0.1).sum().item())
    metrics["online_std_below_0.5"] = int((online_std_sorted < 0.5).sum().item())

    # Per-dim std summary (target full-sequence).
    target_std_flat = target_std.detach().float().flatten()
    metrics["target_std_p5"] = _percentile(target_std_flat, 0.05)
    metrics["target_std_p50"] = _percentile(target_std_flat, 0.50)
    metrics["target_std_p95"] = _percentile(target_std_flat, 0.95)
    metrics["target_std_below_0.1"] = int((target_std_flat < 0.1).sum().item())

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

    # Encoder-asymmetry cosine: online_context vs target_at_context_positions.
    # Position-aligned per 4.8 2026-06-06 review of jepa_loss.py.
    target_at_context = target_latents[:, :ctx_len, :].detach().float()
    cos_asym = F.cosine_similarity(
        online_context_latents.detach().float(),
        target_at_context,
        dim=-1,
    )  # [B, ctx_len]
    metrics["encoder_asymmetry_cosine_mean"] = float(cos_asym.mean().item())
    metrics["encoder_asymmetry_cosine_std"] = float(cos_asym.std().item())

    # Predictor-trivial cosine (if predicted_target provided).
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
    """Rolling per-modality metric history for kill-criteria evaluation."""

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
        config_dict["loss"] = {
            "invariance_weight": self.loss_module.invariance_weight,
            "variance_weight": self.loss_module.variance_weight,
            "covariance_weight": self.loss_module.covariance_weight,
            "variance_target": self.loss_module.variance_target,
            "ema_momentum": self.loss_module.ema_momentum,
            "std_ema_momentum": self.loss_module.std_ema_momentum,
            "std_ema_floor": self.loss_module.std_ema_floor,
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

        # EMA target update (after optimizer step so online has just been
        # updated and the EMA tracks the latest slow params).
        self.loss_module.update_target_ema()

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
            "l_var": float(raw["l_var"].item()),
            "l_cov": float(raw["l_cov"].item()),
            "tokens_consumed": dict(self.tokens_consumed),
            "elapsed_seconds": time.monotonic() - self.run_start_time,
        }

        if light:
            light_m = _light_collapse_metrics(
                online_context_latents=raw["online_context_latents"],
                target_latents=raw["target_latents"],
                # JEPALoss returns predicted_target detached (no graph
                # retention), so the predictor-trivial cosine is now live
                # in the light metrics (kill-5's second axis -- 4.8 review
                # 2026-06-06 #9 cheap-win 2026-06-07).
                predicted_target=raw.get("predicted_target"),
                ctx_len=raw["ctx_len"],
                online_std=raw["online_std"],
                target_std=raw["target_std"],
            )
            record["light"] = light_m
            self.history.push(modality, light_m)

        if deep:
            deep_m = _deep_collapse_metrics(raw["online_context_latents"])
            record["deep"] = deep_m
            self.history.push(modality, deep_m)

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
            f"L_var={record['l_var']:.4f} "
            f"L_cov={record['l_cov']:.4f} "
            f"std_p5={light.get('online_std_p5', float('nan')):.4f} "
            f"cos_enc={light.get('encoder_asymmetry_cosine_mean', float('nan')):.4f} "
            f"elapsed={elapsed_h:.2f}h"
        )
        logger.info(msg)
        with open(self.human_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

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
        recent_std = self.history.recent(
            modality, "online_std_p5", cfg.collapse_sustained_checkpoints,
        )
        if len(recent_std) >= cfg.collapse_sustained_checkpoints and all(
            v < cfg.std_collapse_threshold for v in recent_std
        ):
            return (
                f"kill-1 (complete collapse) on {modality}: "
                f"std_p5 < {cfg.std_collapse_threshold} for "
                f"{cfg.collapse_sustained_checkpoints} checkpoints"
            )

        # Criterion 3: dimensional collapse (off-diagonal correlation).
        recent_corr = self.history.recent(
            modality,
            "mean_abs_off_diag_correlation",
            cfg.dimensional_sustained_checkpoints,
        )
        if len(recent_corr) >= cfg.dimensional_sustained_checkpoints and all(
            v > cfg.correlation_collapse_threshold for v in recent_corr
        ):
            return (
                f"kill-3 (correlation collapse) on {modality}: "
                f"mean_abs_off_diag > {cfg.correlation_collapse_threshold} for "
                f"{cfg.dimensional_sustained_checkpoints} checkpoints"
            )

        # Criterion 5: encoder-asymmetry cosine -- online and target
        # encoders collapsed to the same representation.
        recent_cos = self.history.recent(
            modality,
            "encoder_asymmetry_cosine_mean",
            cfg.dimensional_sustained_checkpoints,
        )
        if len(recent_cos) >= cfg.dimensional_sustained_checkpoints and all(
            v > cfg.cosine_collapse_threshold for v in recent_cos
        ):
            return (
                f"kill-5 (encoder asymmetry lost) on {modality}: "
                f"online-vs-target cosine > {cfg.cosine_collapse_threshold} for "
                f"{cfg.dimensional_sustained_checkpoints} checkpoints"
            )

        # Criterion 5 (second axis, added 2026-06-07): predictor-trivial
        # cosine -- predictor learned the identity / target representation
        # without learning to predict it. Distinct from encoder-asymmetry
        # because the encoders can be diverged but the predictor still
        # trivial; both indicate kill-5 family failures (4.8 review
        # 2026-06-06 #9 cheap-win).
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
            "online_state_dict": self.loss_module.online_encoder.state_dict(),
            "target_state_dict": self.loss_module.target_encoder.state_dict(),
            "target_buffer_snapshots": self.loss_module._target_buffer_snapshots,
            "predictor_state_dict": self.loss_module.predictor.state_dict(),
            "loss_module_buffers": {
                name: buf.detach().clone()
                for name, buf in self.loss_module.named_buffers()
                if not name.startswith(("online_encoder.", "target_encoder.", "predictor."))
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
        state = torch.load(ckpt_path, map_location="cpu")
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
        self.loss_module.online_encoder.load_state_dict(state["online_state_dict"])
        self.loss_module.target_encoder.load_state_dict(state["target_state_dict"])
        self.loss_module._target_buffer_snapshots = state["target_buffer_snapshots"]
        self.loss_module.predictor.load_state_dict(state["predictor_state_dict"])
        # Loss-module's own buffers (action_token, *_target_std_ema).
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
