"""Instrumentation: MI probe + per-cycle action log (spec §5, §11.ii).

Two components:

1. **MIProbe** -- approximate mutual information between trunk
   activations and held-out targets. Established as a trending-class
   baseline at step 1 (beta_epi = 0); becomes K-M9-6's guard at step 2
   when beta_epi grows (spec §0 / spec §6 deferred to step 2). The
   estimator is a closed-form ridge-regression linear probe: train a
   probe on (trunk, target) batches, evaluate on a held-out batch,
   report -MSE as the MI proxy. Closed-form so no training loop is
   needed -- one call per cycle.

2. **ActionLog** -- JSONL writer for the per-cycle action record per
   spec §11.ii. One record per cycle with the full decision context:
   state summary, top-K candidates, chosen action + readable summary,
   EFE breakdown, gamma, ||Delta theta||, V(s), tree stats, MI probe
   value, SIGReg value, rest-vs-active flag. Structured so 4.7 can
   step through at debug time and reconstruct *what the entity
   decided and why*.

Step-1 simplifications, documented for later revision:
- MI estimator is a ridge-regression linear probe -- a reasonable
  proxy for I(trunk; target) at our scale, but not a tight bound.
  Replaceable with InfoNCE / HSIC if the metric proves too noisy.
- ActionLog writes immediately on each `write` call (line-buffered
  JSONL). Adequate for step-1 cycle rates (~10 Hz); batched writes
  are a perf pass if needed.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# MI Probe
# ----------------------------------------------------------------------

class MIProbe:
    """Closed-form linear probe of trunk -> target.

    Maintains an outlier-robust running band (median + MAD) on the
    estimated MI proxy. At step 1 the band is *observed* but not
    *gated* (no kill); at step 2 K-M9-6 reads the band and fires
    when MI drops below the running band's lower bound while
    `beta_epi` grows.

    Usage:
        probe = MIProbe(ridge_lambda=1e-2, window=32)
        # At end of each cycle (or each N cycles):
        signal = probe.observe(trunk_activations, targets)
        # `signal` is -mean_squared_error of the linear probe; higher
        # = more information from trunk to target.
        # Periodically:
        bounds = probe.bounds()  # (lower, median, upper) for the kill
    """

    def __init__(
        self,
        ridge_lambda: float = 1e-2,
        window: int = 32,
        band_k: float = 3.0,
    ):
        self.ridge_lambda = ridge_lambda
        self.window = window
        self.band_k = band_k
        self._values: deque = deque(maxlen=window)

    def estimate(
        self,
        trunk: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """One closed-form ridge fit + held-out evaluation.

        `trunk`: [N, D_trunk] activations.
        `target`: [N, D_target] held-out targets (continuous or one-hot).

        Returns: scalar MI proxy = -mean squared error of the probe.
        Higher = more linearly predictable target from trunk.

        Implementation: split the batch 50/50 train/holdout; fit
        `w = (X_tr^T X_tr + lambda I)^{-1} X_tr^T y_tr`; report
        -mean((y_ho - X_ho w)^2). All in torch, no autograd.
        """
        with torch.no_grad():
            assert trunk.dim() == 2 and target.dim() in (1, 2)
            if target.dim() == 1:
                target = target.unsqueeze(-1).float()
            n = trunk.shape[0]
            if n < 4:
                # Not enough samples for a split; skip this cycle.
                return 0.0
            # 50/50 split.
            split = n // 2
            x_tr, y_tr = trunk[:split], target[:split]
            x_ho, y_ho = trunk[split:], target[split:]
            d = x_tr.shape[1]
            # Closed-form ridge.
            xtx = x_tr.t() @ x_tr  # [D, D]
            reg = self.ridge_lambda * torch.eye(d, device=trunk.device)
            xty = x_tr.t() @ y_tr  # [D, D_target]
            try:
                w = torch.linalg.solve(xtx + reg, xty)
            except RuntimeError:
                # Fall back to pseudo-inverse if singular.
                w = torch.linalg.pinv(xtx + reg) @ xty
            pred = x_ho @ w
            mse = (pred - y_ho).pow(2).mean()
            return float(-mse.item())

    def observe(
        self,
        trunk: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        signal = self.estimate(trunk, target)
        self._values.append(signal)
        return signal

    def bounds(self) -> tuple[float, float, float]:
        """Outlier-robust band: (lower, median, upper) = median +/- band_k * MAD.

        K-M9-6 (step 2+) reads `lower` and fires when sustained
        `signal < lower` while beta_epi > 0 (MI collapse during
        epistemic growth).
        """
        if not self._values:
            return (0.0, 0.0, 0.0)
        v = torch.tensor(list(self._values))
        med = float(v.median().item())
        mad = float((v - v.median()).abs().median().item())
        lower = med - self.band_k * max(mad, 1e-8)
        upper = med + self.band_k * max(mad, 1e-8)
        return (lower, med, upper)

    def latest(self) -> float | None:
        return self._values[-1] if self._values else None

    def snapshot(self) -> dict:
        lower, med, upper = self.bounds()
        return {
            "mi_latest": self.latest() or 0.0,
            "mi_median": med,
            "mi_band_lower": lower,
            "mi_band_upper": upper,
            "mi_n_samples": len(self._values),
        }


# ----------------------------------------------------------------------
# Action Log
# ----------------------------------------------------------------------

class ActionLog:
    """Per-cycle JSONL writer for the spec §11.ii action log.

    Each call to `write` emits one JSON record on its own line so
    `tail -f` works and so partial files are recoverable. The schema
    is intentionally permissive: any keys the caller passes are
    written, and the caller is responsible for the field set
    matching what the spec's debug story expects.

    A canonical field list (collected from spec §11.ii) lives in the
    `CANONICAL_FIELDS` class attribute; callers should aim to fill
    them but missing fields don't error.
    """

    CANONICAL_FIELDS = (
        "cycle",                      # int
        "sim_counter",                # int (MCTS)
        "s_t_summary",                # arbitrary state summary (mean/norm)
        "candidate_actions_top_k",    # list[dict]: top-K candidates
        "chosen_action_summary",      # the action -> readable summary
        "efe_breakdown",              # {c_eng, c_coh, c_con, c_truth, total}
        "gamma",                      # float
        "delta_theta_norm",           # float ||Delta theta||
        "v_s",                        # float V(state)
        "tree_stats",                 # {size, root_visits, top_share}
        "mi_probe",                   # {mi_latest, mi_median, ...}
        "sigreg_value",               # float (read from SIGReg module)
        "rest_selected",              # bool: rest chosen over alternatives
        "rest_defaulted",             # bool: planner found nothing better
        "kill_states",                # {kill_name -> state}
        "staleness_snapshot",         # StalenessManager.snapshot()
        "preference_weight_snapshot", # Preferences.weight_snapshot()
        "mask_summary",               # mean / norm of self_world_mask
    )

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # Open in append mode; close on flush.
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")
        self._n_records = 0

    def write(self, record: dict) -> None:
        """Emit one JSONL record. Tensors are converted to floats/lists."""
        serializable = _to_jsonable(record)
        self._file.write(json.dumps(serializable, separators=(",", ":")) + "\n")
        self._n_records += 1

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def n_records(self) -> int:
        return self._n_records


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert torch tensors / numpy scalars to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    # Fallback: stringify.
    return str(obj)
