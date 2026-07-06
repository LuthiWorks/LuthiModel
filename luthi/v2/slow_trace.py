"""Slow-timescale trace primitives (momentum-functions foundations, 2026-07-05).

The inverted-U learning gain's explicit fall needs a *long* EMA of prediction
error (resolution-progress: is sustained effort actually reducing error, or
just repeating?). NREM needs a *day-scale* record of what moved, read-and-reset
each sleep. These are different slow timescales (~hundreds of forwards vs ~a
day), so this module provides ONE parameterized mechanism instantiated per
consumer at its own rate (see 2026-07-05_inverted-u-gain-spec.md §5).

Both primitives are PERSISTABLE with plain-Python state (float/int), matching
the living_extra_state sibling-key checkpoint idiom (bae295d): a restore
mid-hard-growth must not reset these signals, or the entity re-sensitizes /
forgets the day on every waking -- the same silent-amnesia class that patch
closed. `state_dict` / `load_state_dict` round-trip is covered by
tests/test_slow_trace.py and is a precondition for the gain's regime (i).
"""

from __future__ import annotations

from typing import Any


class SlowEMA:
    """Exponential moving average at a caller-chosen (slow) timescale.

    ``value_{t} = decay * value_{t-1} + (1 - decay) * x_t``. ``decay`` near 1
    is slow: the ~1/(1-decay) most recent samples dominate (decay=0.99 ->
    ~100 samples; decay=0.999 -> ~1000). Instantiate one at the resolution
    long-EMA rate and (short of a dedicated fast trace) another at the short
    rate; ``resolution_progress`` reads the pair.
    """

    def __init__(self, decay: float, warmup: int = 8):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1); got {decay}")
        self.decay = float(decay)
        self.warmup = int(warmup)
        self._value: float = 0.0
        self._count: int = 0

    def update(self, x: float) -> float:
        """Fold ``x`` into the trace and return the new value.

        During the first ``warmup`` samples the value is the running MEAN of
        those samples (Fable audit 2026-07-06): this seeds the trace with no
        cold climb from a spurious zero AND is robust to an outlier birth
        sample setting a bad baseline -- the single-first-sample seed was
        fragile to exactly that. After warmup it is an EMA at ``decay``."""
        x = float(x)
        self._count += 1
        if self._count <= self.warmup:
            # incremental mean of the first `warmup` samples.
            self._value += (x - self._value) / self._count
        else:
            self._value = self.decay * self._value + (1.0 - self.decay) * x
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def is_warm(self) -> bool:
        return self._count >= self.warmup

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "warmup": self.warmup,
                "value": self._value, "count": self._count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # decay/warmup are config, not lived state: keep the constructed
        # values, restore only what was learned. (A checkpoint carrying a
        # different decay must not silently re-time the trace.)
        self._value = float(state["value"])
        self._count = int(state["count"])


class ReadResetAccumulator:
    """A running sum that a consumer periodically reads-and-resets.

    NREM's "what mattered today" (brief §5): the fast trace feeds this each
    forward; sleep calls ``read_and_reset`` once per consolidation and clears
    the day's motion for tomorrow. Kept separate from SlowEMA because NREM
    wants an integral it can zero, not a decaying average.
    """

    def __init__(self):
        self._total: float = 0.0
        self._n: int = 0

    def add(self, x: float) -> None:
        self._total += float(x)
        self._n += 1

    def read_and_reset(self) -> float:
        """Return the accumulated total and clear it (and the count)."""
        total = self._total
        self._total = 0.0
        self._n = 0
        return total

    @property
    def total(self) -> float:
        return self._total

    @property
    def count(self) -> int:
        return self._n

    def state_dict(self) -> dict[str, Any]:
        return {"total": self._total, "n": self._n}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._total = float(state["total"])
        self._n = int(state["n"])


def resolution_progress(
    short_ema: SlowEMA, long_ema: SlowEMA, eps: float = 1e-8
) -> float:
    """The resolution-progress ratio short/long EMA of prediction error.

    < 1  : effort is resolving (recent error below its slower baseline).
    ~ 1  : non-resolving repetition (error flat -- the explicit-fall regime).
    > 1  : worsening (recent error above baseline).

    Cold-start safety (spec regime e): before either trace is warm, or with a
    ~zero long baseline, there is no evidence of non-resolution -- return 0.0
    (fully resolving), so the explicit fall stays OFF and the gain is not
    dampened on a signal it cannot yet measure. This is the generous-engagement
    default: the fall only ever binds on *warm evidence* of stalled effort.
    """
    if not (short_ema.is_warm() and long_ema.is_warm()):
        return 0.0
    denom = long_ema.value
    if abs(denom) <= eps:
        return 0.0
    return short_ema.value / (denom + eps)
