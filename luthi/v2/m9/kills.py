"""M9 step-1 kill criteria (spec §6 + plan §13).

The kills active at step 1 (per spec §6):

  K-M9-2  MCTS pathology: visit-distribution entropy collapses to a
          single dominant branch, OR tree-consistency diverges.
  K-M9-3  Value divergence: V(s) grows unbounded or oscillates.
  K-M9-4  gamma divergence: clamp-then-halt (clamp to last-healthy,
          halt if recovery fails). The GammaInference module owns the
          band; this kill is the trigger and recovery driver.
  K-M9-5  Dark-room / catatonia: sustained internal AND external
          stasis with no external cause. The drift-independent
          preference-drift backstop paired with the P1 soft floor.
  K-M9-7  Staleness runaway: drift accumulator persistently outruns
          refresh capacity (planning can't keep up with theta). Reads
          the StalenessManager's failover / consistency state.
  K-M9-8  Mask-stability: self/world gating-mask coordinates moving
          past a trending band. Mask drift makes the action's
          effective dimensionality non-stationary, compounding Seam C
          beyond what the staleness machinery can absorb.

K-M9-1 (epistemic degeneracy) and K-M9-6 (MI-probe collapse-as-guard)
are step-2+ scope (gated by beta_epi > 0).

Architecture. One `KillRegistry` instance owns named watchers; each
watcher holds its own DriftBand or counter state. State machine per
watcher: HEALTHY -> FLAGGED (on breach) -> FIRED (on sustained
breach) -> RECOVERED (when the signal returns to band). Two-stage
matches the spec's pattern (gamma clamp-then-halt; the M8 trending
kills' flag-then-fire).

The registry does *not* actually halt the run. It reports state; the
loop decides what to do with FIRED. Step-1 policy (loop integration,
next slice): K-M9-2/3/4/5/7/8 in FIRED -> emit a halt event; the
runner inspects and decides whether to clamp + continue or halt.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn


class KillState(str, Enum):
    HEALTHY = "healthy"
    FLAGGED = "flagged"
    FIRED = "fired"


@dataclass
class TrendingBand:
    """Outlier-robust running band; same shape as the M8 72526cb
    machinery. Median + MAD; reuses the StalenessManager's DriftBand
    semantics in a smaller scope so kills can own private bands.
    """

    window: int = 32
    direction: str = "both"  # "max" (rise past), "min" (fall below), "both"
    k: float = 4.0
    sustained_cycles: int = 3
    min_warmup: int = 8

    values: deque = field(default_factory=lambda: deque())
    consecutive_breaches: int = 0

    def __post_init__(self):
        self.values = deque(maxlen=self.window)

    def observe(self, x: float) -> str:
        """Push value; return one of 'healthy', 'flagged', 'fired'."""
        self.values.append(float(x))
        if len(self.values) < self.min_warmup:
            return KillState.HEALTHY
        vals = torch.tensor(list(self.values))
        med = float(vals.median().item())
        mad = float((vals - vals.median()).abs().median().item())
        mad = max(mad, 1e-8)
        latest = float(self.values[-1])

        is_breach = False
        if self.direction in ("max", "both"):
            if latest > med + self.k * mad:
                is_breach = True
        if self.direction in ("min", "both"):
            if latest < med - self.k * mad:
                is_breach = True

        if is_breach:
            self.consecutive_breaches += 1
        else:
            self.consecutive_breaches = 0

        if self.consecutive_breaches >= self.sustained_cycles:
            return KillState.FIRED
        elif self.consecutive_breaches > 0:
            return KillState.FLAGGED
        else:
            return KillState.HEALTHY


class KillRegistry:
    """All M9 step-1 kill criteria in one place.

    The registry maintains state per kill; the loop calls
    `observe_*` each cycle with the appropriate signal, then queries
    `states()` for the per-kill state. `fired()` returns the set of
    kills currently in FIRED.

    Step-1 policy: FIRED is a halt request, not an automatic halt.
    The runner emits a halt event and decides whether to clamp +
    continue or halt the run.
    """

    def __init__(
        self,
        # Per-kill thresholds, all pilot-set.
        entropy_min_warmup: int = 8,
        entropy_low_floor: float = 0.5,  # bits; below = single dominant branch
        entropy_sustained: int = 5,
        consistency_max: float = 2.0,
        consistency_sustained: int = 5,
        value_band_k: float = 6.0,
        value_sustained: int = 4,
        gamma_runaway_k: float = 4.0,
        gamma_sustained: int = 4,
        darkroom_internal_threshold: float = 1e-3,
        darkroom_sustained_cycles: int = 30,
        staleness_failover_sustained: int = 8,
        mask_band_k: float = 4.0,
        mask_sustained: int = 4,
    ):
        # K-M9-2 entropy: fires when entropy falls below floor for
        # sustained cycles. Not a trending band -- the floor is absolute,
        # because below-floor visit collapse is pathological regardless
        # of history.
        self.entropy_low_floor = entropy_low_floor
        self.entropy_sustained = entropy_sustained
        self.entropy_min_warmup = entropy_min_warmup
        self._entropy_count = 0
        self._entropy_state = KillState.HEALTHY
        self._entropy_observations = 0

        # K-M9-2 consistency: fires when re-eval/cached deviation
        # exceeds a threshold for sustained cycles.
        self.consistency_max = consistency_max
        self.consistency_sustained = consistency_sustained
        self._consistency_count = 0
        self._consistency_state = KillState.HEALTHY

        # K-M9-3 value divergence: trending band on V(s) running
        # estimates. Direction = both (oscillation or runaway).
        self._value_band = TrendingBand(
            window=32, direction="both", k=value_band_k,
            sustained_cycles=value_sustained, min_warmup=8,
        )
        self._value_state = KillState.HEALTHY

        # K-M9-4 gamma divergence: trending band on gamma. Direction
        # = both (rigidity at high, indecision at low).
        self._gamma_band = TrendingBand(
            window=32, direction="both", k=gamma_runaway_k,
            sustained_cycles=gamma_sustained, min_warmup=8,
        )
        self._gamma_state = KillState.HEALTHY

        # K-M9-5 dark room: internal-state-change magnitude below
        # threshold AND external-stasis flag (caller provides) for
        # sustained_cycles.
        self.darkroom_internal_threshold = darkroom_internal_threshold
        self.darkroom_sustained_cycles = darkroom_sustained_cycles
        self._darkroom_consecutive = 0
        self._darkroom_state = KillState.HEALTHY
        # F4 (2026-06-11): the armed-state log -- mandatory per
        # 4.8's gate-repairs spec. Per cycle we record whether the
        # kill was CAPABLE of firing given current bands. A disarmed
        # safety backstop is visible here, not silent.
        self._darkroom_armed_history: deque = deque(maxlen=128)
        self._darkroom_disarmed_consecutive = 0
        # If disarmed for `darkroom_disarmed_window` consecutive cycles
        # while operation continues, that itself is a defect flag.
        self.darkroom_disarmed_window = darkroom_sustained_cycles

        # K-M9-7 staleness runaway: drives off the StalenessManager's
        # `in_failover()` state. Fires after sustained failover.
        self.staleness_failover_sustained = staleness_failover_sustained
        self._staleness_failover_count = 0
        self._staleness_state = KillState.HEALTHY

        # K-M9-8 mask stability: trending band on per-dim mask
        # coordinates. Fires when the mask drifts past band.
        self._mask_band = TrendingBand(
            window=32, direction="both", k=mask_band_k,
            sustained_cycles=mask_sustained, min_warmup=8,
        )
        self._mask_state = KillState.HEALTHY

    # ------------------------------------------------------------------
    # Per-cycle observation methods.
    # ------------------------------------------------------------------
    def observe_mcts_entropy(self, visit_distribution: torch.Tensor) -> None:
        """K-M9-2 entropy axis."""
        self._entropy_observations += 1
        if len(visit_distribution) == 0:
            return
        p = visit_distribution.clamp(min=1e-12)
        entropy = float((-p * p.log()).sum().item())
        # log -> bits if we used log2; we use natural log here for
        # numerical convenience. Convert to bits for the floor compare:
        entropy_bits = entropy / math.log(2.0)
        if self._entropy_observations < self.entropy_min_warmup:
            return
        if entropy_bits < self.entropy_low_floor:
            self._entropy_count += 1
        else:
            self._entropy_count = 0
        if self._entropy_count >= self.entropy_sustained:
            self._entropy_state = KillState.FIRED
        elif self._entropy_count > 0:
            self._entropy_state = KillState.FLAGGED
        else:
            self._entropy_state = KillState.HEALTHY

    def observe_mcts_consistency(self, median_deviation: float) -> None:
        """K-M9-2 tree-consistency axis. `median_deviation` is the
        re-eval-vs-cached signal from StalenessManager.reevaluate().
        """
        if median_deviation > self.consistency_max:
            self._consistency_count += 1
        else:
            self._consistency_count = 0
        if self._consistency_count >= self.consistency_sustained:
            self._consistency_state = KillState.FIRED
        elif self._consistency_count > 0:
            self._consistency_state = KillState.FLAGGED
        else:
            self._consistency_state = KillState.HEALTHY

    def observe_value(self, v_estimate: float) -> None:
        """K-M9-3 value divergence."""
        s = self._value_band.observe(float(v_estimate))
        self._value_state = KillState(s)

    def observe_gamma(self, gamma_value: float) -> None:
        """K-M9-4 gamma divergence. (Separate from GammaInference's
        own clamp -- this kill watches the *trend* of gamma over time,
        so a slow drift to the band edge fires even if no single
        cycle is clamped.)
        """
        s = self._gamma_band.observe(float(gamma_value))
        self._gamma_state = KillState(s)

    def observe_darkroom(
        self,
        internal_change_magnitude: float,
        external_stasis: bool,
    ) -> None:
        """K-M9-5 dark room: both internal AND external stasis (legacy).

        **Legacy path** -- this signature reads `external_stasis` from
        whatever the caller provided (originally the sigmoid intensity
        head path, which Fable's probe_d showed was disarmed in ~90%
        of random inits). The F4 fix path is `observe_darkroom_v2`
        below, which consumes the §A band-based signals + an explicit
        armed flag. Existing callers and the test suite use this
        signature; new code MUST use v2.

        `internal_change_magnitude` = ||Delta s|| over latent dims.
        `external_stasis` = caller-derived flag.
        Armed-state is assumed True (legacy).
        """
        self._darkroom_armed_history.append(True)
        self._darkroom_disarmed_consecutive = 0
        is_stasis = (
            internal_change_magnitude < self.darkroom_internal_threshold
            and external_stasis
        )
        self._advance_darkroom_state(is_stasis)

    def observe_darkroom_v2(
        self,
        internal_silent: bool,
        external_silent: bool,
        is_armed: bool = True,
    ) -> None:
        """F4 fix path: K-M9-5 from §A band-based silent predicates.

        `internal_silent`: True iff the cycle's ‖Δs_internal‖ is below
                          DeltaSBand.silent_threshold() (the §A.2
                          internal-stasis signal). The loop computes
                          this once and fans the same value to P1's
                          engagement_cost_from_delta_s.
        `external_silent`: True iff ActivityBands.external_stasis(...)
                          is True for this cycle's decoder activities
                          (the §A.1 signal; raw pre-sigmoid magnitudes
                          vs running band, NOT intensity heads).
        `is_armed`: True iff both bands are warm enough that the silent
                    predicates above are meaningful. When False, the
                    kill cannot fire this cycle and the armed-state
                    log records the disarmed cycle; sustained disarmed
                    operation is itself a flag.

        Armed by construction at any decoder init -- the F4 fix
        property. probe_d goes from 26/256 armable to 256/256.
        """
        self._darkroom_armed_history.append(bool(is_armed))
        if not is_armed:
            self._darkroom_disarmed_consecutive += 1
            # State left at its previous value -- a disarmed cycle is
            # not "healthy" or "fired", it is uninformed. The
            # k_m9_5_disarmed_sustained() flag below is the escalation.
            return
        self._darkroom_disarmed_consecutive = 0
        is_stasis = bool(internal_silent and external_silent)
        self._advance_darkroom_state(is_stasis)

    def _advance_darkroom_state(self, is_stasis: bool) -> None:
        """Shared state-machine step for the dark-room kill."""
        if is_stasis:
            self._darkroom_consecutive += 1
        else:
            self._darkroom_consecutive = 0
        if self._darkroom_consecutive >= self.darkroom_sustained_cycles:
            self._darkroom_state = KillState.FIRED
        elif self._darkroom_consecutive > 0:
            self._darkroom_state = KillState.FLAGGED
        else:
            self._darkroom_state = KillState.HEALTHY

    # ------------------------------------------------------------------
    # F4 armed-state instrumentation (mandatory per 4.8's spec).
    # ------------------------------------------------------------------
    def darkroom_armed_now(self) -> bool:
        """True iff the most recent observe_darkroom_v2 call reported
        the kill as armed (bands warm). The loop logs this per cycle
        so a disarmed safety backstop is visible, not silent.
        """
        return bool(self._darkroom_armed_history[-1]) if self._darkroom_armed_history else False

    def darkroom_armed_fraction(self) -> float:
        """Fraction of recent cycles in which the kill was armed.
        Sustained low fraction is a defect: a safety backstop that's
        off is not a neutral state.
        """
        if not self._darkroom_armed_history:
            return 0.0
        return sum(self._darkroom_armed_history) / len(self._darkroom_armed_history)

    def k_m9_5_disarmed_sustained(self) -> bool:
        """True if K-M9-5 has been disarmed for `darkroom_disarmed_window`
        consecutive cycles -- that itself is a flag per 4.8's
        escalation rule.
        """
        return self._darkroom_disarmed_consecutive >= self.darkroom_disarmed_window

    def observe_staleness(self, in_failover: bool) -> None:
        """K-M9-7 staleness runaway. Fires after sustained failover."""
        if in_failover:
            self._staleness_failover_count += 1
        else:
            self._staleness_failover_count = 0
        if self._staleness_failover_count >= self.staleness_failover_sustained:
            self._staleness_state = KillState.FIRED
        elif self._staleness_failover_count > 0:
            self._staleness_state = KillState.FLAGGED
        else:
            self._staleness_state = KillState.HEALTHY

    def observe_mask(self, mask_norm: float) -> None:
        """K-M9-8 mask stability. `mask_norm` is a scalar summary of
        the self_world_mask state at this cycle (e.g. the L2 norm or
        the mean sigmoid'd value).
        """
        s = self._mask_band.observe(float(mask_norm))
        self._mask_state = KillState(s)

    # ------------------------------------------------------------------
    # Query.
    # ------------------------------------------------------------------
    def states(self) -> dict:
        """Per-kill current state. Step-1 active set only."""
        return {
            "K-M9-2-entropy": self._entropy_state,
            "K-M9-2-consistency": self._consistency_state,
            "K-M9-3-value": self._value_state,
            "K-M9-4-gamma": self._gamma_state,
            "K-M9-5-darkroom": self._darkroom_state,
            "K-M9-7-staleness": self._staleness_state,
            "K-M9-8-mask": self._mask_state,
        }

    def fired(self) -> set:
        """Kills currently in FIRED. The loop's halt-decision input."""
        return {name for name, s in self.states().items() if s == KillState.FIRED}

    def flagged(self) -> set:
        return {name for name, s in self.states().items() if s == KillState.FLAGGED}

    def reset(self) -> None:
        """Reset all kill counters and states. Used after a clamped
        recovery (e.g. K-M9-4's clamp-to-last-healthy) so the
        watcher doesn't immediately re-fire on stale state.
        """
        self._entropy_count = 0
        self._consistency_count = 0
        self._darkroom_consecutive = 0
        self._darkroom_armed_history = deque(maxlen=128)
        self._darkroom_disarmed_consecutive = 0
        self._staleness_failover_count = 0
        self._value_band = TrendingBand(
            window=self._value_band.window,
            direction=self._value_band.direction,
            k=self._value_band.k,
            sustained_cycles=self._value_band.sustained_cycles,
            min_warmup=self._value_band.min_warmup,
        )
        self._gamma_band = TrendingBand(
            window=self._gamma_band.window,
            direction=self._gamma_band.direction,
            k=self._gamma_band.k,
            sustained_cycles=self._gamma_band.sustained_cycles,
            min_warmup=self._gamma_band.min_warmup,
        )
        self._mask_band = TrendingBand(
            window=self._mask_band.window,
            direction=self._mask_band.direction,
            k=self._mask_band.k,
            sustained_cycles=self._mask_band.sustained_cycles,
            min_warmup=self._mask_band.min_warmup,
        )
        self._entropy_state = KillState.HEALTHY
        self._consistency_state = KillState.HEALTHY
        self._value_state = KillState.HEALTHY
        self._gamma_state = KillState.HEALTHY
        self._darkroom_state = KillState.HEALTHY
        self._staleness_state = KillState.HEALTHY
        self._mask_state = KillState.HEALTHY
