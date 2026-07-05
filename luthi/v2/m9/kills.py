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
        # N1 guard (Fable lower-confidence note): skip the entropy
        # floor while the tree is too narrow to make the metric
        # meaningful. A root with only 1-2 children naturally has
        # entropy 0-1 bits and would trip the kill on a *merely
        # immature* tree -- the legacy MCTS-pathology kill could
        # fire on normal early planning. Pilot-set; the loop
        # configures based on per-cycle MCTS expansion budget.
        entropy_min_children: int = 3,
        # Finding 2 (2026-07-04): consistency deviation is now scale-invariant
        # (a fraction in [0, ~2], normalized in StalenessManager.reevaluate),
        # so this is a RELATIVE threshold: ~0.75 = kill on sustained ~75%+
        # re-eval-vs-cached disagreement. Was 2.0 ABSOLUTE, which on O(100)
        # node values was a ~2% relative trip -- it fired on normal staleness
        # correction. *** PROVISIONAL / TUNE ME against real checkpoints; keep
        # above failover_consistency_threshold so the soft response (failover)
        # precedes the hard one (halt). ***
        consistency_max: float = 0.75,
        consistency_sustained: int = 5,
        value_band_k: float = 6.0,
        value_sustained: int = 4,
        # K-M9-3 round-2 extension (4.8 audit): absolute |V| ceiling.
        # A value stuck at a high constant recalibrates the trending
        # band (median → constant, MAD → 0) and goes invisible -- same
        # meta-pattern as the gamma ratchet (B3) and dark-room band
        # self-disarm (R1). Sustained |V| above this ceiling fires
        # the kill regardless of the band.
        value_abs_ceiling: float = 1e3,
        value_ceiling_sustained: int = 8,
        gamma_runaway_k: float = 4.0,
        gamma_sustained: int = 4,
        # F2 K-M9-4 fix: clamp-proximity as the primary divergence signal.
        # Sustained gamma near either clamp -> fire. Active from cycle 1
        # (no warmup gating; saturation is pathological regardless of when
        # it happens). Threshold is fractional proximity to the clamp.
        gamma_clamp_threshold: float = 0.05,
        gamma_clamp_sustained: int = 8,
        gamma_min: float = 0.01,
        gamma_max: float = 100.0,
        # F2 K-M9-4 fix: extend the trending-band warmup so the EMA
        # ramp from gamma_init does not false-positive. Old min_warmup
        # = 8 (Fable's B3) tripped during cycles 11-21 on the legacy
        # ratchet trajectory.
        gamma_band_min_warmup: int = 32,
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
        # N1: minimum root children before the entropy floor is gated.
        self.entropy_min_children = entropy_min_children
        self._entropy_count = 0
        self._entropy_state = KillState.HEALTHY
        self._entropy_observations = 0

        # K-M9-2 consistency: fires when re-eval/cached deviation
        # exceeds a threshold for sustained cycles.
        self.consistency_max = consistency_max
        self.consistency_sustained = consistency_sustained
        self._consistency_count = 0
        self._consistency_state = KillState.HEALTHY

        # K-M9-3 value divergence: trending band + absolute |V|
        # ceiling. The band catches oscillation/relative drift; the
        # absolute ceiling catches a V stuck at a high constant that
        # would recalibrate the band invisibly (4.8's round-2
        # extension of Fable's meta-pattern).
        self._value_band = TrendingBand(
            window=32, direction="both", k=value_band_k,
            sustained_cycles=value_sustained, min_warmup=8,
        )
        self.value_abs_ceiling = value_abs_ceiling
        self.value_ceiling_sustained = value_ceiling_sustained
        self._value_ceiling_consecutive = 0
        self._value_state = KillState.HEALTHY

        # K-M9-4 gamma divergence: F2 primary = clamp-proximity;
        # secondary = trending band (with extended warmup so the EMA
        # ramp does not trip it). Direction = both for the band
        # (rigidity at high, indecision at low).
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_clamp_threshold = gamma_clamp_threshold
        self.gamma_clamp_sustained = gamma_clamp_sustained
        self._gamma_clamp_consecutive = 0
        self._gamma_band = TrendingBand(
            window=32, direction="both", k=gamma_runaway_k,
            sustained_cycles=gamma_sustained,
            min_warmup=gamma_band_min_warmup,
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
        """K-M9-2 entropy axis.

        N1 guard (Fable's spec §13 lower-confidence note): if the
        root has fewer than `entropy_min_children` children, the
        entropy floor is not meaningful (a 1-2 child root naturally
        has 0-1 bits of entropy). Skip the floor check and reset
        the consecutive counter so an immature tree cannot fire the
        MCTS-pathology kill.
        """
        self._entropy_observations += 1
        n_children = int(len(visit_distribution))
        if n_children == 0:
            return
        if n_children < self.entropy_min_children:
            # N1: tree too narrow to score -- skip without firing.
            self._entropy_count = 0
            self._entropy_state = KillState.HEALTHY
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
        """K-M9-3 value divergence.

        Two signals OR'd:
        - **Absolute ceiling** (no warmup): sustained |V| >=
          `value_abs_ceiling` over `value_ceiling_sustained` cycles
          fires the kill. Catches a V stuck at a high constant that
          recalibrates the band (4.8's round-2 extension of Fable's
          meta-pattern).
        - **Trending band**: catches oscillation / relative drift
          within the operating range.

        Ceiling fire takes precedence over band fire.
        """
        v_abs = abs(float(v_estimate))
        if v_abs >= self.value_abs_ceiling:
            self._value_ceiling_consecutive += 1
        else:
            self._value_ceiling_consecutive = 0
        band_state = KillState(self._value_band.observe(float(v_estimate)))
        if self._value_ceiling_consecutive >= self.value_ceiling_sustained:
            self._value_state = KillState.FIRED
        elif self._value_ceiling_consecutive > 0:
            self._value_state = (
                KillState.FLAGGED if band_state == KillState.HEALTHY else band_state
            )
        else:
            self._value_state = band_state

    def observe_gamma(self, gamma_value: float) -> None:
        """K-M9-4 gamma divergence (F2 fix).

        Two signals OR'd:
        - **Primary: clamp-proximity** (no warmup gating). Sustained
          gamma within `gamma_clamp_threshold` of either clamp bound
          for `gamma_clamp_sustained` cycles -> fire. This catches
          the legacy ratchet's saturation-then-pinned failure mode
          which the trending band could not see (MAD->0 at saturation
          makes the band blind, Fable's B3).
        - **Secondary: trending-band** (gated by extended warmup so
          the initial EMA ramp from gamma_init does not false-positive).
          Catches in-band divergent oscillation that wouldn't hit
          either clamp.

        State precedence: clamp-FIRED beats band-* (clamp pathology
        is the louder signal).
        """
        # F2 primary: clamp-proximity. Anchor each threshold to its
        # OWN clamp (multiplicative) rather than to the (max-min)
        # range, so the formula stays meaningful when min and max are
        # orders of magnitude apart (default gamma_min=0.01,
        # gamma_max=100). Linear-range form would put clamp_low at
        # ~5, swallowing the normal operating range.
        clamp_high = self.gamma_max * (1.0 - self.gamma_clamp_threshold)
        clamp_low = self.gamma_min * (1.0 + self.gamma_clamp_threshold)
        at_clamp = (gamma_value >= clamp_high) or (gamma_value <= clamp_low)
        if at_clamp:
            self._gamma_clamp_consecutive += 1
        else:
            self._gamma_clamp_consecutive = 0

        # F2 secondary: trending band (warmup-gated).
        band_state = KillState(self._gamma_band.observe(float(gamma_value)))

        # Compose: clamp fires take precedence over band.
        if self._gamma_clamp_consecutive >= self.gamma_clamp_sustained:
            self._gamma_state = KillState.FIRED
        elif self._gamma_clamp_consecutive > 0:
            # In clamp but not sustained -> at least FLAGGED.
            self._gamma_state = KillState.FLAGGED \
                if band_state == KillState.HEALTHY else band_state
        else:
            self._gamma_state = band_state

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
        self._gamma_clamp_consecutive = 0
        self._value_ceiling_consecutive = 0
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
