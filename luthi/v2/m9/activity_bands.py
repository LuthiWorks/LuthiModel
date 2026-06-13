"""§A.1 unified activity signal: per-modality running bands + predicates.

Per 4.8's 2026-06-11 gate-repair spec, the activity signal feeds two
fixes that previously read a broken signal:

- **F1 P3 connection**: which candidate "emits" was the question. The
  old answer read sigmoid(text_intensity_head(a_t)) -- a sigmoid'd
  untrained Linear at launch, can sit anywhere in [0, 1]. The new
  answer reads raw text-decoder activity vs a per-modality running
  band: active = activity is above the median by a configurable
  margin in MAD units.
- **F4 K-M9-5 dark-room kill**: external_stasis required ALL decoder
  intensities below 0.5 from the same untrained sigmoid heads;
  probe_d showed the kill was armable in only 26/256 random seeds.
  The new external_stasis = ALL modality activities below the
  per-modality band. Armed by construction at any init because the
  band tracks observed activity.

The running band is `DriftBand` (median + MAD over a window) -- the
same shape as the M8 trending-kill machinery (72526cb) the spec asks
us to reuse. Pilot-set tunables: window length, the MAD multiplier
below the median that defines "silent".

Per F4's caveat, the activity signal is ALSO instrumented as the
armed-state log: per cycle we report whether the kill is even
CAPABLE of firing under current observations. A disarmed safety
backstop must be visible, not silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from luthi.v2.m9.staleness import DriftBand


@dataclass
class ActivityBandConfig:
    """Pilot-set tunables for the per-modality activity bands."""

    window: int = 32
    # silence_k * MAD below median = the "below band" threshold.
    # Larger -> more activity needed to count as "silent" (broader
    # tolerance).
    silence_k: float = 1.5
    # Minimum samples in the band before the silent/active predicate
    # returns meaningful values. Before warmup, default to "active"
    # (do not fire safety kills against an uncalibrated band).
    min_warmup: int = 8
    # R1 round-2 interim (4.8 + Fable): absolute floor. Activity at
    # or below this is silent REGARDLESS of band. The non-adapting
    # backstop -- the band alone recalibrates to a catatonic constant
    # (probe_g). The loop-integration target is the `a_rest`-based
    # reference (activity within `rest_tolerance` of decode(a_rest));
    # this absolute floor remains as a launch-time safety net since
    # the RestActionNet output is near-zero (zero-init) and unhelpful
    # before any training has occurred.
    absolute_silent_floor: float = 1e-3
    # Sigmoid scale for the continuous emission signal (R3 fix). A
    # candidate's emission strength is sigmoid((activity - floor)/scale),
    # so c_con varies smoothly with predicted activity rather than
    # collapsing to a binary text_active that fails when candidates
    # bunch on one side of a threshold (probe_e).
    emission_signal_scale: float = 0.1
    # Tolerance for the `a_rest` reference in `is_silent`/`external_stasis`.
    # When the loop passes `rest_activity_value` (= decoders.activity(
    # decode(a_rest(s_t))) per modality), activity is "at rest" if
    # `activity <= rest_activity + rest_tolerance`. Small constant
    # because rest_activity is a learned per-state quantity already
    # near the silence baseline; the tolerance just absorbs sampling
    # noise in the population mean.
    rest_tolerance: float = 1e-4


class ActivityBands:
    """Per-modality bands on the §A.1 activity scalar.

    Usage pattern in the loop:

        bands = ActivityBands(modalities=("text", "attention", "memory"))
        # Each cycle:
        activity = decoders.activity(a_t)          # dict {modality: [B]}
        per_batch_silent = bands.observe_and_classify(activity)
        # per_batch_silent: dict {modality: [B] bool}
        # F1 P3 reads per_batch_silent["text"]
        # F4 K-M9-5 reads per_batch_silent (all-modalities-silent = stasis)

    Activity observations are mean-reduced across batch into the band
    so the band tracks population-level scale, not a single batch
    element's variation. Per-batch silent/active is then computed by
    comparing each batch element's activity to the population-level
    threshold.
    """

    def __init__(
        self,
        modalities: tuple[str, ...] = ("text", "attention", "memory"),
        config: ActivityBandConfig | None = None,
    ):
        self.config = config or ActivityBandConfig()
        self.modalities = modalities
        self._bands: dict[str, DriftBand] = {
            m: DriftBand(window=self.config.window) for m in modalities
        }

    # ------------------------------------------------------------------
    # Observe (push activity into bands).
    # ------------------------------------------------------------------
    def observe(self, activity: dict) -> None:
        """Push the population-mean of each modality's activity into
        its band. Activity is a {modality: [B] tensor} dict.
        """
        for m in self.modalities:
            if m not in activity:
                continue
            pop_mean = float(activity[m].detach().mean().item())
            self._bands[m].push(pop_mean)

    # ------------------------------------------------------------------
    # Per-batch silent / active classification.
    # ------------------------------------------------------------------
    def silent_threshold(self, modality: str) -> float:
        """median - silence_k * MAD for the given modality.

        Activity strictly below this threshold counts as "silent".
        Before warmup, returns -inf so the classifier defaults to
        "active" (never call something silent against an uncalibrated
        band).
        """
        band = self._bands[modality]
        if not band.is_warm(min_samples=self.config.min_warmup):
            return float("-inf")
        med = band.median()
        mad = max(band.mad(), 1e-8)
        return med - self.config.silence_k * mad

    def is_silent(
        self,
        modality: str,
        activity_value: torch.Tensor,
        rest_activity_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B] bool -- True where this modality's activity is below
        its band's silent threshold OR below the absolute floor OR at
        or below the per-state `a_rest` reference (when provided).

        Three OR'd conditions, each catches a regime the others miss:

        - **Band threshold** (median - silence_k * MAD): the running
          band's view of "below typical." Strict but recalibrates to
          sustained constants (probe_g).
        - **Absolute floor** (round-2 R1 backstop): activity at or
          below `absolute_silent_floor` is silent regardless of band
          state. Catches launch-time catatonia and band recalibration
          to constants.
        - **a_rest reference** (loop-integration target per spec
          §6.i): activity at or below the per-state rest reference
          (within tolerance `rest_tolerance`) is silent. This is the
          context-dependent silence reference; once RestActionNet has
          trained it dominates the absolute floor (which becomes
          meaningful only as a launch-time safety net).
        """
        thr = self.silent_threshold(modality)
        floor = self.config.absolute_silent_floor
        below_band = activity_value < thr
        below_floor = activity_value <= floor
        if rest_activity_value is None:
            return below_band | below_floor
        tol = self.config.rest_tolerance
        below_rest = activity_value <= (rest_activity_value + tol)
        return below_band | below_floor | below_rest

    def emission_strength(
        self,
        modality: str,
        activity_value: torch.Tensor,
    ) -> torch.Tensor:
        """[B] in [0, 1] -- continuous probability that this modality
        is *emitting* (not silent) given its current activity. R3 fix
        replacing the binary `text_active = activity >= threshold`
        that collapsed to 0-spread when candidates didn't straddle
        the threshold (probe_e).

        Implementation: sigmoid((activity - floor) / scale). Continuous
        in `activity`, so even when K candidates' activities cluster
        on one side of a threshold, their emission_strength values
        differ smoothly -- c_con spread > 0 in the concentrated regime
        the trained habit net actually produces.
        """
        scale = max(self.config.emission_signal_scale, 1e-8)
        floor = self.config.absolute_silent_floor
        return torch.sigmoid((activity_value - floor) / scale)

    def per_batch_silent(
        self,
        activity: dict,
        rest_activity: dict | None = None,
    ) -> dict:
        """{modality: [B] bool} -- per-batch silent mask per modality.

        When `rest_activity` is provided (= `decoders.activity(
        decode(a_rest(s_t)))` from the loop), each modality's silent
        predicate also fires when activity is at or below the per-
        modality rest reference (within `rest_tolerance`). See
        `is_silent`.
        """
        out: dict = {}
        for m in self.modalities:
            if m not in activity:
                continue
            rest_m = (
                rest_activity[m]
                if rest_activity is not None and m in rest_activity
                else None
            )
            out[m] = self.is_silent(m, activity[m], rest_activity_value=rest_m)
        return out

    def external_stasis(
        self,
        activity: dict,
        rest_activity: dict | None = None,
    ) -> torch.Tensor:
        """F4 dark-room input: [B] bool -- True where ALL modalities
        are silent for this batch element.

        Before warmup at least one modality returns is_silent = all-
        False (because silent_threshold = -inf), so external_stasis is
        all-False before the band is calibrated -- the dark-room kill
        does not fire against an uncalibrated band, *but* see
        `armed_per_modality()` for the explicit armed-state log so
        operators can see the kill is currently disarmed.

        When `rest_activity` is provided, per-modality silence also
        fires when activity is at or below decode(a_rest) -- the
        context-dependent silence reference per spec §6.i.
        """
        masks = list(
            self.per_batch_silent(activity, rest_activity=rest_activity).values()
        )
        if not masks:
            return torch.zeros(0, dtype=torch.bool)
        out = masks[0]
        for m in masks[1:]:
            out = out & m
        return out

    # ------------------------------------------------------------------
    # F1 P3 emission signal -- BINARY (legacy round-1) and CONTINUOUS
    # (round-2 R3 fix).
    # ------------------------------------------------------------------
    def text_active(self, activity: dict) -> torch.Tensor:
        """[B] bool -- True where text activity is at or above its
        silent threshold (i.e. NOT silent). **Legacy round-1 form**;
        Fable's probe_e showed this collapses to 0-spread when K
        candidates don't straddle the threshold (25-50% of seeds).
        Kept for backward compatibility with callers that explicitly
        request the binary gate; new callers must use
        `text_emission_strength(...)` below.
        """
        thr = self.silent_threshold("text")
        return activity["text"] >= thr

    def text_emission_strength(self, activity: dict) -> torch.Tensor:
        """[B] in [0, 1] -- continuous emission probability for the
        text modality. R3 fix: smooth in activity, so c_con varies
        across K candidates even when their activities cluster on
        one side of a binary threshold.
        """
        return self.emission_strength("text", activity["text"])

    # ------------------------------------------------------------------
    # F4 armed-state instrument (mandatory per gate-repairs spec).
    # ------------------------------------------------------------------
    def armed_per_modality(self) -> dict:
        """{modality: bool} -- True where this modality's band is warm
        enough that the silent classifier can return True for some
        activity value. A disarmed safety backstop is visible here,
        not silent.

        K-M9-5 is *armed* iff all modalities are armed (because
        external_stasis = all-silent requires every modality to be
        independently classifiable). If any modality is disarmed,
        K-M9-5 cannot fire.
        """
        return {
            m: self._bands[m].is_warm(min_samples=self.config.min_warmup)
            for m in self.modalities
        }

    def k_m9_5_armed(self) -> bool:
        """K-M9-5 is armed iff every modality's band is warm. Loop
        logs this per cycle; sustained False is itself a flag (safety
        backstop disarmed).
        """
        return all(self.armed_per_modality().values())

    # ------------------------------------------------------------------
    # Diagnostics snapshot.
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Per-cycle log of activity bands + armed state."""
        snap = {"k_m9_5_armed": self.k_m9_5_armed()}
        for m in self.modalities:
            band = self._bands[m]
            snap[f"{m}_activity_median"] = band.median()
            snap[f"{m}_activity_mad"] = band.mad()
            snap[f"{m}_silent_threshold"] = self.silent_threshold(m)
            snap[f"{m}_armed"] = band.is_warm(min_samples=self.config.min_warmup)
        return snap
