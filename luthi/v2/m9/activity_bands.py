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

    def is_silent(self, modality: str, activity_value: torch.Tensor) -> torch.Tensor:
        """[B] bool -- True where this modality's activity is below
        its band's silent threshold.
        """
        thr = self.silent_threshold(modality)
        return activity_value < thr

    def per_batch_silent(self, activity: dict) -> dict:
        """{modality: [B] bool} -- per-batch silent mask per modality."""
        return {
            m: self.is_silent(m, activity[m])
            for m in self.modalities
            if m in activity
        }

    def external_stasis(self, activity: dict) -> torch.Tensor:
        """F4 dark-room input: [B] bool -- True where ALL modalities
        are silent for this batch element.

        Before warmup at least one modality returns is_silent = all-
        False (because silent_threshold = -inf), so external_stasis is
        all-False before the band is calibrated -- the dark-room kill
        does not fire against an uncalibrated band, *but* see
        `armed_per_modality()` for the explicit armed-state log so
        operators can see the kill is currently disarmed.
        """
        masks = list(self.per_batch_silent(activity).values())
        if not masks:
            return torch.zeros(0, dtype=torch.bool)
        out = masks[0]
        for m in masks[1:]:
            out = out & m
        return out

    # ------------------------------------------------------------------
    # F1 P3 emission signal.
    # ------------------------------------------------------------------
    def text_active(self, activity: dict) -> torch.Tensor:
        """[B] bool -- True where text activity is at or above its
        silent threshold (i.e. NOT silent). Used by P3 to drive the
        per-candidate emission signal: a candidate "emits" iff its
        text activity is above the band.
        """
        thr = self.silent_threshold("text")
        return activity["text"] >= thr

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
