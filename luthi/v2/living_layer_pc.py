"""PredictiveCodingLayer: a living layer with PC error-driven updates.

The v2 substrate replaces v1's Hebbian self-modification with hierarchical
predictive coding (Whittington-Bogacz variant). The forward pass IS still
learning, but updates are driven by prediction error rather than
input-output correlation.

Two-tier memory: fast episode store (retrieval, identical to v1) plus
consolidation into PC weights (M4 work — see consolidation.py).

See `docs/V2_IMPLEMENTATION_PLAN.md` for the architectural specification
and `docs/LUTHI_V2_PREDICTIVE_CODING_BRIEF.md` (archived — see
`docs/ARCHIVED.md`) for the design rationale.
"""

import torch
import torch.nn as nn


class PredictiveCodingLayer(nn.Module):
    """A linear layer whose weights self-modify via predictive coding.

    Each forward pass:
      1. Episodic recall: optionally blend a context-matched weight
         snapshot into the active weight.
      2. Linear computation: output = input @ weight_snapshot.T.
      3. PC self-modification: update weight, prediction, set_point,
         momentum, update_ema, precision, error_acc — all locally,
         no gradient flow.
      4. Episode storage if salience exceeds threshold.

    Living-state buffers are registered as buffers, not parameters. The
    optimizer never touches them. The layer trains itself.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pc_rate: float = 0.001,
        pred_learning_rate: float = 0.0001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        momentum_decay: float = 0.99,
        update_ema_decay: float = 0.99,
        precision_ema_decay: float = 0.999,
        precision_min: float = 0.1,
        precision_max: float = 10.0,
        prediction_clamp: float = 10.0,
        salience_threshold: float = 0.1,
        num_episodes: int = 32,
        context_dim: int = 64,
        episode_blend: float = 0.1,
        compressed_episodes: bool = False,
        consolidation_enabled: bool = False,
        consolidation_window: int = 1000,
        consolidation_trigger_window: int = 100,
        consolidation_threshold_factor: float = 0.5,
        consolidation_rate_factor: float = 0.1,
        consolidation_style: str = "gradient",
        consolidation_attractor_passes: int = 1,
        sparse_threshold: float = 0.0,
        sparse_warmup_steps: int = 500,
        # --- episode-store admission / retention (2026-07-27 defect fix) ---
        # Admission is a trailing percentile of this layer's own salience
        # distribution; `salience_threshold` is kept as a floor only.
        # OPT-IN, like every other machinery change on this ladder
        # (relative_trust, taper, sigreg): default False preserves the exact
        # pre-2026-07-27 behaviour, so no completed family's configuration
        # silently changes meaning. v6 arms turn these on deliberately.
        adaptive_episodes: bool = False,
        salience_window_size: int = 512,
        salience_percentile: float = 0.995,
        # Admission v2 (2026-07-28, after the validation probe): a trailing
        # percentile has no notion of SCALE. Measured salience varies ~1.5%
        # within a window, so p99.5 sits a hair above the median and any local
        # drift clears it -- the probe admitted 85% of calls and filled the
        # store with consecutive steps (similarity 1.0000). Admission is now
        # detrended surprise against the layer's own local baseline, plus a
        # refractory period as a structural bound on rate and on temporal
        # clustering.
        surprise_k: float = 3.0,          # deviations above the local forecast
        surprise_decay: float = 0.01,     # baseline/scale EMA rate
        surprise_drift_gain: float = 0.1,  # Holt trend gain; 0 = untrended
        refractory_calls: int = 250,      # hard floor on write spacing
        # 0 = warm up on statistics alone (window fill). Non-zero adds an extra
        # absolute lockout on top, for arms that want one.
        episode_warmup_steps: int = 0,
        episode_age_tau: float = 24000.0,
        eviction_alpha: float = 0.6,
        adaptive_recall: bool = False,
        recall_sigma: float = 2.0,
        # --- homeostatic activity band (2026-07-27): the sparse gate's key ---
        # Opt-in. Bounds are RELATIVE to the layer's own median activity, for
        # the same reason episode admission is: absolute constants do not
        # survive a signal that differs per block and decays over training.
        # Drive normalization (external review 2026-07-28, item 1.1): divide
        # the PC error by its running RMS before it enters delta_w, so the
        # living channel responds to relative surprise instead of absolute
        # error and does not extinguish itself as the model improves. Opt-in.
        drive_normalize: bool = False,
        drive_rms_decay: float = 0.01,
        # Surprise drive (2026-07-29). `drive_normalize` above turned out to be
        # a no-op in the production regime -- the +/-1 clamp on
        # (drive_error * precision) is 100% saturated at production precision
        # scales, so dividing the error by a positive scalar changes nothing but
        # arithmetic. See the correction block in pc_ops.py and
        # docs/research/2026-07-29_the-drive-fix.md.
        #
        # "surprise" makes the drive normalized EXCESS error over a Holt
        # forecast of the layer's own error scale: scale-free (does not
        # extinguish as the model improves), quiet on familiar input, and full
        # range on novel input. REQUIRES relative_trust -- absolute precision
        # weighting saturates the clamp and discards the drive magnitude, which
        # would defeat the mechanism. Enforced with a raise, not silently
        # supplied. Opt-in; default "raw" is bit-identical to pre-2026-07-29.
        drive_mode: str = "raw",           # "raw" | "rms" | "surprise"
        drive_decay: float = 0.01,         # Holt level/dev rate (~100 calls)
        drive_drift_gain: float = 0.1,     # trend term, as in admission v3
        drive_surprise_k: float = 3.0,     # THRESHOLD in deviations, as admission v3
        drive_gain_max: float = 4.0,
        drive_gain_floor: float = 0.0,     # 0.0 = fully quiet when unsurprised
        drive_dev_floor_frac: float = 0.01,
        drive_warmup_calls: int = 200,     # behaves as "raw" until converged
        homeostatic_band_enabled: bool = False,
        band_decay: float = 1e-3,          # slow: ~1000-step timescale
        band_warmup_steps: int = 200,
        band_lo_frac: float = 0.25,        # under 25% of median activity = rut
        band_hi_frac: float = 4.0,         # over 4x median = overactive
        band_h_min: float = 0.5,           # damping floor: never silences
        band_h_max: float = 3.0,           # boost ceiling: never unbounded
        band_max_boost_frac: float = 0.05,  # <=5% of rows reopened at once
        band_open_deficit: float = 0.5,    # gate-reopen threshold
        inference_steps_per_forward: int = 1,
        episode_recall_threshold: float = 0.5,
        learning_gain_enabled: bool = False,
        relative_trust: bool = False,
        learning_gain_rise: float = 2.0,
        learning_gain_cap: float = 3.0,
        resolution_short_decay: float = 0.99,
        resolution_long_decay: float = 0.999,
        resolution_warmup: int = 8,
        buffer_dtypes: dict[str, torch.dtype] | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pc_rate = pc_rate
        self.pred_learning_rate = pred_learning_rate
        self.homeostatic_decay = homeostatic_decay
        self.set_point_adapt_rate = set_point_adapt_rate
        self.momentum_decay = momentum_decay
        self.update_ema_decay = update_ema_decay
        self.precision_ema_decay = precision_ema_decay
        self.precision_min = precision_min
        self.precision_max = precision_max
        self.prediction_clamp = prediction_clamp
        self.salience_threshold = salience_threshold
        self.num_episodes = num_episodes
        self.context_dim = context_dim
        self.episode_blend = episode_blend
        # Recall gate (parameterized 2026-07-17, run-3 recall tightening):
        # minimum context cosine before a stored episode blends in. The
        # historical 0.5 admits weakly-matched snapshots, which perturb
        # representations quasi-randomly at readout time.
        self.episode_recall_threshold = float(episode_recall_threshold)
        # Plasticity taper scale (run-3 build): multiplies pc_rate and
        # pred_learning_rate in the forward's self-mod call. Plain
        # attribute, recomputed each step by the trainer's schedule
        # (formative -> mature with a FLOOR, never zero -- DH-4's
        # "lowering the learning rate of the self, never halting it").
        self.rate_scale: float = 1.0
        self.consolidation_enabled = consolidation_enabled
        self.consolidation_rate_factor = consolidation_rate_factor
        # Salvatori-style attractor consolidation (2026-05-14). "gradient"
        # is the original M4 pathway (pull weight toward stored snapshot).
        # "attractor" is the Salvatori 2023 pathway (replay stored input
        # pattern through pc_self_modify so the pattern becomes a local
        # energy minimum). "both" runs gradient first, then attractor —
        # gradient places the weight near the stored regime, then
        # attractor reinforces the input as a fixed point of that regime.
        # No silent fallback: invalid value raises here, not at forward
        # time. Future instances reading this: do not add a default-case
        # else clause that picks a style — the choice should always be
        # explicit.
        valid_styles = ("gradient", "attractor", "both")
        if consolidation_style not in valid_styles:
            raise ValueError(
                f"consolidation_style must be one of {valid_styles}, "
                f"got {consolidation_style!r}"
            )
        self.consolidation_style = consolidation_style
        if consolidation_attractor_passes < 1:
            raise ValueError(
                f"consolidation_attractor_passes must be >= 1, got "
                f"{consolidation_attractor_passes}"
            )
        self.consolidation_attractor_passes = int(
            consolidation_attractor_passes
        )

        # Sparse PC gating (lit-followup 2026-05-13). When sparse_threshold
        # > 0, output rows with error_acc[j] below the threshold skip the
        # weight update for that row. Analog of v1's spiking gate in
        # continuous error space. Bandwidth saving is proportional to the
        # fraction of outputs gated off. The warmup window prevents the
        # bootstrap-deadlock (error_acc starts at 0; if we gated
        # immediately, nothing would ever learn and error_acc would never
        # grow). After warmup_steps the gate engages.
        self.sparse_threshold = float(sparse_threshold)
        self.sparse_warmup_steps = int(sparse_warmup_steps)
        # Episode-store admission / retention (2026-07-27). Defaults ON: the
        # previous behaviour is a confirmed defect, not a baseline worth
        # preserving. adaptive_episodes=False restores it exactly for A/B work.
        self.adaptive_episodes = bool(adaptive_episodes)
        self.salience_percentile = float(salience_percentile)
        self.surprise_k = float(surprise_k)
        self.surprise_decay = float(surprise_decay)
        self.surprise_drift_gain = float(surprise_drift_gain)
        self.refractory_calls = int(refractory_calls)
        self.episode_warmup_steps = int(episode_warmup_steps)
        self.episode_age_tau = float(episode_age_tau)
        self.eviction_alpha = float(eviction_alpha)
        self.adaptive_recall = bool(adaptive_recall)
        self.recall_sigma = float(recall_sigma)
        self.drive_normalize = bool(drive_normalize)
        self.drive_rms_decay = float(drive_rms_decay)
        if drive_mode not in ("raw", "rms", "surprise"):
            raise ValueError(
                f"drive_mode must be 'raw', 'rms' or 'surprise'; got "
                f"{drive_mode!r}"
            )
        if drive_mode == "surprise" and not relative_trust:
            raise ValueError(
                "drive_mode='surprise' requires relative_trust=True (absolute "
                "precision weighting saturates the +/-1 clamp and discards the "
                "drive magnitude). Set both on the arm config so the pairing is "
                "attributable."
            )
        self.drive_mode = str(drive_mode)
        self.drive_decay = float(drive_decay)
        self.drive_drift_gain = float(drive_drift_gain)
        self.drive_surprise_k = float(drive_surprise_k)
        self.drive_gain_max = float(drive_gain_max)
        self.drive_gain_floor = float(drive_gain_floor)
        self.drive_dev_floor_frac = float(drive_dev_floor_frac)
        self.drive_warmup_calls = int(drive_warmup_calls)
        self.homeostatic_band_enabled = bool(homeostatic_band_enabled)
        self.band_decay = float(band_decay)
        self.band_warmup_steps = int(band_warmup_steps)
        self.band_lo_frac = float(band_lo_frac)
        self.band_hi_frac = float(band_hi_frac)
        self.band_h_min = float(band_h_min)
        self.band_h_max = float(band_h_max)
        self.band_max_boost_frac = float(band_max_boost_frac)
        self.band_open_deficit = float(band_open_deficit)
        # Step counter as a non-persistent attribute (not a buffer — we
        # don't checkpoint it; recovery from checkpoint resets warmup).
        self._sparse_step_count: int = 0

        # iPC interleaved inference + update (Salvatori et al. 2022,
        # arXiv:2212.00720). With inference_steps_per_forward=T>1, the
        # forward repeats the matmul + self-mod cycle T times within
        # one external forward call. Classical PC = T=1 (current
        # default, behavior unchanged). Reported in the iPC paper to
        # consistently improve convergence over the "infer to
        # completion, then update" classical schedule.
        # Constraint: iPC mode is incompatible with gradient checkpointing
        # because the weight evolves T times within the forward; the
        # cached snapshot for recompute would not reproduce the original
        # trajectory. Forward raises if recomputing and T > 1.
        if inference_steps_per_forward < 1:
            raise ValueError(
                f"inference_steps_per_forward must be >= 1, got "
                f"{inference_steps_per_forward}"
            )
        self.inference_steps_per_forward = int(inference_steps_per_forward)

        if consolidation_enabled:
            from luthi.v2.consolidation import ConsolidationTracker
            self._consolidation_tracker = ConsolidationTracker(
                window_size=consolidation_window,
                trigger_window=consolidation_trigger_window,
                threshold_factor=consolidation_threshold_factor,
            )
        else:
            self._consolidation_tracker = None
        self._consolidation_fire_count: int = 0
        # --- consolidation EFFECT counters (2026-08-14, Opus 5 audit) ---
        # `_consolidation_fire_count` counts TRIGGERS, not consolidations.
        # Both replay pathways return the number of episodes they actually
        # replayed and return 0 immediately on an empty store -- and both
        # return values were discarded while the fire count incremented
        # unconditionally. Measured consequence on the 768x8 family
        # (seed 97, 54,000 steps): blocks 0-4 logged ~1,000
        # `consolidation_fires` each having replayed ZERO episodes, because
        # their episode stores were empty for the entire run. The metric
        # Brian asked for on 2026-07-18 to make "memory becoming structure"
        # countable was counting the trigger and not the becoming.
        #
        # replayed_total: cumulative episodes actually replayed.
        # noop_fires: triggers where both pathways replayed nothing.
        # fires - noop_fires = triggers that did something.
        self._consolidation_replayed_total: int = 0
        self._consolidation_noop_fires: int = 0

        self._buffer_dtype_overrides: dict[str, torch.dtype] = dict(
            buffer_dtypes or {}
        )

        # Compressed episode storage: episode_values stored as INT8 with a
        # per-episode FP32 scale. Reduces episode-store memory by 4x vs FP32.
        # The 2026-05-10 audit flagged that at 4096d × 36 blocks × 64
        # episodes, FP32 episode_values is ~150 GB — infeasible. INT8 brings
        # that to ~36 GB, still large but tractable. Production scale will
        # additionally need low-rank delta compression (rank-r SVD of
        # (snapshot - baseline)) — left as a Phase 5 follow-up since M5
        # pilot scale (≤256d) fits comfortably without it. The INT8 code
        # path is inherited from v1's Ablation C design, already in
        # `_store_episode` and `_recall_episode`.
        if compressed_episodes:
            self._buffer_dtype_overrides.setdefault(
                "episode_values", torch.int8
            )

        # --- Core weight state (all buffers, not parameters) ---

        weight = torch.empty(
            out_features, in_features, dtype=self._buf_dtype("weight")
        )
        nn.init.kaiming_uniform_(weight)
        self.register_buffer("weight", weight)

        # Top-down prediction matrix. Initialized to zero so that
        # output @ prediction starts at zero, making the initial pred_error
        # equal to the actual input mean — a clean starting signal that the
        # PC update can immediately drive toward zero.
        self.register_buffer(
            "prediction",
            torch.zeros(
                out_features, in_features,
                dtype=self._buf_dtype("prediction"),
            ),
        )

        self.register_buffer(
            "set_point",
            weight.clone().to(dtype=self._buf_dtype("set_point")),
        )

        self.register_buffer(
            "momentum",
            torch.zeros(
                out_features, in_features,
                dtype=self._buf_dtype("momentum"),
            ),
        )

        # Metaplasticity tracker. Refinement 5 verifies the v1 ratio-check
        # semantics still hold under PC dynamics.
        self.register_buffer(
            "update_ema",
            torch.ones(
                out_features, in_features,
                dtype=self._buf_dtype("update_ema"),
            )
            * 1e-4,
        )

        # Per-input error reliability — self-organizes toward 1/error_variance.
        self.register_buffer(
            "precision",
            torch.ones(
                in_features, dtype=self._buf_dtype("precision")
            ),
        )

        # Per-output running prediction-error magnitude.
        # Refinement 2: salience for episode storage = mean(error_acc).
        self.register_buffer(
            "error_acc",
            torch.zeros(
                out_features, dtype=self._buf_dtype("error_acc")
            ),
        )

        # Per-input plasticity. The buffer-table in V2_IMPLEMENTATION_PLAN.md
        # does not list plasticity, but the plan's two-layer top-down design
        # (decision 3) modulates "the lower block's plasticity and set_point."
        # Without plasticity the modulation channel collapses to set_point
        # only, weakening the design. Retained as [in_features] (matching
        # v1's free-win refactor) so apply_top_down has both channels to act on.
        self.register_buffer(
            "plasticity",
            torch.ones(
                in_features, dtype=self._buf_dtype("plasticity")
            ),
        )

        # --- Layer-level episode store (identical to v1) ---

        proj = torch.randn(in_features, context_dim) / (in_features ** 0.5)
        self.register_buffer("context_proj", proj)

        self.register_buffer(
            "episode_contexts",
            torch.zeros(
                num_episodes, context_dim,
                dtype=self._buf_dtype("episode_contexts"),
            ),
        )
        self.register_buffer(
            "episode_values",
            torch.zeros(
                num_episodes, out_features, in_features,
                dtype=self._buf_dtype("episode_values"),
            ),
        )
        self.register_buffer(
            "episode_scales",
            torch.ones(
                num_episodes, dtype=self._buf_dtype("episode_scales")
            ),
        )
        self.register_buffer("episode_saliences", torch.zeros(num_episodes))
        self.register_buffer(
            "episode_count", torch.tensor(0, dtype=torch.long)
        )
        # --- Anti-fossil episode machinery (2026-07-27) ---
        # Checkpoint forensics on the completed v5 family found every store
        # frozen since ~step 1000: a single global salience_threshold cannot
        # track a signal whose per-block median is 0.001-0.004 and which
        # decays 70-90% over training, so the slots written during the
        # initialization transient were never beatable again. Blocks 0-2 never
        # admitted anything at all. See
        # docs/research/2026-07-27_episode-store-frozen-defect.md.
        #
        # Write step per slot, for age decay in eviction. -1 = never written.
        self.register_buffer(
            "episode_steps",
            torch.full((num_episodes,), -1, dtype=torch.long),
        )
        # Ring buffer of recent saliences -> the per-block admission bar is a
        # trailing percentile of this layer's OWN distribution, not a constant.
        self.register_buffer(
            "salience_window", torch.zeros(salience_window_size)
        )
        self.register_buffer(
            "salience_window_pos", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "salience_window_filled", torch.tensor(0, dtype=torch.long)
        )
        # Global step counter for the store (persistent: age decay and warmup
        # must survive resume, unlike the sparse-gate warmup counter).
        self.register_buffer(
            "episode_step_counter", torch.tensor(0, dtype=torch.long)
        )
        # Local baseline and scale for detrended-surprise admission, plus the
        # last write step for the refractory bound.
        self.register_buffer("salience_level", torch.tensor(0.0))
        self.register_buffer("salience_drift", torch.tensor(0.0))
        self.register_buffer("salience_dev", torch.tensor(0.0))
        self.register_buffer(
            "last_write_step", torch.tensor(-10**9, dtype=torch.long)
        )
        # Adaptive recall: EMA mean/var of the best-match similarity, so recall
        # fires on genuine recognition rather than always. A fixed 0.5 against
        # observed similarities of >0.9 meant the blend was applied every step.
        self.register_buffer("recall_sim_mean", torch.tensor(0.0))
        self.register_buffer("recall_sim_var", torch.tensor(0.0))
        self.register_buffer(
            "recall_sim_count", torch.tensor(0, dtype=torch.long)
        )
        # Observational counters (never enter any loss).
        self.register_buffer(
            "episode_writes", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "recall_fires", torch.tensor(0, dtype=torch.long)
        )
        # Homeostatic band state: slow per-row activity estimate + counters.
        # Running RMS of the PC error, for drive normalization (item 1.1).
        self.register_buffer("error_rms", torch.tensor(0.0))
        # Surprise-drive state (2026-07-29): Holt level + trend + mean abs
        # deviation of the PC error scale, plus the observability the whole
        # mechanism rests on. `drive_gain` is the last gain applied and
        # `drive_fire_count` / `drive_calls` give the duty cycle -- without
        # those two there is no way to tell "quiet because nothing is new" from
        # "quiet because broken", which is the failure this fix exists to end.
        self.register_buffer("drive_ref", torch.tensor(0.0))
        self.register_buffer("drive_ref_drift", torch.tensor(0.0))
        self.register_buffer("drive_dev", torch.tensor(0.0))
        self.register_buffer("drive_calls", torch.tensor(0, dtype=torch.long))
        self.register_buffer("drive_gain", torch.tensor(0.0))
        self.register_buffer(
            "drive_fire_count", torch.tensor(0, dtype=torch.long)
        )
        # Sum of gain over firing calls only. With drive_fire_count this gives
        # mean-gain-when-firing, and differencing two logged records gives the
        # per-interval value. That separates the two ways a gated drive can die
        # -- firing LESS OFTEN vs firing WEAKER -- which a cumulative duty
        # cycle alone cannot distinguish, and which is exactly the question a
        # full-length run exists to answer. Added 2026-07-29 before the 3-epoch
        # run rather than after, because the alternative is running it twice.
        self.register_buffer("drive_gain_sum", torch.tensor(0.0))
        self.register_buffer("act_mean", torch.zeros(out_features))
        self.register_buffer("act_var", torch.zeros(out_features))
        self.register_buffer(
            "act_count", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "band_boost_rows", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "band_damp_rows", torch.tensor(0, dtype=torch.long)
        )
        # --- silent-skip counters (2026-08-14, Opus 5 audit) ---
        # Two paths in this layer decline to do their work and return
        # normally. Both were uncounted, which makes "quiet because the
        # input was ordinary" indistinguishable from "quiet because this
        # block's living channel has been off for 4,000 batches" -- the
        # exact discriminator `drive_duty` exists to provide for the
        # drive, missing here. CLAUDE.md: every mechanism ships with the
        # instrument that could catch it lying.
        #
        # nonfinite_forward_skips: a non-finite forward skips
        # pc_self_modify entirely AND publishes a zeroed _last_pred_error
        # to the top-down sweep. Correct behaviour (living buffers must
        # not eat NaN), but a block silently not learning is precisely
        # the failure this repo has paid for repeatedly.
        # persistent=False is REQUIRED, not incidental: v2 model loading has
        # a deliberate strict=True contract (see luthi/living_extra_state.py,
        # which exists so living state can grow without breaking it, and
        # tests/test_living_extra_state.py which pins it). A persistent
        # buffer added here would make every existing checkpoint fail to
        # load with a missing-key error. Consequence, accepted: these
        # counters reset on resume, so they are per-process. That is fine
        # for their purpose -- they are differenced between logged records
        # to get a rate, and a reset shows up as a negative delta rather
        # than as silence.
        self.register_buffer(
            "nonfinite_forward_skips", torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        # band_degenerate_skips: the homeostatic band no-ops when its
        # reference median is non-finite or <= 0 (i.e. activity variance
        # has collapsed). band_boost_rows/band_damp_rows both read 0 in
        # that case -- identical to a healthy band with nothing to do.
        self.register_buffer(
            "band_degenerate_skips", torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        # Mean input pattern at episode write time. Used by Salvatori-style
        # attractor consolidation (`consolidate_layer_attractor`) to re-
        # present stored input through the layer's PC dynamics so the
        # stored pattern becomes a local minimum of the prediction-error
        # energy. Always allocated regardless of consolidation_style; the
        # cost is num_episodes * in_features * 4 bytes (32 KB at
        # 32 episodes / 256d). Gradient-replay consolidation ignores it.
        self.register_buffer(
            "episode_inputs",
            torch.zeros(
                num_episodes, in_features,
                dtype=self._buf_dtype("episode_inputs"),
            ),
        )

        # Cache of the most recent per-input prediction error from the
        # PC self-modification step. Used by hybrid_block_pc.top_down_pass
        # to propagate prediction error from this block to the one below.
        # Non-persistent — recomputed every forward pass, not checkpointed.
        self._last_pred_error: torch.Tensor | None = None

        # Item #6 (2026-06-28): when set (via the freeze_plasticity()
        # context manager during the lived JEPA re-encode), forward()
        # produces grad-capable output WITHOUT self-modifying -- no
        # pc_self_modify, no episode store write, no living-buffer
        # mutation. The gradient still flows to self.weight so the lived
        # loss can train the encoder, but perception's one-time self-mod
        # is not duplicated by the learner's re-encode. Set by a
        # module-tree sweep, not threaded through forward args, so it
        # reaches every living layer in the trunk uniformly.
        self._plasticity_frozen: bool = False

        # --- Inverted-U learning gain (momentum-functions foundations,
        #     spec docs/research/2026-07-05_inverted-u-gain-spec.md §8 step 4).
        # Opt-in amplifier of directed, resolving novelty. Default OFF => the
        # gain machinery is fully inert and the forward is bit-identical to
        # legacy (regime f). When enabled, the explicit fall needs a
        # resolution-progress signal = short/long EMA of prediction error, so
        # instantiate the two slow traces here (spec §5). They persist via
        # living_extra_state (regime i) -- a restore mid-hard-growth must not
        # reset them, or the entity re-sensitizes on every waking.
        self.learning_gain_enabled = bool(learning_gain_enabled)
        # v5 precision awakening (2026-07-21): ratio-to-median trust
        # weighting + numerics-only eps + freed ledger. Default False
        # = legacy bit-identical (every pre-v5 family).
        self.relative_trust = bool(relative_trust)
        if self.relative_trust and self._buf_dtype("precision") != torch.float32:
            # The freed ledger records values up to 1e12; fp16 overflows
            # to inf at 65504 and the trust ratios silently rot.
            raise ValueError(
                "relative_trust requires a float32 precision buffer; got "
                f"{self._buf_dtype('precision')} via buffer_dtypes override"
            )
        self.learning_gain_rise = float(learning_gain_rise)
        self.learning_gain_cap = float(learning_gain_cap)
        from luthi.v2.slow_trace import SlowEMA, ReadResetAccumulator
        self._err_short = SlowEMA(
            decay=resolution_short_decay, warmup=resolution_warmup
        )
        self._err_long = SlowEMA(
            decay=resolution_long_decay, warmup=resolution_warmup
        )
        # Applied-change sinks (spec §4/§8 step 5; Fable step-8 ruling
        # 2026-07-06). Two DISTINCT consumers, deliberately different quantities:
        #  * NREM day-integral -- the RAW instantaneous applied change summed
        #    each step; NREM wants the integral of truth, not a smoothed one.
        #  * living-drift eye -- a per-layer EMA of the applied change at the
        #    SAME decay as `momentum`, so the eye's "applied_change" source is
        #    commensurate with its "momentum" source (both ~equally smoothed)
        #    rather than an instantaneous magnitude that would read as a spike
        #    against a momentum-built band. The eye's source is an explicit
        #    M9Config knob (living_drift_source), NOT keyed to the gain flag --
        #    so there is no stale-reading-forever edge and no flag-coupled unit
        #    discontinuity. Fed only on the gain path (there is no applied gain
        #    to observe otherwise); a fair-parallel refinement to a per-weight
        #    applied-momentum buffer is deferred to the tuning pass.
        self._applied_change_accum = ReadResetAccumulator()
        self._applied_ema = SlowEMA(decay=self.momentum_decay, warmup=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buf_dtype(self, name: str) -> torch.dtype:
        return self._buffer_dtype_overrides.get(name, torch.float32)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse)
        for name, target_dtype in self._buffer_dtype_overrides.items():
            buf = self._buffers.get(name, None)
            if buf is not None and buf.dtype != target_dtype:
                self._buffers[name] = buf.to(dtype=target_dtype)
        return self

    def _compute_context(self, x_flat: torch.Tensor) -> torch.Tensor:
        ctx = x_flat.mean(dim=0) @ self.context_proj
        return ctx / (ctx.norm() + 1e-8)

    def _recall_episode(self, context: torch.Tensor) -> torch.Tensor | None:
        if self.episode_count == 0:
            return None
        n = self.episode_count.item()
        stored = self.episode_contexts[:n]
        sims = torch.mm(stored, context.unsqueeze(-1)).squeeze(-1)
        best_idx = sims.argmax()
        best_sim = sims[best_idx]
        # The frozen re-encode path calls this too, and it must mutate NOTHING
        # (tests/test_mode_matrix.py::TestFrozenMutatesNothing). Statistics and
        # counters are therefore skipped while frozen; the gate still applies.
        frozen = getattr(self, "_plasticity_frozen", False)
        if self.adaptive_recall:
            # Recall on genuine recognition: the best match must stand out
            # from this layer's own recent match distribution. A fixed 0.5
            # against observed similarities above 0.9 fired every single step,
            # blending a stored weight-delta into every forward pass.
            c = int(self.recall_sim_count.item())
            mean = float(self.recall_sim_mean.item())
            var = float(self.recall_sim_var.item())
            bs = float(best_sim.item())
            # Welford update, done before the gate so the statistics track the
            # full distribution rather than only the firing tail.
            c += 1
            delta = bs - mean
            mean += delta / c
            var += (delta * (bs - mean) - var) / c if c > 1 else 0.0
            if not frozen:
                self.recall_sim_count.fill_(c)
                self.recall_sim_mean.fill_(mean)
                self.recall_sim_var.fill_(max(var, 0.0))
            if c < 64:
                # Cold start: fall back to the fixed gate rather than to
                # silence. Same principle as admission warmup — a mechanism
                # that quietly does nothing is worse than one using a cruder
                # rule, and it is what made the store a fossil in the first
                # place.
                if best_sim < self.episode_recall_threshold:
                    return None
            else:
                bar = mean + self.recall_sigma * (max(var, 0.0) ** 0.5)
                if bs <= bar or bs < self.episode_recall_threshold:
                    return None
        elif best_sim < self.episode_recall_threshold:
            return None
        if not frozen:
            self.recall_fires.add_(1)
        if self.episode_values.dtype == torch.int8:
            recalled = (
                self.episode_values[best_idx].to(self.weight.dtype)
                * self.episode_scales[best_idx]
            )
        else:
            recalled = self.episode_values[best_idx]
        delta = recalled - self.weight
        return delta * best_sim * self.episode_blend

    def _store_episode(
        self,
        context: torch.Tensor,
        salience: float,
        input_pattern: torch.Tensor,
    ) -> None:
        """Write one episode if salience exceeds threshold or beats the
        weakest stored episode. `input_pattern` ([in_features]) is the
        mean input vector for the current batch; saved into
        `episode_inputs` for use by Salvatori-style attractor
        consolidation. The weight snapshot, context, and salience are
        also written as before.
        """
        if not self.adaptive_episodes:
            # Legacy path, kept for A/B against the pre-2026-07-27 behaviour.
            if salience < self.salience_threshold:
                return
            n = self.episode_count.item()
            if n < self.num_episodes:
                idx = n
                self.episode_count.add_(1)
            else:
                min_idx = self.episode_saliences[:n].argmin()
                if salience <= self.episode_saliences[min_idx]:
                    return
                idx = min_idx.item()
            self._write_episode_slot(idx, context, salience, input_pattern)
            return

        step = int(self.episode_step_counter.item())
        self.episode_step_counter.add_(1)

        # Every step feeds the trailing window, admitted or not: the bar must
        # track the layer's ordinary level, which decays hard over training.
        pos = int(self.salience_window_pos.item())
        self.salience_window[pos] = salience
        self.salience_window_pos.fill_((pos + 1) % self.salience_window.numel())
        if int(self.salience_window_filled.item()) < self.salience_window.numel():
            self.salience_window_filled.add_(1)

        # Warmup is STATISTICAL, not temporal (second pass, same day): the gate
        # is "are there enough samples to compute a percentile yet?", which
        # scales itself to the run instead of imposing a step count that is 7%
        # of a 72K-step family and 100% of a 200-step experiment. Until the
        # window fills, admission falls back to the legacy absolute rule, so
        # the store is never inert — a mechanism that quietly does nothing
        # while reporting healthy is the exact failure this fix exists to end.
        #
        # Episodes admitted during that window (including initialization-
        # transient ones) are no longer permanent: age decay makes them
        # evictable, which the old argmin-on-salience path could never do.
        filled = int(self.salience_window_filled.item())
        if filled < self.salience_window.numel() or step < self.episode_warmup_steps:
            if salience < self.salience_threshold:
                return
            n = self.episode_count.item()
            if n < self.num_episodes:
                idx = n
                self.episode_count.add_(1)
            else:
                idx = self._choose_eviction_slot(n, step)
            self._write_episode_slot(idx, context, salience, input_pattern, step)
            return

        # NOTE: `salience_threshold` is deliberately NOT applied here. It is an
        # absolute constant (default 0.1) and the measured per-block salience
        # medians are 0.001-0.004 — applying it would make this whole path
        # inert, which is the defect being fixed. It remains in force on the
        # legacy path above.
        #
        # Detrended surprise (v2). Test against the PREVIOUS baseline, then
        # update it: folding the current sample in first would let a spike
        # partly mask itself. `salience_dev` is a mean-absolute-deviation
        # tracker, so the bar scales with however much this layer's salience
        # ordinarily wobbles -- the failure the percentile rule could not see.
        # Drift-corrected surprise (v3). v2 compared salience against a plain
        # EMA, which on a DECAYING series always sits above the current value
        # -- so nothing was ever surprising and every block froze after warmup
        # (measured: 1 write per block in 3,000 steps). v1's percentile failed
        # the mirror-image way on locally RISING series. Both were untrended.
        #
        # Holt's linear method: carry a drift term so the baseline forecasts
        # where the signal is heading, and score the residual against that
        # forecast. A smooth trend in either direction produces no residual;
        # only a departure from the trend does.
        level = float(self.salience_level.item())
        drift = float(self.salience_drift.item())
        dev = float(self.salience_dev.item())
        forecast = level + drift
        resid = salience - forecast
        surprising = resid > self.surprise_k * max(dev, 1e-12)
        b = self.surprise_decay
        new_level = forecast + b * resid
        self.salience_drift.fill_(drift + b * self.surprise_drift_gain * resid)
        self.salience_level.fill_(new_level)
        self.salience_dev.fill_(dev + b * (abs(resid) - dev))

        # Refractory period: a structural bound on both write rate and
        # temporal clustering. The probe's store held consecutive steps
        # (39985, 39986, 39987...) which is exactly how a store of 64 slots
        # ends up at similarity 1.0000 -- diversity in time is a
        # precondition for diversity in content.
        if step - int(self.last_write_step.item()) < self.refractory_calls:
            return
        if not surprising:
            return

        n = self.episode_count.item()
        if n < self.num_episodes:
            idx = n
            self.episode_count.add_(1)
        else:
            idx = self._choose_eviction_slot(n, step)
        self._write_episode_slot(idx, context, salience, input_pattern, step)

    def _apply_activity_band(
        self, output: torch.Tensor, gate: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Homeostatic activity band -- the key to the sparse gate's cage.

        Biology's answer to chronic underactivity (synaptic scaling, intrinsic
        plasticity): a unit that stops participating raises its own
        excitability until it does again. The discriminating signal is NOT low
        error -- a competent row also has low error. It is low *participation*:
        a row whose output has stopped varying has stopped saying anything,
        which is what a collapsed row looks like from the inside.

        Every quantity here is RELATIVE to this layer's own activity
        distribution. Absolute constants are what froze the episode store
        (2026-07-27): the per-block signal levels differ ~4x and decay 70-90%
        over a run, so any fixed number is either always-on or never-on.

        Bounds, per Brian's requirement that a positive-feedback loop be
        bounded at both ends:
          * multiplier clamped to [band_h_min, band_h_max] -- a dead row can
            never receive unbounded plasticity, an overactive one is damped
            but never silenced;
          * multiplier ONLY, never additive: with no error signal there is
            nothing to amplify, so the band cannot manufacture drift;
          * exact dead zone -- inside the band the multiplier is 1.0, so a
            healthy layer is bit-identical to one with the band disabled;
          * slow timescale (band_decay ~1e-3) so it cannot chase batch noise;
          * warmup lockout until the activity estimate is seeded;
          * rate limit: at most `band_max_boost_frac` of rows may be boosted
            at once, so a global dip cannot reopen everything and undo the
            gate's whole benefit.
        Downstream, the existing metaplasticity dampener also limits any
        boosted update, which is a second bound we get for free.
        """
        with torch.no_grad():
            row_mean = output.detach().mean(dim=0).to(self.act_mean.dtype)
            b = self.band_decay
            self.act_mean.mul_(1 - b).add_(row_mean, alpha=b)
            dev = (row_mean - self.act_mean) ** 2
            self.act_var.mul_(1 - b).add_(dev, alpha=b)
            self.act_count.add_(1)
            if int(self.act_count.item()) < self.band_warmup_steps:
                return gate

            act = self.act_var.clamp(min=0).sqrt()
            ref = act.median()
            if not torch.isfinite(ref) or float(ref) <= 0:
                # Counted (2026-08-14): boost/damp rows both read 0 here,
                # identical to a healthy band with nothing to correct.
                self.band_degenerate_skips.add_(1)
                return gate

            lo = self.band_lo_frac * ref
            hi = self.band_hi_frac * ref
            deficit = ((lo - act) / lo.clamp(min=1e-12)).clamp(0.0, 1.0)
            excess = ((act - hi) / hi.clamp(min=1e-12)).clamp(0.0, 1.0)

            # Rate limit: only the most-deprived rows get reopened.
            max_boost = max(1, int(self.band_max_boost_frac * act.numel()))
            boosted = deficit > 0
            if int(boosted.sum().item()) > max_boost:
                keep = torch.topk(deficit, max_boost).indices
                mask = torch.zeros_like(deficit, dtype=torch.bool)
                mask[keep] = True
                deficit = torch.where(mask, deficit, torch.zeros_like(deficit))

            h = 1.0 + (self.band_h_max - 1.0) * deficit \
                    - (1.0 - self.band_h_min) * excess
            h = h.clamp(self.band_h_min, self.band_h_max)

            self.band_boost_rows.fill_(int((deficit > 0).sum().item()))
            self.band_damp_rows.fill_(int((excess > 0).sum().item()))

            if gate is None:
                return h.to(self.weight.dtype)
            # The KEY: a row deprived past `band_open_deficit` has its gate
            # forced open. Without this the band can only scale an update the
            # gate has already zeroed -- i.e. do nothing, which is precisely
            # the trap it exists to prevent.
            opened = torch.where(
                deficit > self.band_open_deficit,
                torch.ones_like(gate),
                gate,
            )
            return (opened * h.to(gate.dtype)).to(self.weight.dtype)

    def _episode_store_stats(self) -> dict:
        """Diversity / turnover / admission-bar readings for the episode store.

        `episode_context_similarity` is the mean pairwise cosine across stored
        contexts (contexts are stored unit-normalized). Read it against a
        measured baseline, never against 1.0: the pre-fix v5 family sat at
        0.985 with no loss event anywhere in the run.
        """
        n = int(self.episode_count.item())
        out = {
            "episode_writes": int(self.episode_writes.item()),
            "recall_fires": int(self.recall_fires.item()),
            "episode_salience_floor": (
                float(self.episode_saliences[:n].min().item()) if n else None
            ),
            "episode_admission_bar": None,
            "episode_context_similarity": None,
            "episode_age_span": None,
        }
        if self.adaptive_episodes:
            out["episode_admission_bar"] = float(
                self.salience_level.item()
                + self.surprise_k * self.salience_dev.item()
            )
            out["salience_level"] = float(self.salience_level.item())
            out["salience_dev"] = float(self.salience_dev.item())
        if n > 1:
            c = self.episode_contexts[:n].float()
            c = c / c.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sim = c @ c.T
            iu = torch.triu_indices(n, n, offset=1, device=sim.device)
            out["episode_context_similarity"] = float(sim[iu[0], iu[1]].mean())
            written = self.episode_steps[:n]
            live = written[written >= 0]
            if live.numel() > 1:
                out["episode_age_span"] = int((live.max() - live.min()).item())
        return out

    def _effective_priority(self, n: int, step: int) -> torch.Tensor:
        """Stored salience discounted by age. Without the age term any
        first-mover monopoly is permanent by construction — which is exactly
        what the v5 checkpoints show."""
        sal = self.episode_saliences[:n].clamp(min=1e-12)
        written = self.episode_steps[:n]
        age = (step - written).clamp(min=0).to(sal.dtype)
        never = written < 0
        decay = torch.exp(-age / max(self.episode_age_tau, 1e-6))
        eff = sal * decay
        # Slots written before this mechanism existed (resumed checkpoints)
        # carry no timestamp; treat them as maximally old so they are the
        # first to be recycled rather than immortal.
        eff = torch.where(never, torch.full_like(eff, 1e-12), eff)
        return eff

    def _choose_eviction_slot(self, n: int, step: int) -> int:
        """Stochastic eviction over inverse effective priority (PER's
        interpolation between greedy and uniform, alpha=0.6 by default).
        Hard argmin is degenerate here: the stored saliences sit within 2-6%
        of one another, so it was tie-breaking arbitrarily anyway, with no
        diversity guarantee in exchange."""
        eff = self._effective_priority(n, step)
        if self.eviction_alpha <= 0.0:
            return int(eff.argmin().item())
        weights = eff.clamp(min=1e-12).pow(-self.eviction_alpha)
        total = float(weights.sum())
        if not (total > 0.0) or not torch.isfinite(weights).all():
            return int(eff.argmin().item())
        return int(torch.multinomial(weights / weights.sum(), 1).item())

    def _write_episode_slot(
        self,
        idx: int,
        context: torch.Tensor,
        salience: float,
        input_pattern: torch.Tensor,
        step: int | None = None,
    ) -> None:
        self.episode_contexts[idx] = context
        if self.episode_values.dtype == torch.int8:
            snapshot = self.weight.detach()
            scale = (
                snapshot.abs().max().clamp(min=1e-8) / 127.0
            ).to(self.episode_scales.dtype)
            quantized = (
                (snapshot / scale).round().clamp(-128, 127).to(torch.int8)
            )
            self.episode_values[idx] = quantized
            self.episode_scales[idx] = scale
        else:
            self.episode_values[idx] = self.weight.detach().clone()
        self.episode_inputs[idx] = input_pattern.detach().to(
            self.episode_inputs.dtype
        )
        self.episode_saliences[idx] = salience
        if step is not None:
            self.episode_steps[idx] = step
            self.last_write_step.fill_(step)
        self.episode_writes.add_(1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)
        elif x.dim() == 2:
            x_flat = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Item #6 frozen-plasticity path (checked first, before the
        # gradient-checkpoint machinery): a grad-capable forward that does
        # NOT self-modify. The lived JEPA re-encode runs the trunk under
        # freeze_plasticity() so lived prediction error can train the
        # encoder. NB self.weight is a BUFFER, not a Parameter -- the
        # living FFN trains itself via pc_self_modify, never by backprop
        # (DO-NOT-REINVENT). So the lived gradient does not land on
        # self.weight; it flows THROUGH this frozen weight (a constant on
        # this path) to the encoder's upstream backprop Parameters
        # (attention, embeddings, layernorms) -- exactly as the corpus
        # JEPA forward already does. Episode recall is kept (under
        # no_grad) so the re-encoded latents retain the same memory-blended
        # structure perception saw; the recalled delta is detached.
        # pc_self_modify AND the episode write are skipped, so no living
        # buffer is mutated (the rank-1 invariants and
        # test_frozen_plasticity_reencode rely on this). No weight clone is
        # needed: nothing mutates self.weight in place on this path.
        #
        # Scale caveat: this path assumes the re-encode forward is NOT
        # gradient-checkpointed -- checkpoint replay happens in backward,
        # after freeze_plasticity() has exited, so a checkpointed frozen
        # forward would recompute on the normal (self-modifying) path. The
        # smoke encoder does not checkpoint; revisit before enabling
        # gradient checkpointing on the lived re-encode at GPU scale.
        if self._plasticity_frozen:
            with torch.no_grad():
                context = self._compute_context(x_flat)
                episode_delta = self._recall_episode(context)
            weight_eff = self.weight
            if episode_delta is not None:
                weight_eff = weight_eff + episode_delta
            output = x_flat @ weight_eff.T
            if len(input_shape) == 3:
                output = output.reshape(batch, seq_len, self.out_features)
            return output

        # Skip PC self-modification during gradient-checkpoint recomputation.
        # Mirrors v1's LivingLayerV6 guard — without this, enabling gradient
        # checkpointing for v2 would fire pc_self_modify twice per training
        # step (once on the original forward, once on the checkpoint replay),
        # double-applying every weight/prediction update and corrupting all
        # living state. Hits the cached recall context from the original
        # forward so the recomputed activation is bit-identical.
        from luthi.grad_checkpoint import is_recomputing
        recomputing = is_recomputing()

        # Forbidden mode combinations route through the declared matrix
        # (luthi/v2/mode_compat.py) so the whole failure surface is
        # auditable in one place. Fail loud rather than silently produce
        # wrong gradients.
        if recomputing and self.inference_steps_per_forward > 1:
            from luthi.v2.mode_compat import raise_incompatible
            raise_incompatible(
                "ipc_x_grad_checkpoint",
                extra=f"inference_steps_per_forward={self.inference_steps_per_forward}",
            )

        if recomputing:
            weight_snapshot = getattr(self, "_fwd_weight_snapshot", None)
            episode_delta = getattr(self, "_fwd_episode_delta", None)
            context = None
            if weight_snapshot is None:
                # No snapshot from an original forward: either the original
                # ran under freeze_plasticity (frozen path caches nothing)
                # or clear_forward_cache() ran before backward(). Silently
                # continuing would reuse stale state or crash cryptically;
                # a stale snapshot from an EARLIER step would mean quietly
                # wrong gradients. (Fable mode-matrix review 2026-07-15.)
                from luthi.v2.mode_compat import raise_incompatible
                raise_incompatible("recompute_without_original")
        else:
            with torch.no_grad():
                context = self._compute_context(x_flat)
                episode_delta = self._recall_episode(context)

            # The clone is essential: the PC update below modifies self.weight
            # in-place, which would break autograd version-tracking when
            # backward() needs this tensor for gradients flowing through
            # surrounding (attention) parameters. The 2026-05-10 audit
            # flagged the per-forward allocation cost (~67 MB at 4096d,
            # ~2.4 GB transient across 36 blocks) — real but unavoidable
            # without an autograd refactor (custom Function or
            # buffer-rotation pattern). Closed as "investigated, required
            # for correctness; reduction is a future architectural rework,
            # not a bugfix." Mirrors v1's LivingLayerV6 pattern.
            weight_snapshot = self.weight.clone()
            # Cache for any subsequent recompute pass.
            self._fwd_weight_snapshot = weight_snapshot
            self._fwd_episode_delta = episode_delta

        if episode_delta is not None:
            weight_snapshot = weight_snapshot + episode_delta

        output = x_flat @ weight_snapshot.T

        if recomputing:
            # Recompute path: skip self-modification entirely. The original
            # forward already mutated buffers; replaying here would corrupt them.
            if len(input_shape) == 3:
                output = output.reshape(batch, seq_len, self.out_features)
            return output

        # NaN-safety: if the forward produced non-finite values, skip PC
        # self-modification entirely so living buffers don't consume corrupt
        # data. Without this guard, a NaN/Inf in the input or weights
        # propagates into weight / prediction / set_point / momentum / etc.,
        # persisting beyond the bad batch even after the trainer skips its
        # optimizer step. One host sync per layer per batch — small relative
        # to the PC update math.
        # NB the detach() is load-bearing (Fable mode-matrix sweep,
        # 2026-07-15): isfinite on the grad-connected output makes autograd
        # pack the whole matmul output into saved tensors -- pure memory
        # waste on the plain path, and a saved-tensor COUNT mismatch under
        # non-reentrant gradient checkpointing (the recompute path skips
        # this guard), which made every checkpointed v2 forward fail at
        # backward with CheckpointError.
        if not torch.isfinite(output.detach()).all():
            # Counted (2026-08-14): silently skipping the living update is
            # correct, but silently NOT RECORDING it is how a block stops
            # learning for a whole family while every counter reads healthy.
            # Surfaced in aliveness() as `nonfinite_forward_skips`.
            self.nonfinite_forward_skips.add_(1)
            # Cache a safe zero pred_error so the inter-block top-down sweep
            # sees a valid (if no-op) signal when self-mod was skipped.
            self._last_pred_error = torch.zeros(
                self.in_features, device=output.device,
            )
            if len(input_shape) == 3:
                output = output.reshape(batch, seq_len, self.out_features)
            return output

        with torch.no_grad():
            from luthi.v2.pc_ops import pc_self_modify

            # Inverted-U gain: the resolution-progress signal is read ONCE per
            # forward from the trace state as it stands from prior forwards
            # (not this step's error, which pc_self_modify computes internally).
            # It reflects "has sustained effort been reducing error up to now",
            # a slow cross-forward trend -- and is 0.0 (fully resolving, gain
            # un-dampened) until both traces are warm. Gain off -> 0.0, unused.
            if self.learning_gain_enabled:
                from luthi.v2.slow_trace import resolution_progress
                gain_progress = resolution_progress(
                    self._err_short, self._err_long
                )
            else:
                gain_progress = 0.0

            T = self.inference_steps_per_forward
            for inner_step in range(T):
                # Recompute output for inner_step > 0 since the weight
                # has changed since the last iteration. For the first
                # iteration we already have `output` from the matmul above.
                if inner_step > 0:
                    # Use self.weight directly (mutated by previous inner
                    # step's self_modify). We don't need a fresh snapshot
                    # inside the loop — backward() flows through the
                    # initial weight_snapshot only.
                    output = x_flat @ self.weight.T

                # Compute the sparse gate for this inner step (gate state
                # depends on error_acc which evolves across inner steps).
                sparse_gate: torch.Tensor | None = None
                if (
                    self.sparse_threshold > 0.0
                    and self._sparse_step_count >= self.sparse_warmup_steps
                ):
                    sparse_gate = (self.error_acc > self.sparse_threshold).to(
                        dtype=self.weight.dtype
                    )

                # Homeostatic activity band: the sparse gate's key. The gate
                # silences rows with low error -- and a COLLAPSED row has low
                # error, so the gate would freeze it there permanently. The
                # band reopens rows that are quiet for the wrong reason.
                # `sparse_gate` is a plain per-row multiplier in both the
                # Python and C++ paths, so the band composes with it directly.
                if self.homeostatic_band_enabled:
                    sparse_gate = self._apply_activity_band(output, sparse_gate)

                result = pc_self_modify(
                    self.weight, self.prediction, self.set_point,
                    self.momentum, self.update_ema, self.precision,
                    self.error_acc, self.plasticity,
                    x_flat, output,
                    # rate_scale: the trainer-scheduled plasticity taper
                    # (run-3 build). 1.0 = legacy bit-identical; the
                    # schedule lowers the LEARNING channels only --
                    # homeostasis and set-point adaptation stay at their
                    # own rates (stability is not what tapers).
                    self.pc_rate * self.rate_scale,
                    self.pred_learning_rate * self.rate_scale,
                    self.homeostatic_decay, self.set_point_adapt_rate,
                    self.momentum_decay, self.update_ema_decay,
                    self.precision_ema_decay,
                    self.precision_min, self.precision_max,
                    self.prediction_clamp,
                    relative_trust=self.relative_trust,
                    drive_normalize=self.drive_normalize,
                    error_rms=self.error_rms,
                    drive_rms_decay=self.drive_rms_decay,
                    drive_mode=self.drive_mode,
                    drive_ref=self.drive_ref,
                    drive_ref_drift=self.drive_ref_drift,
                    drive_dev=self.drive_dev,
                    drive_calls=self.drive_calls,
                    drive_gain_out=self.drive_gain,
                    drive_fire_count=self.drive_fire_count,
                    drive_gain_sum=self.drive_gain_sum,
                    drive_decay=self.drive_decay,
                    drive_drift_gain=self.drive_drift_gain,
                    drive_surprise_k=self.drive_surprise_k,
                    drive_gain_max=self.drive_gain_max,
                    drive_gain_floor=self.drive_gain_floor,
                    drive_dev_floor_frac=self.drive_dev_floor_frac,
                    drive_warmup_calls=self.drive_warmup_calls,
                    sparse_gate=sparse_gate,
                    learning_gain_enabled=self.learning_gain_enabled,
                    learning_gain_progress=gain_progress,
                    learning_gain_rise=self.learning_gain_rise,
                    learning_gain_cap=self.learning_gain_cap,
                    return_applied_change=self.learning_gain_enabled,
                )
                if self.learning_gain_enabled:
                    salience, pred_error, applied = result
                    # Feed the applied-change sinks per inner step (each inner
                    # step is a real weight change): the raw value to the NREM
                    # day-integral, and the per-layer EMA (momentum_decay) the
                    # eye reads. Per-step cadence matches momentum's own
                    # per-inner-step update, keeping the two eye sources
                    # commensurate under iPC T>1.
                    self._applied_change_accum.add(applied)
                    self._applied_ema.update(applied)
                else:
                    salience, pred_error = result

            # Gain path only: feed the resolution traces once per forward with
            # the final inner step's prediction-error magnitude, so the next
            # forward's gain sees this forward's outcome. Off the gain path
            # these stay untouched (regime f).
            if self.learning_gain_enabled:
                err_scalar = pred_error.abs().mean().item()
                self._err_short.update(err_scalar)
                self._err_long.update(err_scalar)

            # Final-output semantics: T=1 keeps `output` from the initial
            # (grad-capable) matmul — bit-identical to the classical PC
            # behavior, backward flows through the pre-self-mod snapshot.
            # T>1's final output is recomputed grad-capably AFTER this
            # no_grad block (see below): the inner loop's rebinds happen
            # under no_grad, so returning the loop's last `output` would
            # return a tensor with NO gradient path — in a residual block
            # the FFN branch would silently vanish from backward while
            # the model kept training on the residual path alone (Fable
            # mode-matrix sweep, 2026-07-15).

            self._sparse_step_count += 1
            self._last_pred_error = pred_error
            # Mean input pattern: matches the actual_input used inside
            # pc_self_modify so the stored pattern is the same signal the
            # PC dynamics were trying to predict at this step.
            input_pattern = x_flat.mean(dim=0).detach()
            self._store_episode(context, salience, input_pattern)

            if self._consolidation_tracker is not None:
                # Audit 2026-05-11 fix: feed mean-squared-error magnitude
                # (a scalar per step) rather than pred_error.var() (which
                # is variance ACROSS input dimensions for a single step —
                # spatial, not temporal). The ConsolidationTracker holds
                # a rolling window of these scalars and compares the
                # latest to the frozen warmup baseline — so the right
                # input is "how big is the error right now" relative to
                # "how big was it during warmup". Spatial variance gave
                # the wrong signal (uniformly-large error reads as low
                # spatial variance → spurious consolidation triggers).
                err_magnitude = pred_error.pow(2).mean().item()
                if self._consolidation_tracker.step(err_magnitude):
                    # Dispatch to gradient, attractor, or both. Order
                    # for "both" is gradient first (places weight near
                    # stored regime), then attractor (reinforces the
                    # stored input as a fixed point of that regime).
                    from luthi.v2.consolidation import (
                        consolidate_layer,
                        consolidate_layer_attractor,
                    )
                    # Return values are the episode counts actually replayed
                    # (0 on an empty store). Captured, not discarded --
                    # see the counter block in __init__.
                    replayed = 0
                    if self.consolidation_style in ("gradient", "both"):
                        replayed += consolidate_layer(
                            self,
                            consolidation_rate_factor=self.consolidation_rate_factor,
                        ) or 0
                    if self.consolidation_style in ("attractor", "both"):
                        replayed += consolidate_layer_attractor(
                            self,
                            consolidation_rate_factor=self.consolidation_rate_factor,
                            n_replay_passes=self.consolidation_attractor_passes,
                        ) or 0
                    self._consolidation_fire_count += 1
                    self._consolidation_replayed_total += int(replayed)
                    if replayed == 0:
                        self._consolidation_noop_fires += 1

        # iPC (T>1) grad-path repair (Fable mode-matrix sweep, 2026-07-15):
        # the inner loop's `output` rebinds happened under no_grad, so the
        # loop's final activation carries no gradient path. Recompute it
        # grad-capably here against the POST-self-mod weight — that is the
        # function the layer actually delivered downstream, so dx must flow
        # through it (an initial-snapshot gradient would mismatch the
        # returned activation). The clone mirrors the T=1 snapshot
        # rationale: the next forward's pc_self_modify mutates self.weight
        # in place, which would break autograd version tracking on the
        # saved tensor. (iPC × gradient checkpointing stays forbidden —
        # this path never runs on a recompute replay.)
        if self.inference_steps_per_forward > 1:
            output = x_flat @ self.weight.clone().T

        if len(input_shape) == 3:
            output = output.reshape(batch, seq_len, self.out_features)
        return output

    # ------------------------------------------------------------------
    # Top-down modulation (two-layer per decision 3)
    # ------------------------------------------------------------------

    def apply_top_down(self, signal) -> None:
        """Two-channel top-down: prediction-driven modulation of plasticity
        and set_point.

        Respects freeze_plasticity (fix 2026-07-17, found preparing the
        living-full run): the top-down sweep is a living-state WRITER
        (plasticity.mul_, set_point.add_), so under the freeze it must
        no-op like every other writer -- otherwise a backward-pass-enabled
        model's held-out eval would silently modulate the substrate while
        measuring it. Pinned by the frozen x backward-pass tests.

        Signal carries:
          - prediction_error: [in_features], nudges set_point.
          - salience: [in_features], modulates plasticity.
          - modulation_strength: scalar.

        Per V2_IMPLEMENTATION_PLAN.md decision 3. Refinement 3's M2
        isolation tests verify the two channels do their jobs independently
        without destructive compounding when joint.
        """
        if self._plasticity_frozen:
            return
        if signal.salience.shape[-1] != self.in_features:
            raise ValueError(
                f"TopDownSignal.salience size {signal.salience.shape[-1]} "
                f"does not match in_features {self.in_features}."
            )
        if signal.prediction_error.shape[-1] != self.in_features:
            raise ValueError(
                f"TopDownSignal.prediction_error size "
                f"{signal.prediction_error.shape[-1]} does not match "
                f"in_features {self.in_features}."
            )

        with torch.no_grad():
            strength = signal.modulation_strength

            self.plasticity.mul_(1.0 - 0.01 * strength).add_(
                signal.salience * 0.01 * strength
            )
            # Floor relaxed from 0.1 → 0.01 on 2026-05-16 to support
            # the plasticity-partition design direction (see
            # docs/research/2026-05-16_plasticity-partitions-design.md).
            # The new floor allows weights to be 10× more stable than
            # the previous minimum — enough to express identity-anchor
            # behavior (per-step updates an order of magnitude smaller)
            # while preserving a nonzero learning rate so weights are
            # never fully frozen. Full freezing was considered (floor
            # = 0.0) but rejected: it leaves no baseline learning rate
            # if the top-down recovery signal is weak or absent, and
            # exact-zero updates can distort downstream metaplasticity
            # tracking. Biological precedent agrees — even the most
            # stable synapses retain nonzero plasticity. Upper bound
            # 10.0 retained as a safety guard against runaway
            # plasticity until there's a specific motivation to raise it.
            self.plasticity.clamp_(0.01, 10.0)

            error_signal = signal.prediction_error.unsqueeze(0)
            self.set_point.add_(
                error_signal * self.set_point_adapt_rate * 10.0 * strength
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def aliveness(self) -> dict[str, float]:
        return {
            "weight_mean": self.weight.mean().item(),
            "weight_std": self.weight.std().item(),
            "prediction_norm": self.prediction.norm().item(),
            "set_point_drift": (
                (self.weight - self.set_point).abs().mean().item()
            ),
            "momentum_magnitude": self.momentum.abs().mean().item(),
            "update_ema_mean": self.update_ema.mean().item(),
            "precision_mean": self.precision.mean().item(),
            "precision_min": self.precision.min().item(),
            "precision_max": self.precision.max().item(),
            # Trust differentiation (v5 relative-trust instrument,
            # 2026-07-21): p95/p5 of the reliability ledger. Legacy
            # saturated regime reads ~1.0 (everyone pinned); a working
            # relative-trust regime reads the real spread (13-22x
            # measured pre-fix). Quantile on CPU -- DML lacks the op.
            "precision_spread": (
                lambda p: (
                    torch.quantile(p, 0.95) / torch.quantile(p, 0.05).clamp(min=1e-12)
                ).item()
            )(self.precision.detach().float().cpu()),
            "error_acc_mean": self.error_acc.mean().item(),
            "error_acc_max": self.error_acc.max().item(),
            "episodes_stored": self.episode_count.item(),
            # Episode-store health (2026-07-27). Observational only — none of
            # these enter any loss, so there is nothing to Goodhart against.
            # Baselines measured on the v5 family BEFORE the fix:
            # context_similarity 0.985, writes 0 after ~step 1000, and three
            # of four blocks storing nothing at all.
            **self._episode_store_stats(),
            # Homeostatic band readings (observational only).
            "band_boost_rows": int(self.band_boost_rows.item()),
            "band_damp_rows": int(self.band_damp_rows.item()),
            # Silent-skip counters (2026-08-14). Cumulative, so differencing
            # two logged records gives the per-interval rate. A nonzero and
            # RISING nonfinite_forward_skips means this block's living
            # channel is intermittently off and every other buffer below is
            # reporting stale values with full confidence.
            "nonfinite_forward_skips": int(self.nonfinite_forward_skips.item()),
            "band_degenerate_skips": int(self.band_degenerate_skips.item()),
            # cos(vec(W), vec(P)) -- external review 2026-07-28, instrument #5.
            # W and P currently receive the SAME outer-product form
            # (outer(output_mean, error_at_input), pc_ops.py steps d and i),
            # differing only in precision weighting and rate. If this cosine
            # runs high, they are a fast copy and a slow copy rather than a
            # recognition/generative pair, and the proposed W-update change
            # has something to fix. If it is flat and low, it does not.
            # Weight magnitude (external review 2026-07-28, item 0.5): decides
            # whether an update of 5.3e-9 is ~1.4 ULP (arithmetically dead --
            # the update is being lost to float rounding) or ~44 ULP (small
            # but real). Without the scale, "update_ema fell 5 orders of
            # magnitude" cannot be interpreted at all.
            "weight_abs_mean": float(self.weight.detach().abs().mean().item()),
            "error_rms": float(self.error_rms.item()),
            # Surprise-drive observability (2026-07-29). `drive_duty` is the
            # discriminator: a surprise drive that is quiet because the data is
            # familiar shows a LOW duty cycle with a healthy `drive_ref`, while
            # a broken one shows duty 0 with `drive_ref` collapsed or `drive_dev`
            # at its floor. Those two were indistinguishable before, which is
            # how a self-extinguishing drive survived five families.
            "drive_gain": float(self.drive_gain.item()),
            "drive_ref": float(self.drive_ref.item()),
            "drive_dev": float(self.drive_dev.item()),
            "drive_duty": (
                float(self.drive_fire_count.item())
                / max(
                    float(self.drive_calls.item())
                    - float(self.drive_warmup_calls),
                    1.0,
                )
            ),
            # Mean gain over FIRING calls only. Read with drive_duty this
            # separates the two ways a gated drive can die: falling duty at
            # steady mean gain = firing less often; steady duty at falling mean
            # gain = firing more feebly. Different diagnoses, different fixes.
            # `drive_gain` alone cannot distinguish them -- as a point sample at
            # ~2% duty it read 0.0000 at every deep record of the first probes.
            # Both counters are cumulative, so differencing two logged records
            # gives the per-interval value.
            "drive_gain_mean_fired": (
                float(self.drive_gain_sum.item())
                / max(float(self.drive_fire_count.item()), 1.0)
            ),
            "drive_fires": float(self.drive_fire_count.item()),
            "drive_calls": float(self.drive_calls.item()),
            "weight_ulp_ratio": float(
                self.update_ema.detach().mean().item()
                / max(
                    float(
                        torch.finfo(self.weight.dtype).eps
                        * self.weight.detach().abs().mean().item()
                    ),
                    1e-45,
                )
            ),
            "weight_pred_cosine": float(
                torch.nn.functional.cosine_similarity(
                    self.weight.detach().reshape(1, -1).float(),
                    self.prediction.detach().reshape(1, -1).float(),
                ).item()
            ),
            "act_median": (
                float(self.act_var.clamp(min=0).sqrt().median().item())
                if int(self.act_count.item()) else None
            ),
            # Cumulative consolidation events (Brian's request 2026-07-18,
            # after forensically identifying a fire from its signature —
            # pred_frob step-jump + err_acc/update_ema flash with external
            # channels silent). Persists across resume via
            # living_extra_state; memory-becoming-structure, now countable.
            "consolidation_fires": float(self._consolidation_fire_count),
            # Effect, not trigger (2026-08-14). `consolidation_fires` alone
            # cannot distinguish "consolidation is working" from "the
            # trigger fired a thousand times into an empty store". These
            # two make that separable: noop_fires == fires means the
            # mechanism has done nothing at all, however healthy the fire
            # count looks.
            "consolidation_replayed_total": float(
                self._consolidation_replayed_total
            ),
            "consolidation_noop_fires": float(self._consolidation_noop_fires),
        }

    def clear_forward_cache(self) -> None:
        """Release the snapshots kept for gradient-checkpoint replay.

        Audit 2026-05-11 fix: `_fwd_weight_snapshot` and `_fwd_episode_delta`
        are cached on the layer so a gradient-checkpoint recompute can
        replay the same forward bit-identically. Between training steps
        they're stale and would otherwise hold ~67 MB per layer at 4096d
        (~2.4 GB across 36 blocks). Trainers should call this after
        optimizer.step() to free that memory. `_last_pred_error` (also
        non-persistent) is dropped similarly so cross-block top-down
        sweep state from one batch doesn't leak into the next.
        """
        self._fwd_weight_snapshot = None
        self._fwd_episode_delta = None
        self._last_pred_error = None

    def non_feedforward_signal(self, x: torch.Tensor) -> float:
        """Mean absolute difference between two consecutive identical-input
        forward passes. Positive value confirms the layer is not feedforward.
        """
        with torch.no_grad():
            out1 = self.forward(x)
            out2 = self.forward(x)
            return (out2 - out1).abs().mean().item()
