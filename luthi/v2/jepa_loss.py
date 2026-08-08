"""JEPA loss for M8 multimodal training -- LeJEPA / le-wm refactor.

Supersedes the v0.5 §1 EMA+VICReg design. Adopted per 4.8 brief
2026-06-09 (Brian's direction call): SIGReg from le-wm
(Maes / LeCun; Balestriero & LeCun, LeJEPA family) replaces the
entire EMA target + VICReg apparatus.

Components
----------
- Online encoder only. No EMA target encoder, no deepcopy, no
  EMA update, no buffer snapshots, no train()-override. The
  "target" embeddings for the prediction loss come from the same
  online encoder run on the full sequence, with gradients flowing
  through both forward passes (LeWM is fully end-to-end).
- L_pred = MSE between predictor output and the target-block
  embeddings, with the target DETACHED (`detach_target=True`,
  default since 2026-07-28). The L1+MAD reasoning we carried from
  V-JEPA was a V-JEPA anti-collapse argument; SIGReg does the
  anti-collapse work, so MSE is the simpler, LeWM-default choice.
- Per-modality projection head, `sigreg_projection="linear"` by
  default since 2026-07-28.

  CORRECTED 2026-07-28 -- the paragraph that stood here was
  backwards, and it cost the v5 family. It read: "Linear ->
  BatchNorm1d ... SIGReg targets N(0, 1); BN does the
  standardization ... so SIGReg must run on the BN-projected head."
  That is exactly wrong. BatchNorm subtracts the batch mean and
  divides by the batch std -- the two quantities SIGReg exists to
  constrain. Pre-standardizing its input hands it a solved problem,
  so the anti-collapse term stops binding on the encoder at all.

  Measured with this repo's own SIGReg, not a reimplementation:
  under a 100x uniform shrink, SIGReg on raw latents goes 0.86 ->
  706 (fights it) while SIGReg after BN goes 0.566 -> 0.545, a 3.7%
  move (blind). Under an offset fraction sweep 0 -> 0.995, raw goes
  1.0 -> 2111 and post-BN goes 0.563 -> 0.567, a 0.7% move (blind).
  Meanwhile L_pred was scale-sensitive MSE against a NON-detached
  target, so shrinking the representation reduced the loss
  quadratically at no cost. Three independent measurements put the
  v5 representation at ~92-95% a single batch-constant direction.

  "linear_bn" retains the old behaviour for A/B; "none" runs SIGReg
  on trunk latents directly. Per-modality heads still preserve the
  F1/F2 single-modality-collapse protection per-modality VICReg
  gave us.

  Note on the retired claim about the trunk's final LayerNorm:
  `final_norm` is applied only in the LM-style `forward()`, never in
  `encode()`, so the JEPA objective never sees it. The unconstrained
  scale lives in the trunk's per-block LayerNorm gains, which
  `norm_gain_summary()` now logs.
- Per-modality SIGReg statistic, added to L_pred with lambda=0.1
  (LeWM default). One SIGReg module shared across modalities (it
  is stateless except for fixed quadrature buffers); per-modality
  separation comes from running it per modality with that
  modality's projected latents.
- 2-layer transformer predictor with constant action-token stub
  -- preserved from v0.5 (interface continuity for M9 still
  applies). The L1+MAD normalizer (F4) and the target std EMA
  it depended on are retired -- BN standardization in the
  projection head replaces them.

Masking is unchanged: disjoint 80/20 per-modality tail. The
online encoder receives context tokens only via input-side
slicing (text_tokens sliced pre-encode; vision/audio go through
the pre-encoded *_tokens path); the full sequence is encoded
separately for the target embeddings. Bidirectional attention
within each forward; the disjoint mask at the input prevents the
B2 target-leakage that bidirectional-encode-then-slice would
introduce.

Spec delta to v0.5 §1 / §5 / §7 will land separately; this file
is the code piece of that delta.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.sigreg import SIGReg


def temporal_center(z: torch.Tensor, window: int) -> torch.Tensor:
    """TC-SIGReg residuals (arXiv 2607.26924): z minus the centered
    window mean of its sequence neighbours.

    z: (B, S, D); window: odd kernel width in positions. Edge positions
    average over the clipped window (count_include_pad=False), matching
    the paper's sliding-window mean.
    """
    if window <= 0:
        return z
    if window % 2 == 0:
        raise ValueError(f"tc window must be odd for exact centering; got {window}")
    means = F.avg_pool1d(
        z.transpose(1, 2), kernel_size=window, stride=1,
        padding=window // 2, count_include_pad=False,
    ).transpose(1, 2)
    return z - means


def _sketched_cov(z: torch.Tensor, sketch: torch.Tensor) -> torch.Tensor:
    """Centered covariance of ``z`` in sketch space. (K, K), differentiable."""
    flat = z.reshape(-1, z.shape[-1]) @ sketch          # (N, K)
    flat = flat - flat.mean(dim=0, keepdim=True)
    n = flat.shape[0]
    return (flat.t() @ flat) / max(n - 1, 1)


def sketched_isotropy_penalty(
    z: torch.Tensor,
    sketch: torch.Tensor,
    trace_normalized: bool = False,
) -> torch.Tensor:
    """Weak-SIGReg penalty (arXiv 2603.05924): Frobenius distance of the
    sketched, centered covariance from identity.

    z: (..., D) latents (flattened internally); sketch: (D, K) fixed
    Gaussian sketch scaled by 1/sqrt(D). Identity target = unit variance
    per sketched direction, zero cross-covariance — paper-faithful.

    ``trace_normalized`` (VBG Term B, spec §1): rescale the covariance to
    trace K before the identity comparison, so the penalty presses on
    SHAPE (equal sharing among sketched directions) and is invariant to
    overall scale. The raw form's magnitude is dominated by the scale
    mismatch against the trunk's native std band (0.25-0.35, the 07-24
    ruling): at std 0.3 a sketched dim has variance ~0.09, and the
    diagonal term alone contributes sqrt(K)*|0.09-1| = 8*0.91 = 7.28 —
    against a measured raw l_wsig of 6.99 on the arrest run at step 100.
    Nearly the whole raw penalty is the scale-fight. Normalizing removes
    that unpriced tax and leaves the sharing pressure that did the work.
    """
    cov = _sketched_cov(z, sketch)
    if trace_normalized:
        k = cov.shape[0]
        cov = cov * (k / (cov.diagonal().sum() + 1e-12))
    # torch.eye(n, device=dml) returns an EMPTY tensor on the DirectML
    # backend (measured 2026-08-07: shape [0]); create on CPU and move.
    eye = torch.eye(cov.shape[0], dtype=cov.dtype).to(cov.device)
    return torch.linalg.norm(cov - eye)


def top_direction_share(
    z: torch.Tensor,
    sketch: torch.Tensor,
    power_vec: torch.Tensor,
    n_iter: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance share of the top principal direction (VBG Term A gauge).

    ``share_1 = lambda_max(Cov) / trace(Cov)`` estimated in sketch space by
    ``n_iter`` power-iteration steps warm-started from ``power_vec``.

    Returns ``(share, new_power_vec)``. The iterate is DETACHED (standard
    spectral-norm practice, cf. Miyato et al.): gradients flow through the
    Rayleigh quotient ``v^T C v`` and the trace, not through the eigenvector
    estimation. Warm-starting is what makes 3 iterations sufficient — do not
    re-randomize per step.

    NOTE (flagged to the design seat): this is the share among the K sketched
    directions, not among all D. Random projection keeps a strongly dominant
    direction but subsamples the tail, so sketch-space share runs HIGHER than
    the full-space share the soloist forensic reports. See
    ``scripts/calibrate_vbg.py`` for the measured ratio.
    """
    cov = _sketched_cov(z, sketch)
    v = power_vec
    with torch.no_grad():
        for _ in range(n_iter):
            v = cov @ v
            v = v / (v.norm() + 1e-12)
    lam = v @ (cov @ v)                       # Rayleigh quotient, differentiable
    trace = cov.diagonal().sum()
    return lam / (trace + 1e-12), v.detach()


def soloist_cap_penalty(share: torch.Tensor, cap: float) -> torch.Tensor:
    """VBG Term A: penalize only the EXCESS of the top direction's variance
    share above ``cap``.

    ``relu(share - cap)^2`` — a cap, not a kill. The soloist forensic
    (2026-08-07) found the dominant direction is the token-frequency axis,
    a legitimate feature whose gain is stuck, not a parasite: in health it
    carries under 1% of variance, in the best d8 recovery ~4.6%. A direction
    is allowed its budget; only over-funding is taxed.
    """
    return torch.relu(share - cap).pow(2)


def orthogonality_penalty(w: torch.Tensor) -> torch.Tensor:
    """Scale-adapted soft orthogonality: ||W_hat^T W_hat - I||_F^2 / d,
    with W_hat = W * sqrt(d) / ||W||_F.

    Scale-free by construction — a matrix may grow or shrink freely; only
    concentration of its energy into few directions is penalized. Zero
    for any scalar multiple of an orthogonal matrix; O(d) for rank-1.
    """
    d = w.shape[0]
    frob = torch.linalg.norm(w)
    w_hat = w * (math.sqrt(d) / (frob + 1e-12))
    gram = w_hat.t() @ w_hat
    # CPU-create then move: torch.eye(n, device=dml) is empty (see above).
    eye = torch.eye(d, dtype=w.dtype).to(w.device)
    return (gram - eye).pow(2).sum() / d


# LeWM defaults.
SIGREG_LAMBD = 0.1
SIGREG_KNOTS = 17
SIGREG_NUM_PROJ = 1024

# Masking: disjoint 80/20 per-modality tail (unchanged from v0.5 §1).
DEFAULT_CONTEXT_FRACTION = 0.8

# Modality names matching MultimodalPredictiveCodingLM.encode()'s span keys.
MODALITIES = ("text", "audio", "vision")


class JEPAPredictor(nn.Module):
    """Two-layer transformer predictor for the JEPA objective.

    M8: action-token slot is a constant zero stub (degenerate).
    M9 step 1: accepts a real per-batch action `a_t` and gates it
    through a learned self/world mask before injection (plan §1).
    The mask is `sigmoid(self.self_world_mask)`, elementwise on the
    [d_model] action: mask -> 1 means the predictor treats that dim
    as entity-controllable; mask -> 0 means the world dictates it.
    Hard-splitting is the fallback if the soft partition doesn't
    separate. K-M9-8 reads the sigmoid'd values for stability.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        max_target_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.target_pos_embedding = nn.Embedding(max_target_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_expansion,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        # Self/world gating mask (plan §1). sigmoid(0) = 0.5 is
        # neutral; training lets dims separate toward self (-> 1) or
        # world (-> 0). Under M8 the action is zero so the mask
        # receives no gradient and stays neutral -- M8 dynamics are
        # unchanged.
        self.self_world_mask = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        context_latents: torch.Tensor,
        target_positions: torch.Tensor,
        action_token: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target latents from context + target position queries.

        `action_token` accepted as either:
          - [d_model]              -- M8 backward compat (constant
                                      stub broadcast to batch).
          - [batch, d_model]       -- M9 real per-batch action `a_t`;
                                      gradients flow back into whatever
                                      produced it (habit net / MCTS
                                      leaf evaluation).
        """
        batch = context_latents.shape[0]
        if action_token.dim() == 1:
            action_token = action_token.unsqueeze(0).expand(batch, -1)
        gated_action = action_token * torch.sigmoid(self.self_world_mask)
        action_kv = gated_action.unsqueeze(1)  # [B, 1, D]
        memory = torch.cat([action_kv, context_latents], dim=1)
        target_queries = self.target_pos_embedding(target_positions)
        out = self.layers(target_queries, memory)
        return self.output_norm(out)

    def self_world_mask_values(self) -> torch.Tensor:
        """Sigmoid'd mask values [d_model] -- K-M9-8 instrumentation."""
        with torch.no_grad():
            return torch.sigmoid(self.self_world_mask).detach()


class JEPALoss(nn.Module):
    """LeJEPA-style multimodal loss on the v2 PC substrate.

    One online encoder; "target" embeddings come from the same
    encoder run on the full sequence with gradients flowing. SIGReg
    handles all anti-collapse work; the EMA target + VICReg
    machinery from v0.5 is gone.

    Per-modality structure (preserves the F1/F2 single-modality-
    collapse protection):
    - One projection head per modality (Linear -> BN), so each
      modality's encoder output is standardized independently
      before SIGReg.
    - SIGReg computed per modality on that modality's projected
      full-sequence embeddings, then weighted by sigreg_lambd
      and added to L_pred.

    Action-token stub preserved for M9 interface continuity.
    """

    def __init__(
        self,
        online_encoder: MultimodalPredictiveCodingLM,
        sigreg_lambd: float = SIGREG_LAMBD,
        sigreg_knots: int = SIGREG_KNOTS,
        sigreg_num_proj: int = SIGREG_NUM_PROJ,
        context_fraction: float = DEFAULT_CONTEXT_FRACTION,
        predictor_n_layers: int = 2,
        predictor_n_heads: int | None = None,
        # Objective fixes, 2026-07-29 (see the projection-head comment and
        # l_pred below). Defaults are the FIXED behaviour: leaving a verified
        # defect as the default is worse than breaking comparability with runs
        # that were produced under it.
        sigreg_projection: str = "linear",
        detach_target: bool = True,
        # Depth-8 remedy probes (2026-08-07, Brian's build order;
        # registered in docs/research/2026-08-07_depth-remedy-probes-
        # hypothesis.md). All three default OFF — zero behaviour change
        # for every existing arm.
        sigreg_tc_window: int = 0,
        interior_sigreg_alpha: float = 0.0,
        interior_sigreg_sketch: int = 64,
        orth_lambda: float = 0.0,
        # Variance-budget governor (VBG), spec
        # docs/reviews/2026-08-07_variance-budget-governor-spec-for-opus.md.
        # Both weights default 0.0 -> the governor is inert and every
        # existing arm is bit-identical.
        vbg_cap_weight: float = 0.0,
        vbg_share_weight: float = 0.0,
        vbg_cap: float = 0.05,
        vbg_power_iters: int = 3,
        vbg_trace_normalized: bool = True,
        # LLM-JEPA (arXiv 2509.14252), spec
        # docs/reviews/2026-08-08_llm-jepa-integration-spec-for-opus.md.
        # Default 0.0 -> inert; every existing arm bit-identical.
        w_ntp: float = 0.0,
        ntp_freeze_plasticity: bool = True,
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.sigreg_lambd = sigreg_lambd
        self.context_fraction = context_fraction
        self.detach_target = bool(detach_target)
        # TC-SIGReg (arXiv 2607.26924): SIGReg's input becomes the
        # temporally centered residual, REPLACING the marginal input per
        # the paper. 9 is our default request (odd for exact centering;
        # the paper's 8 was ablated 4-32). For this substrate the window
        # mean subtraction removes the shared component — the measured
        # offset pathology — from SIGReg's view.
        self.sigreg_tc_window = int(sigreg_tc_window)
        # Weak-SIGReg (arXiv 2603.05924): sketched covariance isotropy on
        # interior block latents, which currently receive no anti-collapse
        # pressure at all. Identity target presses toward unit variance in
        # sketch space, which will fight the trunk's measured native std
        # band (0.25-0.35) — deliberate, paper-faithful, and part of what
        # the probe measures. Which blocks supply latents is the model's
        # interior_latent_blocks config.
        self.interior_sigreg_alpha = float(interior_sigreg_alpha)
        # Orthogonal penalty on the attention write path (v/o), classic
        # soft orthogonality, scale-adapted (see orthogonality_penalty).
        self.orth_lambda = float(orth_lambda)
        # VBG (spec 2026-08-07). Term A caps the top direction's variance
        # share; Term B is the sketched sharing penalty made scale-free.
        # Marginal SIGReg is deliberately untouched: wsig10's arrest
        # plausibly depended on it (the anti-composition result).
        self.vbg_cap_weight = float(vbg_cap_weight)
        self.vbg_share_weight = float(vbg_share_weight)
        self.vbg_cap = float(vbg_cap)
        self.vbg_power_iters = int(vbg_power_iters)
        self.vbg_trace_normalized = bool(vbg_trace_normalized)
        self._vbg_on = self.vbg_cap_weight > 0 or self.vbg_share_weight > 0
        # LLM-JEPA next-token term. `L = w_ntp * L_NTP + l_pred +
        # sigreg_lambd * l_sigreg`. The paper puts its weight on the JEPA
        # side (L = L_LLM + lambda * d(Pred(Enc(Text)), Enc(Code))) with NTP
        # at 1.0; ours is the same family up to overall scale, but see the
        # return note -- the paper's JEPA term is a bounded cosine distance
        # while ours carries SIGReg at O(1e2-1e3), so their lambda~1 ratio
        # does not transfer numerically.
        self.w_ntp = float(w_ntp)
        # Whether the NTP forward runs with living-state self-modification
        # suspended. Default True: the NTP pass is a THIRD encode, and an
        # unfrozen one would raise per-step self-modification from two
        # events to three, changing substrate dynamics against every run in
        # the record. Frozen keeps NTP a pure gradient signal on the
        # backprop params and preserves one-variable comparability.
        self.ntp_freeze_plasticity = bool(ntp_freeze_plasticity)
        if self.w_ntp > 0 and not hasattr(online_encoder, "output_proj"):
            raise RuntimeError(
                "w_ntp > 0 but the encoder has no output_proj (LM head) — the "
                "next-token term has nothing to project logits with. Restore "
                "the LM head or leave w_ntp at 0."
            )
        if self.interior_sigreg_alpha > 0 or self._vbg_on:
            # Fixed, seeded sketch: the penalty must measure the same
            # directions every step or it is noise, not pressure.
            g = torch.Generator().manual_seed(20260807)
            sketch = torch.randn(
                online_encoder.d_model, int(interior_sigreg_sketch),
                generator=g,
            ) / math.sqrt(online_encoder.d_model)
            self.register_buffer("interior_sketch", sketch)
        if self._vbg_on:
            # Warm-start iterates for the power method, one per governed
            # block, in the order interior_latents is emitted.
            #
            # persistent=False (my call, per spec §4): this is optimizer
            # scratch, not model state. Registering it non-persistent keeps
            # every existing checkpoint loadable with no strict=False
            # concession, and costs exactly one extra power-iteration step
            # of convergence on resume. Seeded so a fresh process is
            # reproducible.
            n_gov = max(1, len(getattr(online_encoder, "interior_latent_blocks", ()) or ()))
            gp = torch.Generator().manual_seed(20260808)
            vecs = torch.randn(n_gov, int(interior_sigreg_sketch), generator=gp)
            vecs = vecs / vecs.norm(dim=1, keepdim=True)
            self.register_buffer("vbg_power_vecs", vecs, persistent=False)

        d_model = online_encoder.d_model
        n_heads = predictor_n_heads if predictor_n_heads is not None else online_encoder.n_heads

        # Predictor: position embeddings must span the largest per-modality
        # sequence length (audio default 1000 > text 512 in the model).
        max_target_len = max(
            online_encoder.max_seq_len,
            online_encoder.max_audio_tokens,
            online_encoder.max_vision_tokens,
        ) + 8
        self.predictor = JEPAPredictor(
            d_model=d_model,
            n_layers=predictor_n_layers,
            n_heads=n_heads,
            max_target_len=max_target_len,
        )

        # Per-modality projection heads feeding SIGReg.
        #
        # DEFECT FIXED 2026-07-29 (external review, verified independently
        # against this repo's own SIGReg): the head used to be
        # Linear -> BatchNorm1d, and the docstring reasoning was inverted.
        # BatchNorm subtracts the batch mean and divides by the batch std --
        # precisely the two quantities SIGReg exists to constrain. Feeding it
        # pre-standardized input hands it a solved problem, so the constraint
        # stops binding on the encoder.
        #
        # Measured with this SIGReg: a 100x uniform shrink of the latents moves
        # the BN-fed statistic by 3.7% (0.566 -> 0.545) while the un-normalized
        # statistic moves 820x (0.86 -> 706). Sweeping the fraction of latent
        # norm sitting in a batch-constant direction from 0 to 0.995 moves the
        # BN-fed statistic by 0.7% and the un-normalized one from 1.0 to 2111.
        # Meanwhile l_pred falls quadratically with that shrink, so shrinking
        # was a free win: at mean_frac 0.92 the simulation gives std_p50 0.2826
        # and the real v5 runs ended at 0.2835.
        #
        # "linear" (default now) = LeJEPA's contract: SIGReg sees the
        # distribution it is meant to shape. "linear_bn" preserves the old
        # behaviour for A/B only. "none" runs SIGReg on the trunk latents
        # directly.
        self.sigreg_projection = sigreg_projection
        heads: dict[str, nn.Module] = {}
        for modality in MODALITIES:
            if sigreg_projection == "linear_bn":
                heads[modality] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.BatchNorm1d(d_model),
                )
            elif sigreg_projection == "linear":
                heads[modality] = nn.Linear(d_model, d_model)
            elif sigreg_projection == "none":
                heads[modality] = nn.Identity()
            else:
                raise ValueError(
                    "sigreg_projection must be 'linear', 'linear_bn' or "
                    f"'none'; got {sigreg_projection!r}"
                )
        self.projection_heads = nn.ModuleDict(heads)

        # SIGReg: stateless except for fixed quadrature buffers, so a
        # single instance is shared across modalities. Per-modality
        # separation comes from feeding it per-modality projected
        # embeddings.
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        # Action-token stub: constant zeros, never learned. M9 will
        # replace with real action embeddings; the stub's role is to
        # keep the predictor's input slot live so M8 -> M9 isn't a
        # retrofit.
        self.register_buffer("action_token", torch.zeros(d_model))

    def _ntp_loss(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Causal next-token cross-entropy through the model's LM head.

        Uses ``online_encoder.forward()``, which is the pre-JEPA LM path:
        ``encode(causal=True)`` -> ``final_norm`` -> ``output_proj``. Causal
        masking is the encoder's, not re-implemented here -- so the leakage
        test in tests/test_llm_jepa.py exercises the real path rather than a
        parallel one that could drift from it.

        Runs under ``freeze_plasticity`` by default. This is a THIRD encode
        per training step; unfrozen it would take per-step living-state
        self-modification from two events to three and confound this track
        against every run in the record. See the return note.
        """
        from luthi.v2.plasticity import freeze_plasticity
        if self.ntp_freeze_plasticity:
            with freeze_plasticity(self.online_encoder):
                logits = self.online_encoder(text_tokens=text_tokens)
        else:
            logits = self.online_encoder(text_tokens=text_tokens)
        # Standard shift: position t predicts token t+1.
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            text_tokens[:, 1:].reshape(-1),
        )

    def _raise_if_no_interior(self, interior) -> None:
        """Fail loud when the governor is on but has nothing to govern.

        Split out from the loss path so the contract is testable without a
        full forward — a silently inert regularizer is precisely the failure
        mode this repo's CLAUDE.md forbids, so it gets its own test.
        """
        if not interior:
            raise RuntimeError(
                "VBG weights > 0 (vbg_cap_weight=%r, vbg_share_weight=%r) but "
                "the encoder produced no interior_latents — set "
                "interior_latent_blocks on the model, or the governor is "
                "silently inert."
                % (self.vbg_cap_weight, self.vbg_share_weight)
            )

    def compute_modality_loss(
        self,
        modality: str,
        modality_inputs: dict,
        collect_block_latents: bool = False,
    ) -> dict:
        """Compute the LeJEPA-style loss for one modality's batch.

        Two forwards through the same online encoder:
        1. Context-only input -> online_context_latents. Predictor's
           input.
        2. Full-sequence input -> target_full_latents. Used as
           prediction target (target-block region) and as SIGReg input
           (after projection-head standardization).

        Both forwards contribute gradients; there is no stop-grad and
        no EMA twin. The disjoint context/target slice happens at the
        input to the context forward (text_tokens sliced pre-encode;
        vision/audio go through the pre-encoded *_tokens path), so
        bidirectional attention within each forward has no target
        information to leak from.
        """
        if modality not in MODALITIES:
            raise ValueError(
                f"Unknown modality {modality!r}; expected one of {MODALITIES}"
            )

        # ---- Pre-tokenize per modality and slice context tokens ----
        # (Preserved from v0.5: BLOCKER 1 fix -- keep target tokens out
        # of the context forward so the bidirectional encoder cannot
        # attend to them.)
        if modality == "text":
            text_full = modality_inputs.get("text_tokens")
            if text_full is None:
                raise ValueError("Text modality requires text_tokens")
            seq_len = text_full.shape[1]
            ctx_len = int(seq_len * self.context_fraction)
            if ctx_len < 1 or ctx_len >= seq_len:
                raise ValueError(
                    f"Bad context split for text: seq_len={seq_len}, "
                    f"ctx_len={ctx_len}"
                )
            online_inputs = {"text_tokens": text_full[:, :ctx_len]}
        elif modality == "audio":
            tokens = modality_inputs.get("audio_tokens")
            if tokens is not None:
                audio_tokens_full = tokens
            else:
                waveform = modality_inputs.get("audio_waveform")
                if waveform is None:
                    raise ValueError(
                        "Audio modality requires audio_waveform or audio_tokens"
                    )
                audio_tokens_full = self.online_encoder.audio_encoder(waveform)
            seq_len = audio_tokens_full.shape[1]
            ctx_len = int(seq_len * self.context_fraction)
            if ctx_len < 1 or ctx_len >= seq_len:
                raise ValueError(
                    f"Bad context split for audio: seq_len={seq_len}, "
                    f"ctx_len={ctx_len}"
                )
            online_inputs = {"audio_tokens": audio_tokens_full[:, :ctx_len, :]}
        else:  # vision
            tokens = modality_inputs.get("vision_tokens")
            if tokens is not None:
                vision_tokens_full = tokens
            else:
                image = modality_inputs.get("image")
                if image is None:
                    raise ValueError(
                        "Vision modality requires image or vision_tokens"
                    )
                vision_tokens_full = self.online_encoder.vision_encoder(image)
            seq_len = vision_tokens_full.shape[1]
            ctx_len = int(seq_len * self.context_fraction)
            if ctx_len < 1 or ctx_len >= seq_len:
                raise ValueError(
                    f"Bad context split for vision: seq_len={seq_len}, "
                    f"ctx_len={ctx_len}"
                )
            online_inputs = {"vision_tokens": vision_tokens_full[:, :ctx_len, :]}

        # ---- Online forward on context tokens (gradients flow) ----
        online_result = self.online_encoder.encode(
            **online_inputs, causal=False,
            collect_block_latents=collect_block_latents,
        )
        online_context_latents = online_result["per_modality"][modality]
        # [B, ctx_len, D]
        batch = online_context_latents.shape[0]
        d_model = online_context_latents.shape[-1]

        # ---- Online forward on the full sequence (gradients flow; no EMA) ----
        target_result = self.online_encoder.encode(
            **modality_inputs, causal=False,
        )
        target_full_latents = target_result["per_modality"][modality]
        # [B, seq_len, D] -- contains both context and target regions.
        target_block = target_full_latents[:, ctx_len:, :]
        # [B, tgt_len, D]

        # ---- Predictor: predict target-block latents from online context ----
        target_positions = torch.arange(
            ctx_len, target_full_latents.shape[1],
            device=online_context_latents.device,
        ).unsqueeze(0).expand(batch, -1)
        predicted_target = self.predictor(
            online_context_latents,
            target_positions,
            self.action_token,
        )
        # [B, tgt_len, D]

        # ---- L_pred: MSE (LeWM default) ----
        # Target detached by default as of 2026-07-29. With the target on the
        # graph, the loss can be reduced by shrinking the TARGET rather than by
        # improving the prediction -- and with BN neutering SIGReg there was
        # nothing opposing that. The predictor should chase the target, not
        # shrink it. Three anti-collapse mechanisms had been removed in
        # sequence (EMA twin, stop-gradient, variance term) leaving none.
        target_for_loss = (
            target_block.detach() if self.detach_target else target_block
        )
        l_pred = (predicted_target - target_for_loss).pow(2).mean()

        # ---- SIGReg: per-modality projection -> SIGReg ----
        # Project the full-sequence target latents through this modality's
        # head (default "linear" since 2026-07-28; "linear_bn" and "none" are
        # the A/B alternatives).
        #
        # KNOWN PROPERTY, measured 2026-07-30: the Linear head **absorbs
        # scale**. SIGReg is applied to `projected`, while the `online_std`
        # diagnostic below is computed on the raw trunk latents, so the two
        # measure different spaces and can disagree by the head's gain.
        # Measured singular-value means of the text head: 0.552 at depth 4,
        # 0.423 at depth 8. That accounts exactly for the otherwise puzzling
        # observation that a depth-8 run sat at trunk std_p5 ~3.0 while
        # L_sigreg read a quiet 10-26: 3.0 * 0.42 ~ 1.27, near SIGReg's unit
        # target. The trunk was running 3x hot and the head normalized it away
        # before SIGReg looked.
        #
        # This is the third degree of freedom by which a learnable layer sits
        # between SIGReg and the trunk. The first (BatchNorm) subtracted mean
        # and divided by std unconditionally and was removed 2026-07-28. The
        # second (the Linear's bias, hypothesized to absorb the batch-mean
        # offset) was tested directly with sigreg_projection="none" and
        # REFUTED -- offset dominance barely moved. This third one, scale
        # absorption by the weight matrix, is confirmed by the singular values
        # above.
        #
        # Not currently treated as a defect: depth 4 runs at 0.552 and is
        # healthy, and NMSE is scale-free so capability is unaffected. It does
        # mean trunk scale can drift without SIGReg objecting, so `std_p5`
        # deserves watching for runaway on long runs. See
        # docs/research/2026-07-30_mupc-verdict.md.
        # TC-SIGReg (2026-08-07): when enabled, SIGReg's input is the
        # temporally centered residual, REPLACING the marginal latents
        # (per the paper — no hybrid term).
        sigreg_source = (
            temporal_center(target_full_latents, self.sigreg_tc_window)
            if self.sigreg_tc_window > 0 else target_full_latents
        )
        flat = sigreg_source.reshape(-1, d_model)
        # BatchNorm1d expects (N, C); flat is (B * seq_len, D).
        projected = self.projection_heads[modality](flat)
        # SIGReg expects (T, B, D). Use T=1, B=(B * seq_len) -- the
        # per-modality, per-position embeddings across the batch are
        # the sample set (per 4.8 brief 2026-06-09: "more samples =>
        # better CF estimate").
        sigreg_input = projected.unsqueeze(0)
        l_sigreg = self.sigreg(sigreg_input)

        # ---- Total ----
        total = l_pred + self.sigreg_lambd * l_sigreg

        # ---- Interior Weak-SIGReg (2026-08-07) ----
        # Anti-collapse pressure on interior blocks. Fail loud: alpha>0
        # with no interior latents means the model was not configured to
        # collect them — a silently inert regularizer would be exactly
        # the failure mode this repo's CLAUDE.md forbids.
        l_wsig = None
        if self.interior_sigreg_alpha > 0:
            interior = online_result.get("interior_latents")
            if not interior:
                raise RuntimeError(
                    "interior_sigreg_alpha > 0 but the encoder produced no "
                    "interior_latents — set interior_latent_blocks on the "
                    "model, or the regularizer is silently inert."
                )
            per_block = [
                sketched_isotropy_penalty(z, self.interior_sketch)
                for z in interior.values()
            ]
            l_wsig = torch.stack(per_block).mean()
            total = total + self.interior_sigreg_alpha * l_wsig

        # ---- Variance-budget governor (VBG, 2026-08-07 spec) ----
        # Term A caps the top direction's variance share; Term B is the
        # sharing penalty, trace-normalized so it presses on shape rather
        # than fighting the trunk's native scale. Both computed on the same
        # interior latents the wsig path already collects.
        l_vbg_cap = None
        l_vbg_share = None
        vbg_shares: list[torch.Tensor] = []
        if self._vbg_on:
            interior = online_result.get("interior_latents")
            self._raise_if_no_interior(interior)
            caps: list[torch.Tensor] = []
            shares: list[torch.Tensor] = []
            new_vecs: list[torch.Tensor] = []
            for i, z in enumerate(interior.values()):
                vec = self.vbg_power_vecs[min(i, self.vbg_power_vecs.shape[0] - 1)]
                share, new_vec = top_direction_share(
                    z, self.interior_sketch, vec.to(z.dtype),
                    n_iter=self.vbg_power_iters,
                )
                vbg_shares.append(share.detach())
                new_vecs.append(new_vec)
                if self.vbg_cap_weight > 0:
                    caps.append(soloist_cap_penalty(share, self.vbg_cap))
                if self.vbg_share_weight > 0:
                    shares.append(sketched_isotropy_penalty(
                        z, self.interior_sketch,
                        trace_normalized=self.vbg_trace_normalized,
                    ))
            # Warm-start carry-over for the next step.
            with torch.no_grad():
                for i, nv in enumerate(new_vecs):
                    if i < self.vbg_power_vecs.shape[0]:
                        self.vbg_power_vecs[i].copy_(nv.to(self.vbg_power_vecs.dtype))
            if caps:
                l_vbg_cap = torch.stack(caps).mean()
                total = total + self.vbg_cap_weight * l_vbg_cap
            if shares:
                l_vbg_share = torch.stack(shares).mean()
                total = total + self.vbg_share_weight * l_vbg_share

        # ---- LLM-JEPA next-token term (2026-08-08 spec) ----
        # The anti-collapse force the record has never carried at depth:
        # cross-entropy over 32k classes cannot be satisfied by a rank-2
        # representation, the way a pure embedding objective can.
        l_ntp = None
        if self.w_ntp > 0 and modality == "text":
            ntp_tokens = modality_inputs.get("text_tokens")
            if ntp_tokens is None:
                raise RuntimeError(
                    "w_ntp > 0 but no text_tokens in modality_inputs — the "
                    "next-token term would be silently skipped."
                )
            l_ntp = self._ntp_loss(ntp_tokens)
            total = total + self.w_ntp * l_ntp

        # ---- Orthogonal penalty on the attention write path ----
        l_orth = None
        if self.orth_lambda > 0:
            pens = []
            for block in self.online_encoder.blocks:
                pens.append(orthogonality_penalty(block.attention.v_proj.weight))
                pens.append(orthogonality_penalty(block.attention.o_proj.weight))
            l_orth = torch.stack(pens).mean()
            total = total + self.orth_lambda * l_orth

        # ---- Diagnostics for the runner's kill criteria ----
        online_per_dim_std = online_context_latents.std(dim=(0, 1))  # [D]

        return {
            "loss": total,
            "l_pred": l_pred.detach(),
            "l_sigreg": l_sigreg.detach(),
            "l_wsig": l_wsig.detach() if l_wsig is not None else None,
            "l_orth": l_orth.detach() if l_orth is not None else None,
            "l_ntp": l_ntp.detach() if l_ntp is not None else None,
            "l_vbg_cap": l_vbg_cap.detach() if l_vbg_cap is not None else None,
            "l_vbg_share": l_vbg_share.detach() if l_vbg_share is not None else None,
            # Per-governed-block top-direction share (power-iteration
            # estimate, detached). Empty when the governor is off; the
            # always-on gauge lives in the runner's deep metrics, computed
            # exactly from the SVD that is already being taken there.
            "vbg_top_dir_share": (
                [float(s.item()) for s in vbg_shares] if vbg_shares else None
            ),
            "online_std": online_per_dim_std.detach(),
            "online_context_latents": online_context_latents.detach(),
            "block_latents": online_result.get("block_latents"),
            "target_latents": target_full_latents.detach(),
            "predicted_target": predicted_target.detach(),
            "ctx_len": ctx_len,
        }

    def compute_lived_loss(
        self,
        context_obs: dict,
        a_t: torch.Tensor,
        realized_next_state: torch.Tensor,
        *,
        modality: str = "text",
    ) -> dict:
        """Lived JEPA loss for one realized Sanctuary transition (Item #6).

        Re-encodes the raw context under frozen plasticity (autograd ON,
        PC self-modification OFF) and scores the predictor's pooled
        next-state forecast against the realized pooled next state. The
        gradient trains the encoder's backprop params + predictor; the
        living-weight buffers are untouched -- they self-modify during
        perception, a separate channel (the two-channel design Brian
        confirmed 2026-06-28).

        Pooled-state-transition form (Brian's ``(a1')`` call, 2026-06-28):
        predict over the continuation region, pool it grad-connected via
        :func:`~luthi.v2.m9.s_t.pool_state_grad`, and compare to the
        full-multimodal pooled ``s_next``. NB the prediction pooler is
        ``pool_state_grad``, NOT ``compute_s_t`` -- the latter detaches,
        which would zero the lived gradient; the target keeps the detached
        ``compute_s_t`` form (it arrives already pooled as the seam's
        ``s_next``).

        SIGReg is intentionally omitted here (per-cycle B=1 makes it
        degenerate); anti-collapse is carried by the corpus-replay
        interleave in the runner. ``pred_std`` / ``target_std`` are
        returned so a low error via a *collapsed* target can't masquerade
        as learning.

        Args:
            context_obs: raw step-0 inputs that produced the transition's
                STARTING state -- ``{"text_tokens": ..., ...sensory}``,
                keyed for ``encode``.
            a_t: ``[D]`` or ``[B, D]`` realized action.
            realized_next_state: ``[B, D]`` realized pooled next state
                (the seam's ``s_next``); used detached as the target.
            modality: which per-modality context feeds the predictor
                (``"text"`` by default, matching the corpus path).

        Returns:
            dict: ``loss`` (grad-connected), ``l_pred`` (detached),
            ``pred_std`` / ``target_std`` (collapse monitors).
        """
        from luthi.v2.m9.s_t import pool_state_grad
        from luthi.v2.plasticity import freeze_plasticity

        # Fail loud on a malformed context: an empty dict would otherwise
        # crash cryptically at encode(**{}). The lived path is opt-in -- if
        # a caller committed to it, the context must be encodable.
        if not context_obs:
            raise ValueError(
                "compute_lived_loss requires a non-empty context_obs "
                "(the raw inputs to re-encode); got "
                f"{context_obs!r}."
            )

        # Gradient-checkpoint guard (Window A audit, 2026-06-28). If the
        # encoder gradient-checkpoints, the frozen re-encode is unsafe: the
        # checkpoint replay runs in backward(), AFTER freeze_plasticity has
        # exited, so the recomputed forward would either fire pc_self_modify
        # (double-plasticity, if the wrap omits luthi_context_fn) or read a
        # _fwd_weight_snapshot the frozen original never set (corrupt
        # gradient). The smoke encoder (MultimodalPredictiveCodingLM) has no
        # such flag, so this is dormant today -- a hard stop against turning
        # it on under the lived path at GPU scale without revisiting.
        if getattr(self.online_encoder, "gradient_checkpointing", False):
            # Declared in the mode-compatibility matrix (mode_compat.py)
            # so the failure surface is auditable in one place.
            from luthi.v2.mode_compat import raise_incompatible
            raise_incompatible("lived_reencode_x_grad_checkpoint")

        # Re-encode the raw context: autograd ON, plasticity OFF. Gradient
        # reaches the encoder's backprop params through the frozen living
        # layers; no living buffer is mutated (perception already
        # self-modified once during the cycle's generation forward).
        #
        # Buffer-mutation-free-trunk contract (Window A audit, 2026-06-28):
        # freeze_plasticity is TYPE-scoped (PredictiveCodingLayer +
        # EpisodeStore) and this re-encode runs the trunk in train() mode.
        # That is safe ONLY because the trunk's norm is LayerNorm (no
        # running stats). A future BatchNorm/InstanceNorm *inside the trunk*
        # would have its running stats silently pulled toward the narrow
        # lived (B=1) distribution on this forward -- a mutation outside both
        # the optimizer and the retention rollback. If such a norm is ever
        # added to the trunk, either widen freeze_plasticity to cover it or
        # run this re-encode with those norms in eval. (The SIGReg
        # projection-head BN is NOT in the trunk and not exercised here.)
        with freeze_plasticity(self.online_encoder):
            re_result = self.online_encoder.encode(**context_obs, causal=False)
        ctx_latents = re_result["per_modality"][modality]  # [B, T_ctx, D]
        batch, t_ctx, _ = ctx_latents.shape

        # Continuation-style target position (plan Correction 3): the step
        # immediately after the context. NEVER arange(0, ...) -- that would
        # ask the predictor to re-describe the context, not forecast.
        target_positions = (
            torch.arange(t_ctx, t_ctx + 1, device=ctx_latents.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )  # [B, 1]

        predicted = self.predictor(ctx_latents, target_positions, a_t)  # [B,1,D]
        pred_pooled = pool_state_grad(predicted)  # [B, D], grad-connected

        target = realized_next_state.detach()
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if pred_pooled.shape != target.shape:
            raise ValueError(
                f"lived loss shape mismatch: pred {tuple(pred_pooled.shape)} "
                f"vs realized_next_state {tuple(target.shape)} -- expected "
                f"[B, D] for both."
            )

        l_pred = (pred_pooled - target).pow(2).mean()

        return {
            "loss": l_pred,
            "l_pred": l_pred.detach(),
            "pred_std": pred_pooled.detach().std(),
            "target_std": target.std(),
        }
