"""JEPA loss for M8 multimodal training.

Implements the v0.5 brief's multimodal JEPA objective on the v2 PC substrate.
Spec: docs/research/2026-06-06_m8-brief-v0.5.md

Components:
- Online encoder + EMA target encoder. Target tracks slow nn.Parameters only
  (B6 per v0.4): living-weight buffers, prec, and episode-store state are NOT
  EMA'd. The target's buffers evolve via its own forward passes; they are not
  synchronized with the online encoder's buffers.
- L1 invariance loss per modality, normalized by an EMA of per-modality target
  std with epsilon floor (F4: eps = 1e-3) so a genuine collapse produces a
  bounded signal-bearing loss rather than a numerical explosion.
- VICReg variance hinge (std must exceed 1.0) and covariance penalty (off-
  diagonal mass squared), both computed per modality -- single-modality
  collapse fires its own hinge directly rather than being averaged away
  under healthy modalities (F1 / F2).
- 2-layer transformer predictor with constant action-token stub prepended to
  the context K/V (M9 interface continuity).
- Disjoint 80/20 per-modality tail masking: context = positions 0..0.8 L,
  target = positions 0.8 L..L within each unimodal sequence.

The loss is computed per modality (one modality per call to
compute_modality_loss). The runner orchestrates the multimodal mixing by
calling this per modality and aggregating, which matches the brief's
per-modality VICReg formulation.

Coefficients (VICReg paper defaults, Bardes/Ponce/LeCun, ICLR 2022):
invariance = 25, variance = 25, covariance = 1.

EMA momentum constant at 0.996 (V-JEPA 2 dropped the ramp with minimal impact).
"""

from copy import deepcopy

import torch
import torch.nn as nn

from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM


# VICReg coefficients (paper defaults; see v0.5 brief §1).
VICREG_INVARIANCE_WEIGHT = 25.0
VICREG_VARIANCE_WEIGHT = 25.0
VICREG_COVARIANCE_WEIGHT = 1.0
VICREG_VARIANCE_TARGET = 1.0  # per-dim std hinge target

# Per-modality target std EMA for L_pred normalization.
TARGET_STD_EMA_MOMENTUM = 0.99
TARGET_STD_EMA_FLOOR = 1e-3  # F4: floor prevents divisor blow-up on collapse.

# EMA target-encoder momentum (V-JEPA 2: constant, no ramp).
DEFAULT_TARGET_EMA_MOMENTUM = 0.996

# Masking: disjoint 80/20 per-modality tail.
DEFAULT_CONTEXT_FRACTION = 0.8

# Modality names matching MultimodalPredictiveCodingLM.encode()'s span keys.
MODALITIES = ("text", "audio", "vision")


class JEPAPredictor(nn.Module):
    """Two-layer transformer predictor for the JEPA objective.

    Takes the online encoder's context latents (K/V), learned target-position
    queries, and a constant action-token stub appended to K/V. Returns
    predicted target latents. Kept small by design so the substrate carries
    the load, not the predictor.
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

        # Learned position embeddings for target positions (the queries).
        # Indexed by absolute position so the predictor can distinguish
        # "predict position 100" from "predict position 200".
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

    def forward(
        self,
        context_latents: torch.Tensor,
        target_positions: torch.Tensor,
        action_token: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target latents from context + target position queries.

        Args:
            context_latents: [B, L_ctx, D] online encoder's context output.
            target_positions: [B, L_tgt] integer absolute positions for the
                target span. Used to look up learned position queries.
            action_token: [D] constant stub. Prepended to context as an
                extra K/V entry so the M9 action-conditioning interface
                is structurally live (the M8 stub itself is degenerate;
                see v0.5 brief §1).

        Returns:
            [B, L_tgt, D] predicted target latents.
        """
        batch = context_latents.shape[0]

        # Action-token stub: [D] -> [B, 1, D] prepended to context K/V.
        action_kv = action_token.view(1, 1, -1).expand(batch, 1, -1)
        memory = torch.cat([action_kv, context_latents], dim=1)

        # Target queries from learned position embeddings.
        target_queries = self.target_pos_embedding(target_positions)

        # Cross-attention: queries attend to (action_kv + context).
        out = self.layers(target_queries, memory)
        return self.output_norm(out)


class JEPALoss(nn.Module):
    """Multimodal JEPA loss for M8.

    The loss is computed per modality (one modality per call to
    compute_modality_loss) so the per-modality VICReg formulation (§1, §3)
    is preserved: a single-modality collapse fires that modality's own
    variance hinge directly rather than being averaged away by healthy
    modalities' contributions.

    EMA target encoder semantics (B6 per v0.4):
    - Target encoder is a deepcopy of the online encoder at __init__.
    - update_target_ema() averages nn.Parameters only (slow weights);
      living-weight buffers, prec, and episode-store state are NOT
      synchronized -- the target's buffers evolve via its own forwards.
    - The target encoder stays in .eval() mode permanently and is called
      under torch.no_grad(). In eval mode the PC top-down sweep is gated
      off (see MultimodalPredictiveCodingLM.encode), so the target acts
      as a stable stop-grad snapshot.

    Action-token stub:
    - Registered as a non-trainable buffer of zeros at d_model. The M9
      interface will replace this with real action embeddings; the v0.5
      brief drops the action-token gradient kill-criterion because a
      constant token receives nonzero embedding gradient regardless of
      whether the predictor downstream conditions on it. Keeping the
      stub is for interface continuity only.
    """

    def __init__(
        self,
        online_encoder: MultimodalPredictiveCodingLM,
        ema_momentum: float = DEFAULT_TARGET_EMA_MOMENTUM,
        std_ema_momentum: float = TARGET_STD_EMA_MOMENTUM,
        std_ema_floor: float = TARGET_STD_EMA_FLOOR,
        context_fraction: float = DEFAULT_CONTEXT_FRACTION,
        predictor_n_layers: int = 2,
        predictor_n_heads: int | None = None,
        invariance_weight: float = VICREG_INVARIANCE_WEIGHT,
        variance_weight: float = VICREG_VARIANCE_WEIGHT,
        covariance_weight: float = VICREG_COVARIANCE_WEIGHT,
        variance_target: float = VICREG_VARIANCE_TARGET,
    ):
        super().__init__()
        self.online_encoder = online_encoder

        # B6: deepcopy creates an independent target encoder.
        # Slow params (nn.Parameter) will be EMA'd via update_target_ema().
        # Living-weight buffers are NOT synchronized to online -- the target
        # has its own buffer state which evolves via its own forwards only.
        self.target_encoder = deepcopy(online_encoder)
        # Disable gradient flow through every target parameter so the
        # target is a true stop-grad snapshot.
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        # Permanent eval mode: skips the PC top-down sweep (gated on
        # training mode in MultimodalPredictiveCodingLM.encode).
        # NOTE: nn.Module.train() recurses into children, so eval() set
        # here does NOT survive jepa_loss.train() from the runner. The
        # train() override below re-asserts target.eval() on every flip,
        # and compute_modality_loss re-asserts again before each target
        # forward (belt-and-suspenders; BLOCKER 2 from 4.8 review).
        self.target_encoder.eval()

        # HIGH (4.8 review 2026-06-06; resolves Q1): snapshot the target's
        # living-weight buffers at init. _restore_target_buffers() copies
        # these back before every target forward so the target's living
        # state stays at the controlled baseline -- v0.5 §1 calls for the
        # target's living state to be "fresh per forward (not persisted,
        # not averaged)". The target is therefore defined by its EMA'd
        # slow params, not by its own free-drifting trajectory; this is
        # the asymmetry BYOL/V-JEPA rely on for collapse resistance.
        # Filter to ``blocks.*`` so we snapshot PC layer state and episode
        # store, but not static encoder buffers (e.g. torchaudio MelScale).
        self._target_buffer_snapshots: dict[str, torch.Tensor] = {
            name: buf.detach().clone()
            for name, buf in self.target_encoder.named_buffers()
            if name.startswith("blocks.")
        }

        d_model = online_encoder.d_model
        n_heads = predictor_n_heads if predictor_n_heads is not None else online_encoder.n_heads

        # MEDIUM (4.8 review 2026-06-06): the predictor's position embedding
        # table must span the largest per-modality sequence length, because
        # target_positions indexes from ctx_len up to seq_len-1 and audio
        # sequences can reach max_audio_tokens (=1000 default), past the
        # original 512 default. Take the max across modalities + safety margin.
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

        # Action-token stub: constant zeros. Not learned in M8.
        self.register_buffer("action_token", torch.zeros(d_model))

        # Per-modality target std EMAs for L_pred normalization. Initialized
        # to ones so the first batch's L_pred is unit-scaled until the EMA
        # warms up.
        for modality in MODALITIES:
            self.register_buffer(
                f"{modality}_target_std_ema",
                torch.ones(d_model),
            )

        self.ema_momentum = ema_momentum
        self.std_ema_momentum = std_ema_momentum
        self.std_ema_floor = std_ema_floor
        self.context_fraction = context_fraction
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_target = variance_target

    @torch.no_grad()
    def update_target_ema(self) -> None:
        """Update the EMA target encoder's slow params from online (B6).

        Only nn.Parameters are touched. All buffers (living-weight state,
        prec, episode store) remain as-is on the target side.

        Call this after optimizer.step() in the training loop.
        """
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data.mul_(self.ema_momentum).add_(
                online_param.data,
                alpha=1.0 - self.ema_momentum,
            )

    def train(self, mode: bool = True) -> "JEPALoss":
        """Override to keep target_encoder in eval mode regardless of the
        parent module's training state. BLOCKER 2 fix (4.8 review
        2026-06-06): without this override, nn.Module.train() recurses
        into submodules and flips the target back to training=True, which
        re-enables the PC top-down sweep on the target and destroys the
        stable stop-grad snapshot. The target must stay eval-only across
        train()/eval() flips.
        """
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _restore_target_buffers(self) -> None:
        """Restore the target encoder's living-weight buffers to the init
        snapshots, so the target's living state is held at the controlled
        baseline (v0.5 §1; HIGH from 4.8 review 2026-06-06).

        Called before every target forward in compute_modality_loss. The
        target's slow params drift toward the online via EMA; the target's
        living-weight buffers do not drift -- they are anchored. The
        target is therefore defined by its EMA'd slow params and a fixed
        living-weight baseline.
        """
        target_buffer_map = dict(self.target_encoder.named_buffers())
        for name, snapshot in self._target_buffer_snapshots.items():
            target_buffer_map[name].data.copy_(snapshot)

    def compute_modality_loss(
        self,
        modality: str,
        modality_inputs: dict,
    ) -> dict:
        """Compute the JEPA loss for one modality's batch.

        Asymmetric encoding (BLOCKER 1 fix, 4.8 review 2026-06-06):
        - The online encoder sees **only context tokens** (positions
          0..ctx_len-1).
        - The target encoder sees the **full sequence** (positions
          0..seq_len-1).

        Encoding the full sequence into the online encoder and slicing
        latents afterward would leak: with ``causal=False`` (bidirectional
        attention, per v0.5 §10.9), every context position has already
        attended to every target position, and the predictor could read
        the answer off the context. The disjoint mask is enforced at the
        **input** to the online encoder, which is the I-JEPA standard.

        For vision/audio the slice happens on pre-encoded tokens
        (vision/audio encoders run externally on the online's encoder
        weights, then sliced tokens are fed via vision_tokens / audio_tokens
        which bypass internal re-encoding); for text, raw token IDs are
        sliced directly. The target encoder consumes whatever
        ``modality_inputs`` provides (image, waveform, or text_tokens)
        unsliced.

        Per-modality VICReg (F1/F2): L_var and L_cov are computed on this
        modality's online context latents only.

        Args:
            modality: "text", "audio", or "vision".
            modality_inputs: kwargs forwarded to encode(). For "text"
                supply ``text_tokens``; for "audio" supply
                ``audio_waveform`` or ``audio_tokens``; for "vision"
                supply ``image`` or ``vision_tokens``. Other modalities
                must be absent or None.

        Returns:
            Dict containing:
              - "loss": scalar tensor with grad (the per-modality total).
              - "l_pred", "l_var", "l_cov": detached scalar component
                tensors for logging.
              - "online_std", "target_std", "target_std_ema": [D] tensors
                of per-dim std for §5 collapse instrumentation.
              - "online_context_latents": [B, ctx_len, D] for downstream
                per-modality collapse metrics (variance, correlation,
                spectrum, LID).
              - "target_latents": [B, seq_len, D] full target latents
                (target-block region is used by the runner for
                online-vs-target cosine).
        """
        if modality not in MODALITIES:
            raise ValueError(
                f"Unknown modality {modality!r}; expected one of {MODALITIES}"
            )

        # --- Pre-tokenize per modality, slice context tokens at the input ---
        # (BLOCKER 1: keep target tokens out of the online encoder so the
        # context latents cannot have already attended to them.)
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
                # Use the ONLINE encoder's audio encoder (substrate-agnostic
                # perceptual encoder; tokens fed back through encode()
                # bypass re-encoding via the audio_tokens path).
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

        # --- Online encoder forward on context tokens only ---
        # encode(causal=False) -- bidirectional within the (context-only)
        # input; there are no target tokens for context to attend to,
        # so the disjoint mask is intact.
        online_result = self.online_encoder.encode(
            **online_inputs, causal=False,
        )
        online_context_latents = online_result["per_modality"][modality]
        # [B, ctx_len, D]
        batch = online_context_latents.shape[0]
        d_model = online_context_latents.shape[-1]

        # --- Target encoder forward on the full sequence ---
        # BLOCKER 2: re-assert eval mode in case the runner toggled
        # train/eval between calls. HIGH: restore the target's
        # living-weight buffers to the init snapshot so the target
        # stays defined by its EMA'd slow params alone.
        self.target_encoder.eval()
        self._restore_target_buffers()
        with torch.no_grad():
            target_result = self.target_encoder.encode(
                **modality_inputs, causal=False,
            )
            target_latents = target_result["per_modality"][modality]
        # [B, seq_len, D]

        # --- Predictor: predict target-block latents from online context ---
        target_block = target_latents[:, ctx_len:, :]
        target_positions = torch.arange(
            ctx_len, seq_len, device=online_context_latents.device,
        ).unsqueeze(0).expand(batch, -1)

        predicted_target = self.predictor(
            online_context_latents,
            target_positions,
            self.action_token,
        )
        # [B, tgt_len, D]

        # --- L_pred: L1 normalized by per-modality target std EMA (eps floored) ---
        target_std_ema = getattr(self, f"{modality}_target_std_ema")
        # F4: floor prevents divisor blow-up on collapse.
        target_std_floored = target_std_ema.clamp(min=self.std_ema_floor)

        abs_err = (predicted_target - target_block).abs()
        l_pred_per_dim = abs_err.mean(dim=(0, 1)) / target_std_floored  # [D]
        l_pred = l_pred_per_dim.mean()

        # In-place EMA update on the std normalizer (buffer; outside grad).
        with torch.no_grad():
            current_target_std = target_block.std(dim=(0, 1))  # [D]
            target_std_ema.mul_(self.std_ema_momentum).add_(
                current_target_std,
                alpha=1.0 - self.std_ema_momentum,
            )

        # --- L_var: VICReg variance hinge on online context latents (per modality) ---
        online_per_dim_std = online_context_latents.std(dim=(0, 1))  # [D]
        l_var = torch.relu(self.variance_target - online_per_dim_std).mean()

        # --- L_cov: VICReg off-diagonal mass squared (per modality) ---
        flat = online_context_latents.reshape(-1, d_model)  # [B*ctx_len, D]
        flat_centered = flat - flat.mean(dim=0, keepdim=True)
        n = flat_centered.shape[0]
        cov = (flat_centered.t() @ flat_centered) / max(n - 1, 1)  # [D, D]
        off_diag = cov - torch.diag(torch.diag(cov))
        l_cov = (off_diag ** 2).sum() / d_model

        # --- Per-modality total ---
        total = (
            self.invariance_weight * l_pred
            + self.variance_weight * l_var
            + self.covariance_weight * l_cov
        )

        return {
            "loss": total,
            "l_pred": l_pred.detach(),
            "l_var": l_var.detach(),
            "l_cov": l_cov.detach(),
            "online_std": online_per_dim_std.detach(),
            "target_std": current_target_std.detach(),
            "target_std_ema": target_std_ema.detach().clone(),
            "online_context_latents": online_context_latents.detach(),
            "target_latents": target_latents.detach(),
            "ctx_len": ctx_len,
        }
