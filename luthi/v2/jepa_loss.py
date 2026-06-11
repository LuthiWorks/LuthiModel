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
  embeddings. The L1+MAD reasoning we carried from V-JEPA was a
  V-JEPA anti-collapse argument; SIGReg now does the anti-collapse
  work, so MSE is the simpler, LeWM-default choice.
- Per-modality projection head (Linear -> BatchNorm1d) on the
  encoder output before SIGReg. SIGReg targets N(0, 1); BN does
  the standardization. The encoder's final LayerNorm in our trunk
  would wash out the distribution SIGReg shapes, so SIGReg must
  run on the BN-projected head, NOT on the LayerNorm'd trunk
  output. Per-modality heads preserve the F1/F2 single-modality-
  collapse protection per-modality VICReg gave us.
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

import torch
import torch.nn as nn

from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.sigreg import SIGReg


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
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.sigreg_lambd = sigreg_lambd
        self.context_fraction = context_fraction

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

        # Per-modality projection heads (Linear -> BN). The encoder's
        # final LayerNorm in the trunk would wash out the distribution
        # SIGReg shapes -- the BN-projected head is what SIGReg runs on.
        self.projection_heads = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
            )
            for modality in MODALITIES
        })

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

    def compute_modality_loss(
        self,
        modality: str,
        modality_inputs: dict,
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
        l_pred = (predicted_target - target_block).pow(2).mean()

        # ---- SIGReg: per-modality projection -> standardized -> SIGReg ----
        # Project the full-sequence target latents through this
        # modality's Linear -> BN head. BN standardizes per channel
        # across the (B * seq_len) sample dimension.
        flat = target_full_latents.reshape(-1, d_model)
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

        # ---- Diagnostics for the runner's kill criteria ----
        online_per_dim_std = online_context_latents.std(dim=(0, 1))  # [D]

        return {
            "loss": total,
            "l_pred": l_pred.detach(),
            "l_sigreg": l_sigreg.detach(),
            "online_std": online_per_dim_std.detach(),
            "online_context_latents": online_context_latents.detach(),
            "target_latents": target_full_latents.detach(),
            "predicted_target": predicted_target.detach(),
            "ctx_len": ctx_len,
        }
