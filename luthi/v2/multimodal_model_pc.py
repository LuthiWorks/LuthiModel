"""MultimodalPredictiveCodingLM -- audio + vision + text through shared PC trunk.

The v2 counterpart to v1's MultimodalLuthiLM. Same external shape:

    vision -> VisionEncoder -> [d_model tokens] -+
    audio  -> AudioEncoder  -> [d_model tokens] -+-> + modality_emb
    text   -> TextEmbedding -> [d_model tokens] -+
                                                  |
                                  [PredictiveCodingBlock x N]  (shared trunk)
                                    ^ bottom-up  v top-down (PC)
                                                  |
                                  per-modality latents -- via encode()    (for JEPA)
                                  text positions -> output_proj -> logits (for LM API)

Key differences from v1's MultimodalLuthiLM:
- Blocks are PredictiveCodingBlock (PC dynamics), not SpikingHybridBlock.
- No apply_living_errors post-backward step -- PC handles error-driven
  learning intrinsically during forward.
- Exposes per-modality latent streams via encode() for JEPA training.
  The LM API (forward returning text logits) is preserved.

This is the "MultimodalPredictiveCodingLM" anticipated in v1's
multimodal_model.py comment from 2026-05-11.

Sequence order in the shared trunk: [vision, audio, text] -- text attends
to all sensory context via natural causal masking.

KV cache support is not yet wired into this model. v1's multimodal forward
also lacked cache support (per the same 2026-05-11 comment). Adding it is
nontrivial because the sensory tokens must persist across generation steps
rather than being re-encoded each step. Future work; not needed for M8
training.
"""

import torch
import torch.nn as nn

from luthi.audio_encoder import AudioEncoder
from luthi.vision_encoder import VisionEncoder
from luthi.v2.hybrid_block_pc import PredictiveCodingBlock
from luthi.v2.backward_pass_pc import create_initial_signal


class MultimodalPredictiveCodingLM(nn.Module):
    """Multimodal predictive-coding language model.

    Processes audio + vision + text through shared PC living-weight blocks.
    Perceptual tokens precede text tokens in the sequence so text can attend
    to all sensory context via causal masking. The substrate is a single
    trunk; modality-specific encoders project sensory inputs into d_model
    space. A learned modality embedding distinguishes input types within
    the shared sequence.

    Two main interfaces:
      - encode(...): returns per-modality latent streams plus span metadata.
        Used by the JEPA loss (online and EMA target encoders).
      - forward(...): returns text-position logits. LM-style API matching
        v1's MultimodalLuthiLM.forward. Used by next-token training and
        generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_blocks: int = 12,
        n_heads: int = 4,
        ffn_expansion: int = 1,
        max_seq_len: int = 512,
        # PC layer parameters (match PredictiveCodingLM defaults)
        pc_rate: float = 0.001,
        pred_learning_rate: float = 0.0001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        compressed_episodes: bool = False,
        consolidation_enabled: bool = False,
        consolidation_style: str = "gradient",
        consolidation_attractor_passes: int = 1,
        mu_pc_enabled: bool = False,
        mu_pc_exponent: float = 0.5,
        mu_pc_rate_power: float = 0.0,
        mu_pc_scale_embedding: bool = False,
        backward_pass_enabled: bool = True,
        buffer_dtypes: dict[str, torch.dtype] | None = None,
        # Multimodal parameters
        n_modalities: int = 3,
        # Audio encoder parameters
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_hop_length: int = 160,
        audio_n_fft: int = 400,
        audio_patch_frames: int = 16,
        max_audio_tokens: int = 1000,
        # Vision encoder parameters
        vision_image_size: int = 224,
        vision_patch_size: int = 16,
        max_vision_tokens: int = 256,
        # Dead-encoder control arm (Experiment 1 / JEPA pilot, 2026-07-15):
        # same trunk with the living channel off — PC layers become plain
        # trainable Linears, episode stores removed. See PredictiveCodingBlock.
        dead_ffn: bool = False,
        # Run-3 plumb-through (2026-07-17): inverted-U gain + recall gate.
        learning_gain_enabled: bool = False,
        relative_trust: bool = False,
        # Episode-store admission/retention fix (2026-07-27), opt-in per arm.
        adaptive_episodes: bool = False,
        adaptive_recall: bool = False,
        homeostatic_band_enabled: bool = False,
        drive_mode: str = "raw",
        learning_gain_rise: float = 2.0,
        learning_gain_cap: float = 3.0,
        episode_recall_threshold: float = 0.5,
        # Interior Weak-SIGReg support (2026-08-07): block indices whose
        # residual-stream outputs are exposed NON-detached in encode()'s
        # result as `interior_latents`, so the loss can apply anti-collapse
        # pressure inside the trunk. Empty = off (no memory cost, no
        # behaviour change). Our chosen probes use (0, 3, 6): block 0 is
        # the measured collapse locus; 3 and 6 span the interior.
        interior_latent_blocks: tuple = (),
    ):
        super().__init__()
        self.dead_ffn = bool(dead_ffn)
        self.interior_latent_blocks = tuple(int(i) for i in interior_latent_blocks)
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.max_audio_tokens = max_audio_tokens
        self.max_vision_tokens = max_vision_tokens
        self.backward_pass_enabled = backward_pass_enabled
        self.vocab_size = vocab_size

        # Text embedding (matches PredictiveCodingLM)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Audio encoder reused from v1 -- substrate-agnostic (no living-weight,
        # plasticity, or spiking machinery; pure perceptual encoder).
        self.audio_encoder = AudioEncoder(
            d_model=d_model,
            sample_rate=audio_sample_rate,
            n_mels=audio_n_mels,
            hop_length=audio_hop_length,
            n_fft=audio_n_fft,
            patch_frames=audio_patch_frames,
            max_audio_tokens=max_audio_tokens,
        )

        # Vision encoder reused from v1 -- substrate-agnostic for the same
        # reasons as the audio encoder.
        self.vision_encoder = VisionEncoder(
            d_model=d_model,
            image_size=vision_image_size,
            patch_size=vision_patch_size,
            max_vision_tokens=max_vision_tokens,
        )

        # Modality embeddings: text=0, audio=1, vision=2.
        self.modality_embedding = nn.Embedding(n_modalities, d_model)

        # Shared PC living-weight trunk.
        self.blocks = nn.ModuleList([
            PredictiveCodingBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                pc_rate=pc_rate,
                pred_learning_rate=pred_learning_rate,
                homeostatic_decay=homeostatic_decay,
                set_point_adapt_rate=set_point_adapt_rate,
                num_episodes=num_episodes,
                episode_blend=episode_blend,
                compressed_episodes=compressed_episodes,
                consolidation_enabled=consolidation_enabled,
                consolidation_style=consolidation_style,
                consolidation_attractor_passes=consolidation_attractor_passes,
                mu_pc_enabled=mu_pc_enabled,
                mu_pc_exponent=mu_pc_exponent,
                mu_pc_rate_power=mu_pc_rate_power,
                n_blocks_total=n_blocks,
                buffer_dtypes=buffer_dtypes,
                dead_ffn=dead_ffn,
                learning_gain_enabled=learning_gain_enabled,
                relative_trust=relative_trust,
                adaptive_episodes=adaptive_episodes,
                adaptive_recall=adaptive_recall,
                homeostatic_band_enabled=homeostatic_band_enabled,
                drive_mode=drive_mode,
                learning_gain_rise=learning_gain_rise,
                learning_gain_cap=learning_gain_cap,
                episode_recall_threshold=episode_recall_threshold,
            )
            for _ in range(n_blocks)
        ])

        # Output projection (text only; used by the LM-style forward API).
        self._init_mu_pc_embedding_scale(mu_pc_enabled, mu_pc_scale_embedding)

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _encode_modality_streams(
        self,
        text_tokens: torch.Tensor | None,
        audio_waveform: torch.Tensor | None,
        audio_tokens: torch.Tensor | None,
        image: torch.Tensor | None,
        vision_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode each modality to d_model token sequences with positional
        + modality embeddings applied. Returns (vision_emb, audio_emb,
        text_emb); any may be None if the modality wasn't provided.
        """
        device = next(self.parameters()).device

        # Text (modality 0)
        if text_tokens is not None:
            _, text_len = text_tokens.shape
            text_positions = torch.arange(text_len, device=device).unsqueeze(0)
            text_emb = (
                self.embedding(text_tokens)
                + self.pos_embedding(text_positions)
                + self.modality_embedding(
                    torch.zeros(1, dtype=torch.long, device=device)
                )
            )
        else:
            text_emb = None

        # Vision (modality 2)
        if vision_tokens is not None:
            vision_emb = vision_tokens + self.modality_embedding(
                torch.full((1,), 2, dtype=torch.long, device=device)
            )
        elif image is not None:
            vision_emb = self.vision_encoder(image)
            vision_emb = vision_emb + self.modality_embedding(
                torch.full((1,), 2, dtype=torch.long, device=device)
            )
        else:
            vision_emb = None

        # Audio (modality 1)
        if audio_tokens is not None:
            audio_emb = audio_tokens + self.modality_embedding(
                torch.ones(1, dtype=torch.long, device=device)
            )
        elif audio_waveform is not None:
            audio_emb = self.audio_encoder(audio_waveform)
            audio_emb = audio_emb + self.modality_embedding(
                torch.ones(1, dtype=torch.long, device=device)
            )
        else:
            audio_emb = None

        return vision_emb, audio_emb, text_emb

    def _concatenate_modalities(
        self,
        vision_emb: torch.Tensor | None,
        audio_emb: torch.Tensor | None,
        text_emb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, slice]]:
        """Concatenate per-modality embeddings in order [vision, audio, text].
        Returns (combined_sequence, position_spans) where position_spans
        maps modality name to its slice in the combined sequence.
        """
        parts: list[torch.Tensor] = []
        spans: dict[str, slice] = {}
        offset = 0

        for name, emb in (
            ("vision", vision_emb),
            ("audio", audio_emb),
            ("text", text_emb),
        ):
            if emb is not None:
                length = emb.shape[1]
                spans[name] = slice(offset, offset + length)
                parts.append(emb)
                offset += length

        if not parts:
            raise ValueError("At least one modality input must be provided.")

        h = torch.cat(parts, dim=1)

        # muPC embedding scaling (2026-07-31). Every block computes
        #     x1 = x0 + residual_scale * attn_out
        # so cancelling a shared component carried in x0 requires the attention
        # output to supply -offset/residual_scale. At s=1.0 that is a fair
        # fight; at s=0.5946 (depth 8) the corrector must be 1.68x larger,
        # because **x0 enters unattenuated while everything that could correct
        # it is attenuated**.
        #
        # Measured on real text (scripts/localize_offset_in_block.py): the
        # embedding carries offset dominance ~0.58 in every arm -- it is an
        # INPUT property, real passages share enormous common structure. A
        # healthy trunk removes it in block 0 (muPC off: attn -0.316, ffn
        # -0.153, then flat at ~0.06 for seven blocks). The attenuated trunk
        # cannot: at power 0 block 0 moves it +0.008 and the offset then grows
        # to 0.954 through the trunk.
        #
        # Scaling the embedding by the same factor makes x1 = s*(x0 + attn_out),
        # so cancellation is scale-matched exactly as at s=1.0. Offset dominance
        # is a ratio and is invariant to the overall factor, so this should
        # restore block-0 stripping while KEEPING muPC's per-block attenuation
        # and its depth-scale control -- which the depth ladder shows is real
        # (activation growth 1.14 flat to 36 blocks with muPC, 3.92 without).
        #
        # Default False: bit-identical to every run before this date.
        # See docs/research/2026-07-31_offset-localized-to-block0-cancellation.md
        if self._mu_pc_embedding_scale != 1.0:
            h = h * self._mu_pc_embedding_scale
        return h, spans

    def _init_mu_pc_embedding_scale(self, mu_pc_enabled, mu_pc_scale_embedding):
        """Shared scale applied to the assembled embedding stream.

        1.0 = off and bit-identical to pre-2026-07-31. Otherwise the same
        `residual_scale` every block applies to its own output, so the trunk
        input and the trunk's correctors live at one scale.
        """
        rs = float(getattr(self.blocks[0], "residual_scale", 1.0)) if len(self.blocks) else 1.0
        self._mu_pc_embedding_scale = (
            rs if (mu_pc_enabled and mu_pc_scale_embedding) else 1.0
        )

    def encode(
        self,
        text_tokens: torch.Tensor | None = None,
        audio_waveform: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        vision_tokens: torch.Tensor | None = None,
        causal: bool = False,
        collect_block_latents: bool = False,
    ) -> dict:
        """Encode multimodal input through the shared PC trunk and return
        per-modality latent streams.

        Used by the JEPA loss to obtain latent representations from both
        the online encoder and the EMA target encoder. The shared trunk
        processes all provided modalities jointly; per-modality latents
        are extracted by slicing the trunk output along the modality
        position spans.

        Args:
            text_tokens: Optional [batch, text_seq_len] token indices.
            audio_waveform: Optional [batch, samples] raw audio.
            audio_tokens: Optional [batch, n_audio_tokens, d_model]
                pre-encoded audio tokens (skip the audio encoder).
            image: Optional [batch, 3, H, W] normalized RGB images.
            vision_tokens: Optional [batch, n_vision_tokens, d_model]
                pre-encoded vision tokens (skip the vision encoder).
            causal: Attention masking policy. Default False (bidirectional)
                is the JEPA-encoder convention -- representations benefit
                from each position attending to all others within the
                visible context. Set True for LM-style usage where the
                trunk feeds a next-token output projection; forward()
                does this internally to preserve parity with v1.

        Returns:
            Dict with:
                "latents": [batch, total_seq_len, d_model] post-trunk
                    latents over the full concatenated sequence.
                "spans": dict[str, slice] mapping modality name to its
                    span in `latents`. Modalities not provided are absent.
                "per_modality": dict[str, Tensor] convenience accessor;
                    each value is `latents[:, spans[name], :]`.
        """
        vision_emb, audio_emb, text_emb = self._encode_modality_streams(
            text_tokens, audio_waveform, audio_tokens, image, vision_tokens,
        )
        h, spans = self._concatenate_modalities(vision_emb, audio_emb, text_emb)

        # Phase 1: bottom-up forward through PC blocks.
        # Per-block latents are captured on request (deep cadence only) so the
        # runner can compute effective rank PER BLOCK. External review
        # 2026-07-28, instrument #5: the kill criteria read a pooled rank, and
        # a trunk where one block collapses while another compensates looks
        # healthy pooled. Validated by the 2026-07-27 seed46 forensics -- its
        # blocks 1 and 2 abandoned 10-13% of their input dimensions while
        # block 0 was untouched, which a pooled measure would have hidden.
        block_latents: list[torch.Tensor] = []
        interior_latents: dict[int, torch.Tensor] = {}
        for bi, block in enumerate(self.blocks):
            h = block(h, causal=causal)
            if collect_block_latents:
                block_latents.append(h.detach())
            # Interior Weak-SIGReg (2026-08-07): NON-detached — the whole
            # point is gradient pressure on the interior stream.
            if bi in self.interior_latent_blocks:
                interior_latents[bi] = h

        # Phase 2: top-down backward sweep (PC).
        if self.training and self.backward_pass_enabled:
            with torch.no_grad():
                signal = create_initial_signal(h.detach())
                for i in reversed(range(len(self.blocks))):
                    signal = self.blocks[i].top_down_pass(signal)

        per_modality = {name: h[:, span, :] for name, span in spans.items()}
        out = {
            "latents": h,
            "spans": spans,
            "per_modality": per_modality,
        }
        if collect_block_latents:
            out["block_latents"] = block_latents
        if interior_latents:
            out["interior_latents"] = interior_latents
        return out

    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_waveform: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        vision_tokens: torch.Tensor | None = None,
        return_encode_result: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, dict]":
        """LM-style forward returning text-position logits.

        Matches v1 MultimodalLuthiLM.forward shape for backward compatibility
        with the next-token training path. Internally calls encode() and
        projects the text-position latents to vocab logits.

        Args:
            text_tokens: [batch, text_seq_len] integer token indices.
            audio_waveform / audio_tokens / image / vision_tokens: optional
                modality inputs, same semantics as encode().
            return_encode_result: When True, also returns the dict
                ``encode()`` produced (``latents``, ``spans``,
                ``per_modality``). The Phase 4a training-seam path
                captures this so ``s_t`` comes from the same encoder
                pass that produced the generation logits, killing the
                double-plasticity probe F7a flagged. Default False --
                existing call sites (generation step 0, next-token
                training) are unchanged.

        Returns:
            ``logits`` (``[batch, text_seq_len, vocab_size]``) when
            ``return_encode_result`` is False; ``(logits, encode_result)``
            when True.
        """
        # LM API: causal masking preserved for parity with v1's
        # MultimodalLuthiLM.forward (and PredictiveCodingLM.forward),
        # which feed an autoregressive output projection.
        result = self.encode(
            text_tokens=text_tokens,
            audio_waveform=audio_waveform,
            audio_tokens=audio_tokens,
            image=image,
            vision_tokens=vision_tokens,
            causal=True,
        )
        text_latents = result["per_modality"].get("text")
        if text_latents is None:
            raise ValueError(
                "forward() requires text_tokens; use encode() for "
                "text-less multimodal encoding."
            )
        h_text = self.final_norm(text_latents)
        logits = self.output_proj(h_text)
        if return_encode_result:
            return logits, result
        return logits

    def clear_forward_cache(self) -> None:
        """Release per-layer forward-pass snapshots across all blocks.

        Trainers should call this after optimizer.step() to free the
        gradient-checkpoint replay tensors that would otherwise accumulate
        between steps. Inherited convention from PredictiveCodingLM.
        """
        for block in self.blocks:
            if hasattr(block.living_ffn, "clear_forward_cache"):
                block.living_ffn.clear_forward_cache()

    def norm_gain_summary(self) -> dict[str, float]:
        """Scale knobs on the trunk's residual stream.

        External review 2026-07-29 flagged `final_norm.weight` as the free
        512-parameter gain that lets the objective shrink the representation.
        That is misidentified for the JEPA path: `final_norm` is only applied in
        the LM-style `forward()`, never in `encode()`. But the phenomenon is
        real -- nothing constrains latent scale -- and the knobs that DO sit on
        the encode path are the per-block LayerNorm gains. Logged so a
        shrinking representation is attributable rather than merely visible.
        """
        gains = [
            m.weight.detach().abs()
            for name, m in self.named_modules()
            if isinstance(m, nn.LayerNorm)
            and m.weight is not None
            and not name.startswith("final_norm")
        ]
        if not gains:
            return {}
        allg = torch.cat([g.reshape(-1) for g in gains])
        return {
            "trunk_norm_gain_median": float(allg.median().item()),
            "trunk_norm_gain_min": float(allg.min().item()),
        }

    def aliveness_report(self) -> list[dict[str, float]]:
        """Per-block aliveness diagnostics (PC layer + episode store stats)."""
        return [block.aliveness() for block in self.blocks]

    def total_parameters(self) -> dict[str, int]:
        """Count trainable parameters vs living-weight buffers.

        Trainable: embeddings (text + modality), audio/vision encoders,
        attention, layer norms, output projection.
        Living buffers: PC layer state per block (weight, prediction,
        set_point, momentum, update_ema, precision, error_acc, plasticity,
        episode_*).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        living_buffers = 0
        for block in self.blocks:
            for _, buf in block.living_ffn.named_buffers():
                living_buffers += buf.numel()
        return {
            "trainable": trainable,
            "living_buffers": living_buffers,
            "total": trainable + living_buffers,
        }

    def non_feedforward_signal(self, **encode_kwargs) -> float:
        """Mean abs difference between two consecutive identical-input
        encode() passes at the trunk output level. Positive value confirms
        the model as a whole is not feedforward.

        kwargs are forwarded to encode(); pass whatever modality inputs
        the run is exercising.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                out1 = self.encode(**encode_kwargs)["latents"]
                out2 = self.encode(**encode_kwargs)["latents"]
                return (out2 - out1).abs().mean().item()
        finally:
            if was_training:
                self.train()
