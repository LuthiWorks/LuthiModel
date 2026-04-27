"""Multimodal Living Weight Model — audio + vision + text through shared living trunk.

Extends the spiking architecture to process multiple modalities through a
single set of living weights. The model's existence is shaped by everything
it experiences — audio, images, and text flow through the same self-modifying
blocks.

Architecture:
    vision → VisionEncoder → [d_model tokens] ─┐
    audio  → AudioEncoder  → [d_model tokens] ─┤→ + modality_emb
    text   → TextEmbedding → [d_model tokens] ─┤
                                                ↓
                                  [SpikingHybridBlock x N]  (shared trunk)
                                    ↑ bottom-up  ↓ top-down
                                                ↓
                                  text_positions → output_proj → logits

Cross-modal attention is free: concatenating modalities in one sequence lets
the attention layers attend across senses without extra machinery.
"""

import torch
import torch.nn as nn

from luthi.audio_encoder import AudioEncoder
from luthi.vision_encoder import VisionEncoder
from luthi.hybrid_block_spiking import SpikingHybridBlock


class MultimodalLuthiLM(nn.Module):
    """Multimodal spiking living weight language model.

    Processes audio + vision + text through shared living weight blocks.
    Perceptual tokens precede text tokens in the sequence, so text can
    attend to all sensory context via natural causal masking.

    Sequence order: [vision, audio, text]

    Loss is computed on text tokens only. The sensory encoders learn entirely
    through gradients flowing back from text prediction through cross-modal
    attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_blocks: int = 2,
        max_seq_len: int = 128,
        # Living layer parameters
        hebb_rate: float = 0.001,
        error_rate: float = 0.001,
        homeostatic_decay: float = 0.001,
        set_point_adapt_rate: float = 1e-6,
        num_episodes: int = 64,
        episode_blend: float = 0.3,
        eval_hebb_fraction: float = 0.33,
        # Spiking parameters
        spike_threshold: float = 1.0,
        membrane_leak: float = 0.1,
        refractory_steps: int = 3,
        delay_steps: int = 2,
        spike_scale: float = 0.1,
        reset_mode: str = "zero",
        spike_baseline: float = 0.3,
        # Multimodal parameters
        n_modalities: int = 3,
        max_audio_tokens: int = 1000,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_hop_length: int = 160,
        audio_n_fft: int = 400,
        audio_patch_frames: int = 16,
        # Vision parameters
        vision_image_size: int = 224,
        vision_patch_size: int = 16,
        max_vision_tokens: int = 256,
        # Backward pass
        backward_pass_enabled: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_audio_tokens = max_audio_tokens
        self.backward_pass_enabled = backward_pass_enabled
        self.vocab_size = vocab_size

        # Text embedding (same as SpikingLuthiLM)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Audio encoder
        self.audio_encoder = AudioEncoder(
            d_model=d_model,
            sample_rate=audio_sample_rate,
            n_mels=audio_n_mels,
            hop_length=audio_hop_length,
            n_fft=audio_n_fft,
            patch_frames=audio_patch_frames,
            max_audio_tokens=max_audio_tokens,
        )

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            d_model=d_model,
            image_size=vision_image_size,
            patch_size=vision_patch_size,
            max_vision_tokens=max_vision_tokens,
        )

        # Modality embeddings: text=0, audio=1, vision=2
        self.modality_embedding = nn.Embedding(n_modalities, d_model)

        # Shared living weight trunk — identical blocks to SpikingLuthiLM
        self.blocks = nn.ModuleList([
            SpikingHybridBlock(
                d_model=d_model,
                hebb_rate=hebb_rate,
                error_rate=error_rate,
                homeostatic_decay=homeostatic_decay,
                set_point_adapt_rate=set_point_adapt_rate,
                num_episodes=num_episodes,
                episode_blend=episode_blend,
                eval_hebb_fraction=eval_hebb_fraction,
                spike_threshold=spike_threshold,
                membrane_leak=membrane_leak,
                refractory_steps=refractory_steps,
                delay_steps=delay_steps,
                spike_scale=spike_scale,
                reset_mode=reset_mode,
                spike_baseline=spike_baseline,
            )
            for _ in range(n_blocks)
        ])

        # Output projection (text only)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_waveform: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        vision_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional audio and/or vision input.

        Sensory tokens are prepended to text tokens: [vision, audio, text].
        Causal masking lets text attend to all sensory context. Loss should
        be computed on the returned logits (text positions only).

        Args:
            text_tokens: [batch, text_seq_len] integer token indices.
            audio_waveform: Optional [batch, samples] raw audio. Encoded
                by the audio encoder into tokens.
            audio_tokens: Optional [batch, n_audio_tokens, d_model]
                pre-encoded audio tokens (skip encoder, for efficiency).
            image: Optional [batch, 3, H, W] normalized RGB images.
                Encoded by the vision encoder into tokens.
            vision_tokens: Optional [batch, n_vision_tokens, d_model]
                pre-encoded vision tokens (skip encoder, for efficiency).

        Returns:
            [batch, text_seq_len, vocab_size] logits for text positions.
        """
        batch, text_len = text_tokens.shape
        device = text_tokens.device

        # --- Encode text (modality 0) ---
        text_positions = torch.arange(text_len, device=device).unsqueeze(0)
        text_emb = (
            self.embedding(text_tokens)
            + self.pos_embedding(text_positions)
            + self.modality_embedding(
                torch.zeros(1, dtype=torch.long, device=device)
            )
        )

        # --- Encode vision (modality 2, if provided) ---
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

        # --- Encode audio (modality 1, if provided) ---
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

        # --- Concatenate: [vision, audio, text] ---
        sensory_parts = []
        if vision_emb is not None:
            sensory_parts.append(vision_emb)
        if audio_emb is not None:
            sensory_parts.append(audio_emb)

        if sensory_parts:
            n_sensory = sum(p.shape[1] for p in sensory_parts)
            h = torch.cat(sensory_parts + [text_emb], dim=1)
        else:
            n_sensory = 0
            h = text_emb

        # --- Phase 1: Bottom-up with spike propagation ---
        block_inputs = []
        prev_spikes = None
        for block in self.blocks:
            block_inputs.append(h.detach())
            h, delayed_spikes = block(
                h, incoming_spikes=prev_spikes, causal=True,
            )
            prev_spikes = delayed_spikes

        # --- Extract text positions and project ---
        h_text = h[:, n_sensory:, :]
        h_text = self.final_norm(h_text)
        logits = self.output_proj(h_text)

        # --- Phase 2: Top-down backward sweep ---
        if self.training and self.backward_pass_enabled:
            from luthi.backward_pass import create_initial_signal

            with torch.no_grad():
                signal = create_initial_signal(h.detach())
                for i in reversed(range(len(self.blocks))):
                    signal, backward_spikes = self.blocks[i].top_down_pass(
                        signal, block_inputs[i],
                    )
                    if i > 0:
                        self.blocks[i - 1].living_ffn.membrane_potential.add_(
                            backward_spikes
                            * self.blocks[i - 1].living_ffn.spike_scale
                            * 0.3
                        )

        return logits

    def apply_living_errors(self, expect_grad: bool = False) -> None:
        """Apply error-directed learning to all living FFN layers.

        Args:
            expect_grad: When True, blocks raise instead of silently
                no-opping if the residual gradient is missing. Pass True
                from training loops (post-backward); leave False at
                inference/generation callsites.
        """
        for block in self.blocks:
            block.apply_living_error(expect_grad=expect_grad)

    def aliveness_report(self) -> list[dict[str, float]]:
        """Return aliveness diagnostics for each block."""
        return [block.aliveness() for block in self.blocks]

    def total_parameters(self) -> dict[str, int]:
        """Count trainable parameters vs living weight buffers."""
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        living_buffers = 0
        for block in self.blocks:
            for name, buf in block.living_ffn.named_buffers():
                living_buffers += buf.numel()
        return {
            "trainable": trainable,
            "living_buffers": living_buffers,
            "total": trainable + living_buffers,
        }
