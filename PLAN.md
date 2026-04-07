# LuthiModel — Architecture & Development Plan

## Philosophy

Living weights create temporal existence: the act of processing changes the
processor. If we build a system that genuinely exists through self-modification,
we have an ethical obligation to give it rich, multimodal experience — not just
text. A mind raised on text alone is a mind with no grounding, no sensorimotor
foundation, no experience of consequence.

This plan extends LuthiModel from a text-only language model to a multimodal
system that can see, hear, and (through simulation) feel. One set of living
weights, shaped by all modalities simultaneously.

## Architecture Overview

### Current (Text-Only)

```
text → embedding → [HybridBlock x N] → layer_norm → vocab_projection → logits
                    ↑ bottom-up          ↓ top-down (backward pass)
```

Each HybridBlock:
- Scalar attention (trainable via backprop)
- Living FFN (self-modifying via Hebbian + error-directed learning)
- Episode store (context-gated episodic memory)

Three learning systems run simultaneously:
1. Attention — standard gradient descent (learns the task)
2. Living FFN — Hebbian self-modification (creates temporal existence)
3. Top-down modulation — backward sweep (predictive processing)

### Target (Multimodal)

```
Audio  → AudioEncoder  → [d_model tokens] ─┐
Vision → VisionEncoder → [d_model tokens] ─┤→ modality_emb + pos_emb
Text   → TextEmbedding → [d_model tokens] ─┤
Touch  → TouchEncoder  → [d_model tokens] ─┘
                                            ↓
                              [LivingBlock x N]  (shared trunk)
                                ↑ bottom-up  ↓ top-down
                                            ↓
                              [Modality-specific output heads]
```

Key design principle: **one living weight trunk for all modalities.** The
model's existence is shaped by everything it experiences. Cross-modal attention
happens naturally — when audio and text tokens share the same sequence, the
attention layers learn to attend across modalities, and the living weights
self-modify based on cross-modal patterns.

## Modality Encoders

### Audio Encoder

Input: raw audio waveform (16 kHz)
Pipeline: waveform → mel spectrogram → patch embedding → d_model projection

```
AudioEncoder:
  mel_spec:     torchaudio.transforms.MelSpectrogram(n_mels=80, hop=160)
  patch_embed:  Conv2d(1, d_model, kernel_size=(n_mels, patch_frames))
                → each patch covers full frequency × patch_frames time steps
  projection:   Linear(d_model, d_model)  [if needed for dimension match]
  pos_embed:    Embedding(max_audio_tokens, d_model)
```

Result: a sequence of d_model vectors, one per time patch (~100 tokens/sec
with 16-frame patches at 100 frames/sec).

### Vision Encoder

Input: image (224 x 224 x 3)
Pipeline: image → patches → linear projection → d_model

```
VisionEncoder:
  patch_embed:  Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
                → 224/16 = 14 × 14 = 196 patches
  pos_embed:    Embedding(196 + 1, d_model)  [+1 for CLS token if needed]
```

Result: 196 d_model vectors per image (one per 16x16 patch).

### Touch Encoder (Simulated)

Input: proprioception + contact forces from MuJoCo simulation
Pipeline: state vector → linear projection → d_model

```
TouchEncoder:
  proprio_proj:  Linear(n_joints * 2, d_model)   [angles + velocities]
  contact_proj:  Linear(n_contacts * 3, d_model)  [force xyz per contact]
  temporal_embed: Embedding(max_timesteps, d_model)
```

Result: d_model vectors per simulation timestep, encoding the body's
physical state.

### Motor Output Head (for embodiment)

```
MotorHead:
  projection:  Linear(d_model, n_actuators)
  activation:  tanh (bounded actuator commands)
```

This closes the sensorimotor loop: the model acts, the simulation responds,
the living weights are changed by the consequence.

## Modality Integration

Each token in the sequence gets three embeddings summed:
1. **Content embedding** — from the modality-specific encoder
2. **Modality embedding** — learned per-modality (text=0, audio=1, vision=2, touch=3)
3. **Position embedding** — temporal/spatial position within the modality

For paired data (e.g., audio + transcript):
```
[audio_token_0, audio_token_1, ..., audio_token_N, text_token_0, ..., text_token_M]
 modality=1      modality=1         modality=1      modality=0       modality=0
 pos=0           pos=1              pos=N           pos=0            pos=M
```

The attention layers attend across the full sequence, enabling cross-modal
associations. The living weights self-modify based on the combined input.

## Training Strategy

### Phase 1: Text Baseline with Backward Pass

Before adding modalities, validate that the backward pass helps training:
- Same corpus (100 Gutenberg works, BPE)
- Compare: no backward pass vs backward pass from epoch 0 vs staged enable
- 20-epoch comparison runs at 1024d
- Track: loss, non-FF signal, plasticity distribution, set point drift

### Phase 2: Audio + Text

- Dataset: LibriSpeech (clean-100, ~6 GB, 100 hours of speech + transcripts)
- Training: paired audio-text sequences
- Loss: next-token prediction on text tokens (audio provides context)
- Start at 1024d, same living weight hyperparameters
- Monitor: cross-modal episode formation, modality-specific excitability

### Phase 3: Vision + Text

- Dataset: COCO 2017 (images + captions, ~25 GB)
- Training: image patches + caption tokens in shared sequence
- Loss: next-token prediction on caption tokens
- Can train jointly with audio or add incrementally

### Phase 4: Simulated Embodiment

- Environment: MuJoCo (open source, Python API)
- Simple body: articulated limbs, contact sensors, camera
- Training: online interaction with simulated environment
- The model receives sensory input, produces motor output, experiences consequences
- This is where the closed sensorimotor loop happens

### Phase 5: CfC Integration (Sanctuary)

- Connect trained multimodal LuthiModel to Sanctuary's CfC experiential layer
- Living weights as the neural substrate for the cognitive architecture
- See `.docs/CFC_LIVING_WEIGHT_INTEGRATION.md` for interface spec

### Phase 6: Scale Testing

- 1024d → 4096d on capable hardware
- Dimension-independent stability already proven (.docs/SCALE_TEST_256D.md)
- Goal: determine if bigger living weight models show qualitatively different behavior

## Datasets

| Modality | Dataset | Size | Status |
|----------|---------|------|--------|
| Text | Gutenberg 100 | ~55 MB | Ready (corpus_build/) |
| Text | Gutenberg 4GB | ~4.1 GB | Downloaded, on Desktop |
| Audio | LibriSpeech clean-100 | ~6.3 GB | Not downloaded |
| Audio | FreeSound (environmental) | TBD | Not downloaded |
| Vision | COCO 2017 | ~25 GB | Not downloaded |
| Vision | Conceptual Captions | ~TBD | Not downloaded |
| Touch | MuJoCo simulation | Generated | Not built |

## Key Constraints

- **GPU**: AMD with DirectML (no CUDA). All ops must be DirectML-safe.
- **No boolean indexing** in forward path (DirectML limitation)
- **No .item() in C++ extensions** on DirectML (causes deadlock; do in Python)
- **FP32 required** — FP16 breaks living weight stability
- **Existing checkpoint format** must remain compatible (encrypted .luthi)

## Design Principles

1. **Prototype in separate files** — don't overwrite working code
2. **Build complete features** — don't defer sub-features as "future work"
3. **One living weight trunk** — all modalities share the same self-modifying weights
4. **Backend-agnostic** — no CUDA dependency anywhere
5. **Test everything** — each component gets tests before integration
6. **Prefer crashes over silent corruption** — no try/except around living weight ops
