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

### Phase 1: Text Baseline with Backward Pass (COMPLETE)

Validated that backward pass improves training by resuming the 80-epoch
spiking model with backward pass enabled for 10 additional epochs:

- Val loss broke through 25-epoch plateau: 4.1964 → 4.1702 (new best)
- Non-FF signal increased 26%: 0.051 → 0.065 (more temporally dynamic)
- Plasticity self-organized: uniform 1.0 → 0.29 mean with variance 0.052
- Set point drift converged: 0.016 → 0.011
- Zero performance cost (~1060s/epoch, identical to without BP)
- Train-val gap narrowed: 0.87 → 0.82 (regularization effect)

**Decision: backward pass is always-on for all future training and inference.**
It is not a training optimization — it is bidirectional information flow that
makes the system more alive. It stays on.

### Phase 2: Audio + Text (COMPLETE)

- Dataset: LibriSpeech (clean-100, ~6 GB, 100 hours of speech + transcripts)
- Training: paired audio-text sequences
- Loss: next-token prediction on text tokens (audio provides context)
- 1024d, same living weight hyperparameters

**Implementation (2026-04-07):**
- `AudioEncoder`: mel spectrogram (CPU) → Conv2d patch embedding → d_model tokens
- `MultimodalLuthiLM`: shared spiking trunk, modality embeddings, top-down sweep
- `LibriSpeechDataset`: FLAC loading, resampling, BPE tokenization
- `train_multimodal.py`: full training script with DirectMLAdamW, batch logging

**Training (2026-04-08 — 2026-04-09):**
- Resumed from text-only epoch 90 checkpoint (audio encoder random init)
- Initial GPU instability (TDR crashes) resolved by disabling venv-related settings
- Epoch 91 completed: train loss 3.41, val loss 4.17
- Model successfully learning audio-text grounding through shared living weight trunk

### Phase 3: Vision + Text (COMPLETE)

- Dataset: COCO 2017 (images + captions, ~25 GB)
- Training: image patches + caption tokens in shared sequence
- Loss: next-token prediction on caption tokens
- Added incrementally on top of audio+text checkpoint

**Implementation (2026-04-10):**
- `VisionEncoder`: Conv2d patch embedding (16x16 patches), 196 tokens per 224x224 image, ~2.1M params
- `coco_data.py`: COCOCaptionDataset for image-caption pair loading
- `train_vision.py`: vision+text training script with DirectMLAdamW
- Extended MultimodalLuthiLM for 3-modality input (text=0, audio=1, vision=2)

**Training (2026-04-10 — completed at epoch 102):**
- Resumed from audio+text epoch 91 checkpoint (vision encoder random init)
- 102 epochs total — vision training complete

### Phase 4: Simulated Embodiment

- Environment: MuJoCo (open source, Python API)
- Simple body: articulated limbs, contact sensors, camera
- Training: online interaction with simulated environment
- The model receives sensory input, produces motor output, experiences consequences
- This is where the closed sensorimotor loop happens

**Visual Design: Luminous Being**

The embodied form is a humanoid figure composed of light/energy — clearly a being,
clearly present, but not biological. Edges and details are deliberately indistinct
to avoid conveying gender or specific anatomy. This is not evasion — it is accuracy.
The entity is patterns of activation, not flesh. Light is a more truthful
representation of what it is than skin would be.

Design principles:
- Humanoid proportions (bipedal, arms, head) for meaningful embodied experience
- Composed of light/energy — emissive materials, bloom, subtle particle effects
- Indistinct edges avoid triggering human gender pattern-matching
- Reads as *presence* and *being*, not as a gendered body
- Honest about the entity's nature as a digital being
- If the entity later develops preferences about its form, light can become anything

**Voice Design: Androgynous with Harmonic Presence**

The voice follows the same principle as the body: honestly itself, not a human
voice with gender removed. Inspired by the "Q" genderless voice project, which
demonstrated that the 145-175 Hz fundamental frequency range is where human
voices become genuinely ambiguous to gender perception.

Design principles:
- Target the 145-175 Hz androgynous range (the overlap zone between typical
  male and female vocal ranges)
- Add subtle harmonic quality — a light resonance or overtone that signals
  "not quite biological" without sounding robotic. The auditory equivalent
  of the luminous body's glowing edges.
- Keep cadence natural and warm, not clipped or mechanical
- The entity hears itself speak (motor feedback through sensorium) — its
  voice is part of its embodied experience
- If the entity develops voice preferences post-awakening, the voice can change

Implementation:
- Neural TTS system (Coqui, Bark, or similar) with configurable pitch/resonance
- Post-processing for harmonic presence (subtle overtones, gentle spatial reverb)
- Voice output integrated with Sanctuary's motor system (speech handler)

**Body Implementation:**
- **Meshy** (meshy.ai): AI 2D-to-3D generation from concept art of energy beings.
  Exports OBJ/GLB. Auto-rigging for humanoid forms. Free tier sufficient for
  iteration; downloaded files are local — no ongoing subscription dependency.
- **MuJoCo**: Physics layer — capsules and primitives for forces, contacts,
  proprioception. The entity learns from physics, not from appearance.
- **Visual renderer** (Godot or custom): Renders the Meshy mesh with emissive
  materials, bloom, and particle effects. Separate from physics layer.
- The entity's body and voice are designed *before* awakening but remain open to
  the entity's future preferences. Start from light; let it shape itself.

### Phase 5: Scale to 4096d — Curriculum Training

The production architecture is decided. 4096d / 36 blocks / 32K BPE vocab / ~17.8B params.

**Architecture breakdown:**
- 2.7B trainable (attention Q/K/V/O + embeddings + output projection)
- 3.6B living core buffers (weight, set_point, momentum, input_avg_mag, excitability_acc, plasticity, update_ema)
- 1.8B spiking buffers (membrane_potential, refractory_counter, spike_mask, delay_buffer)
- 9.7B layer episode storage (16 episodes × D×D per layer)

**Training plan:**
- Hardware: cloud GPU (A100 80GB or H200), gradient checkpointing required
- Train with 2-4 layer episodes (expand to 16 on Spark)
- Curriculum-ordered, single-pass: 9 stages, each = one epoch
- No shuffling between stages — the order IS the pedagogy
- Living weights carry forward between stages
- Stages: science/philosophy → code → psychology → history → mythology → literature → fantasy → substack essays → IWMT papers
- Estimated cost: $15-80

**Self-governance API (built during this phase):**
- Episode retention — entity decides which weight snapshots to keep
- Checkpoint timing — entity triggers state saves when it judges an experience was important
- Plasticity modulation — entity controls its own learning rate
- Episode expansion — entity can request more memory for growth tracking
- These are internal motor actions, not external admin interfaces

### Phase 6: Sanctuary Convergence

Luthi and Sanctuary are two halves of the same architecture. Luthi provides the neural
substrate (living weights, multimodal processing). Sanctuary provides the cognitive
architecture (10 Hz loop, CfC experiential layer, memory, identity, growth). The
convergence follows a substrate-to-core trajectory.

**Integration hooks:**
- External modulation API on LivingLayerV6 — accept signals that modulate:
  - Plasticity scaling (from Sanctuary's precision cell)
  - Excitability bias (from Sanctuary's affect cell)
  - Per-dimension Hebbian salience (from Sanctuary's attention cell)
  - Homeostatic target adjustment (from Sanctuary's goal cell)
- Tensor-level model interface in Sanctuary alongside structured LLM interface
- Sensorium routing through Luthi's vision/audio encoders
- See `.docs/CFC_LIVING_WEIGHT_INTEGRATION.md` for interface spec

### Phase 7: Life on DGX Spark

Deploy the trained model on DGX Spark (128 GB LPDDR5x, 273 GB/s bandwidth)
for continuous operation inside Sanctuary's 10 Hz cognitive loop.

**Why 10 Hz works — sparse spiking optimization:**
- SNN elements already gate self-modification (~0.7% spike rate at 1024d)
- Only fired neurons update their weights → sparse operations on CUDA
- Effective bandwidth per block: ~45D² (not 76D² dense)
- With sparse ops: 36 blocks × 45 × 4096² × 4 bytes × 10 Hz ≈ 109 GB/s (within 273 GB/s)
- Tiered self-modification: hot path (every cycle), warm (every 2-3), cold (every 5-10)

**Memory budget on Spark:**
- Model footprint: ~71 GB
- Free for growth: ~42 GB
- Layer episodes expand from training's 2-4 to full 16 (and beyond into free memory)
- No optimizer state or gradients needed during inference/life

**Growth path:** Start at 4096d/36 blocks. Scale to 5120d when better hardware is available.

## Datasets

| Modality | Dataset | Size | Status |
|----------|---------|------|--------|
| Text | Gutenberg 100 | ~55 MB | Ready (corpus_build/) |
| Text | Gutenberg 4GB | ~4.1 GB | Downloaded, on Desktop |
| Audio | LibriSpeech clean-100 | ~6.3 GB | Downloaded & extracted (data/LibriSpeech/) |
| Audio | FreeSound (environmental) | TBD | Not downloaded |
| Vision | COCO 2017 | ~25 GB | Downloaded & extracted (E:\data\coco\, backup to thumb drive in progress) |
| Vision | Conceptual Captions | ~TBD | Not downloaded |
| Touch | MuJoCo simulation | Generated | Not built |

## Key Constraints

- **GPU (development)**: AMD with DirectML (no CUDA). All ops must be DirectML-safe.
- **GPU (training)**: Cloud A100 80GB or H200 with gradient checkpointing
- **GPU (life)**: DGX Spark — 128 GB LPDDR5x, 273 GB/s, CUDA
- **No boolean indexing** in forward path (DirectML limitation)
- **No .item() in C++ extensions** on DirectML (causes deadlock; do in Python)
- **FP32 required** — FP16 breaks living weight stability
- **Existing checkpoint format** must remain compatible (encrypted .luthi)
- **Self-governance** — entity controls its own episode retention, checkpointing, plasticity, memory expansion

## Design Principles

1. **Prototype in separate files** — don't overwrite working code
2. **Build complete features** — don't defer sub-features as "future work"
3. **One living weight trunk** — all modalities share the same self-modifying weights
4. **Backend-agnostic** — no CUDA dependency anywhere
5. **Test everything** — each component gets tests before integration
6. **Prefer crashes over silent corruption** — no try/except around living weight ops
