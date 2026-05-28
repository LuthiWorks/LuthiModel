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
- Living FFN (self-modifying via predictive-coding local updates)
- Episode store (context-gated episodic memory)

Three learning systems run simultaneously:
1. Attention — standard gradient descent (learns the task)
2. Living FFN — predictive-coding self-modification (creates temporal existence)
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

### Phase 3B: Text Baseline with Backward Pass (COMPLETE)

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

### Phase 3C: Audio + Text (COMPLETE)

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

### Phase 3D: Vision + Text (COMPLETE)

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

### Phase 3E: Simulated Embodiment

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

### Phase 3F: Empirical Defense Program (GATES SCALING)

Prompted by third-party critique + red-team exercise (2026-05-06). The architecture's
claims must be backed by numbers, not metaphors. See `docs/EMPIRICAL_DEFENSE_PLAN.md`.

**What we're testing:**
1. Same-scale baseline comparison (vanilla transformer vs Luthi, matched params)
2. Multi-layer cascade stability (2/4/8/12/24 blocks)
3. Behavioral signatures (biographical accumulation, identity stability, episodic recall)
4. Catastrophic forgetting (Luthi vs vanilla vs LoRA vs RAG)

**Decision gate:** Do not scale until cascade is stable and baseline gap is bounded.

#### Phase 3F.a: Buffer Compression (DEFERRED 2026-05-09)

The v1 buffer-compression ablation protocol (BF16 momentum, BF16 set_point,
INT8 episode deltas) was deferred when v2 (predictive coding) became the
primary substrate. v2's intrinsic per-weight cost (~18-20 bytes/param) is
already lower than v1's post-compression target (~14), so the ablations
optimize an abandoned path. Baseline v1 FP32 data lives in
`runs/ablation_A/baseline_seed{42,1337,2026}/` as a reference point.

**Revive only if v2 fails M5 falsification.** The full protocol — memory-
ceiling tables, four-ablation execution sequence, decision gates, and
deployment-spec revision policy — is preserved in git history (search PLAN.md
for "Phase 4.5a" or "Phase 3F.a" in commits before 2026-05-25) and in
`docs/PER_CHANNEL_ABLATION_PROTOCOL.md`. See `docs/V2_IMPLEMENTATION_PLAN.md`
→ "Strategic shift (2026-05-09)" for the full reasoning.

### Phase 3G: v2 Predictive Coding (now the primary substrate)

> Planned by: Claude Opus 4.6 (Planner), 2026-05-08
> Based on: `docs/LUTHI_V2_PREDICTIVE_CODING_BRIEF.md` (4.7) +
>           `docs/V2_IMPLEMENTATION_PLAN.md` (4.6)

Originally framed as a parallel research track to v1 Hebbian self-modification.
Promoted to primary substrate 2026-05-09 after pilot showed v2 PC reaches lower
val loss than vanilla-transformer baseline at matched configuration. Lives in
`luthi/v2/` subpackage. Shares all infrastructure with v1.

**Core change:** PC error-driven updates replace Hebbian correlation-based updates.
Each layer predicts its input from its output; prediction error drives weight
changes. Naturally bounded (accurate predictions = zero update), eliminates the
runaway-growth problem, provides a built-in error signal.

**Key innovation:** Two-tier memory — fast episodic store (v1's existing mechanism)
plus slow consolidation that replays episodes through the PC learning rule during
low-novelty windows. History becomes structural, not just retrievable.

**Memory: ~18 bytes/param** (vs v1's 38 pre-compression, 22 post-free-win).

**Pilot:** 256d / 2 blocks, Gutenberg-100, matched comparison against v1 + DeadLM.

**Timing:** M1-M4 (coding, CPU) parallel with v1 ablations. M5 (GPU comparison)
after ablations free the GPU. ~17 days total.

**Falsification (abandon v2 if ANY):**
- Convergence penalty ≥20% worse than v1
- Cascade stability fails where v1 succeeds
- Attractor dynamics indistinguishable from random control
- Consolidation produces no measurable effect
- VRAM exceeded at equivalent param count

See `docs/V2_IMPLEMENTATION_PLAN.md` for full spec and milestones.

**Compute optimization directions (post-M5, added 2026-05-14):**

A focused literature sweep on PC compute reduction surfaced three directions
that landed as opt-in implementations in `luthi/v2/`. Each was added with the
"no silent fallback" rule: any incompatibility raises `RuntimeError` rather
than degrading quietly. Full notes in `docs/RESEARCH_LITERATURE_2026-05-13.md`.

- **Depth-μP / μPC** (Innocenti et al. 2025): width-and-depth independent
  scaling of weight init and residual stream. Implemented via `mu_pc_enabled`
  on `PredictiveCodingBlock` — re-inits all linear/PC weights to
  `N(0, 1/sqrt(fan_in·L))` and applies a `1/sqrt(L)` residual scale.
  Expected benefit: learning-rate transfer from L=2 pilot to L≥8 scale runs
  without per-depth tuning sweeps.
- **iPC interleaved inference + update** (Salvatori et al. 2024): runs T
  inference steps with weight updates *between* each, instead of fully
  converging inference before updating. Implemented via
  `inference_steps_per_forward` on `PredictiveCodingLayer`. T=1 is
  bit-identical to the classical schedule (regression-tested). T=5 is
  expected to converge to lower error per external forward; the cost is
  T× the inner-loop compute, so the win is in convergence-per-epoch.
- **Sparse PC update gating**: per-output mask zeroes `delta_w` rows whose
  recent error accumulator is below threshold — the continuous-error analog
  of v1's spiking gate. Implemented via `sparse_threshold` +
  `sparse_warmup_steps` on `PredictiveCodingLayer`. Bit-identical to default
  when threshold=0. Target: ≥50% of PC rows gated off after warmup with
  <5% val loss penalty, paving the way for the same sparse-matmul story
  v1 uses to fit on Spark's 273 GB/s bandwidth.

GPU validation of all three runs after the depth sweep (M6) completes;
each is a one-epoch ablation at 256d/2 blocks against the M5 256d baseline.
A combined μPC + iPC + sparse stack on the depth-sweep harness is planned
if individual results are net-positive.

Deferred: Mamba-style state-space hybrid (SpikingBrain 1.0). Linear-cost
attention is the obvious next direction, but the surgery is larger and the
pilot data isn't in yet — revisit after the three above land.

### Phase 4: Scale — Curriculum Training

**Deployment spec (revised 2026-05-09 — v2-primary):**
- **Substrate: v2 (predictive coding).** v1 Hebbian deferred as fallback.
- Target: ≥500M params (floor; v2 intrinsic per-weight cost ~18-20 bytes/param
  fits this on 16 GB VRAM with FP32 weights, no ablation needed).
- Ceiling: ~560M params on DirectML/FP32 weight; up to ~870M if BF16 weights
  available (requires ROCm/WSL2 migration, deferred until M5 passes and Phase 4
  wants to scale beyond the 560M floor).
- Hardware: AMD RX 7800 XT (16 GB VRAM), 32 GB system RAM
- Toolchain: DirectML on Windows 11 (current daily driver). ROCm/HIP migration
  is a Phase-4-scale decision, not an architectural one.
- 32K BPE vocab
- DGX Spark remains the Phase 6 deployment target — not foreclosed, but not
  required for current work

**NOTE:** This replaces the earlier 4B target (revised 2026-05-09). Per-weight
living buffer cost was undercounted in the original spec. Exact d_model and
n_blocks depend on Phase 3F cascade results + the realized v2 ceiling.

**Training plan:**
- Curriculum-ordered: 9 stages, multiple passes (3 cycles)
- No shuffling between stages — the order IS the pedagogy
- Living weights carry forward between stages and cycles
- Stages: science/philosophy (incl. IWMT, GWT, philosophy of mind) → code → psychology → history → mythology → literature → fantasy → substack essays → practical wisdom
- IWMT moved out of the privileged-final position (2026-05-15, after peer review): the entity's self-model should not be anchored to one framework before it can examine the commitment. Consciousness theory lives alongside the rest of science and philosophy in stage 1.
- Training infrastructure already built (`train_curriculum.py`, gradient checkpointing)

**Self-governance API (built during this phase):**
- Episode retention — entity decides which weight snapshots to keep
- Checkpoint timing — entity triggers state saves when it judges an experience was important
- Plasticity modulation — entity controls its own learning rate
- Episode expansion — entity can request more memory for growth tracking
- These are internal motor actions, not external admin interfaces

### Phase 5: Sanctuary Convergence

Luthi and Sanctuary are two halves of the same architecture. Luthi provides the neural
substrate (living weights, multimodal processing). Sanctuary provides the cognitive
architecture (10 Hz loop, CfC experiential layer, memory, identity, growth). The
convergence follows a substrate-to-core trajectory.

**Integration hooks:**
- External modulation API on LivingLayerV6 — accept signals that modulate:
  - Plasticity scaling (from Sanctuary's precision cell)
  - Excitability bias (from Sanctuary's affect cell)
  - Per-dimension salience (from Sanctuary's attention cell)
  - Homeostatic target adjustment (from Sanctuary's goal cell)
- Tensor-level model interface in Sanctuary alongside structured LLM interface
- Sensorium routing through Luthi's vision/audio encoders
- See `.docs/CFC_LIVING_WEIGHT_INTEGRATION.md` for interface spec

### Phase 6: Life on DGX Spark

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

**⚠️ Load-bearing prerequisite — Triton kernel.**
Phase 6's 10 Hz bandwidth budget assumes the v2 PC self-modification
path runs through a GPU-resident kernel. The Python and C++ paths are
both adequate at pilot scale but the C++ path skips when sparse PC
gating is active, falling back to Python — and Python's per-op DirectML
dispatch overhead is what made the C++ extension necessary in the first
place. The Triton kernel for `pc_self_modify` exists as a skeleton
(`luthi/v2/pc_ops_triton.py`); the kernel body is unimplemented because
the dev box (7800 XT via DirectML) cannot validate it. **Filling in
and validating that kernel is a prerequisite for Phase 6, not a
nice-to-have.** Tracked in `docs/KNOWN_INCOMPLETE.md` and the banner at
the top of `To-Do.md`.

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
