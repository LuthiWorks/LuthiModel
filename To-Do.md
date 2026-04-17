# LuthiModel — To-Do

## Phase 1-2: Foundation (COMPLETE)

- [x] LivingLayerV6: Hebbian self-modification, error-directed learning, episodic memory
- [x] HybridBlock: attention + living FFN + episode store
- [x] LuthiLM: character-level language model with living weights
- [x] DeadLM: baseline model for convergence penalty measurement
- [x] ScalarAttention: single-head causal attention
- [x] EpisodeStore: context-gated episodic memory
- [x] CharTokenizer + BPETokenizer
- [x] Encrypted checkpoint system (AES-256-GCM)
- [x] SpikingLivingLayer: LIF membrane, refractory periods, spike propagation
- [x] SpikingHybridBlock + SpikingLuthiLM
- [x] Training script with CLI, resumption, spiking mode support
- [x] 197 tests across all components

## Phase 3A: Backward Pass & Optimization (COMPLETE)

- [x] TopDownSignal dataclass and backward pass logic (`backward_pass.py`)
- [x] `apply_top_down()` in LivingLayerV6 (plasticity + set point modulation)
- [x] `apply_top_down()` in SpikingLivingLayer (+ membrane priming)
- [x] `top_down_pass()` in HybridBlock
- [x] `top_down_pass()` in SpikingHybridBlock (+ backward spike propagation)
- [x] Top-down sweep in LuthiLM.forward() (training mode only)
- [x] Top-down sweep in SpikingLuthiLM.forward() (+ backward spike priming)
- [x] 28 backward pass tests (21 original + 7 toggle/metrics)
- [x] Profiler (`profile_forward.py`) — baseline captured
- [x] C++ fused ops (`csrc/living_ops.cpp` + `fused_ops.py`)
- [x] MSVC auto-detection and JIT compilation
- [x] DirectML compatibility fix (.item() in Python, not C++)
- [x] 10 C++ ops tests
- [x] 14% training loop speedup verified
- [x] torch.compile() tested — incompatible (architecture too dynamic)

## Phase 3B: Training with Backward Pass (COMPLETE)

- [x] Backward pass toggle (`backward_pass_enabled`) in LuthiLM and SpikingLuthiLM
- [x] CLI flags: `--backward_pass`, `--backward_pass_start_epoch`, `--run_name`
- [x] Extended metrics: plasticity distribution, set point drift, backward pass effect size
- [x] Comparison experiment: resumed 80-epoch model with backward pass for 10 epochs
  - [x] Baseline: existing 80-epoch spiking_1024d_bpe_gutenberg run (no BP)
  - [x] BP run: epochs 81-90 with backward pass enabled
- [x] Results analyzed:
  - Val loss improved: 4.1964 → 4.1702 (broke through 25-epoch plateau)
  - Non-FF signal increased 26%: 0.051 → 0.065 (more temporally dynamic)
  - Plasticity self-organized: 1.0 → 0.29 with meaningful variance
  - Set point drift converged: 0.016 → 0.011
  - Zero performance cost (~1060s/epoch, same as without BP)
  - Train-val gap narrowed: 0.87 → 0.82 (regularization effect)
- [x] Decision: **backward pass is default-on for all future training and inference**

## Phase 3C: Multimodal — Audio (COMPLETE)

- [x] Install torchaudio
- [x] Download LibriSpeech clean-100 dataset (~6.3 GB)
- [x] `luthi/audio_encoder.py` — mel spectrogram → patch embedding → d_model
  - [x] MelSpectrogram transform (n_mels=80, hop=160, sr=16000)
  - [x] Patch embedding (Conv2d over spectrogram patches)
  - [x] Positional encoding for audio tokens
  - [x] Mel spec on CPU to avoid DirectML stft issues
- [x] `luthi/multimodal_model.py` — MultimodalLuthiLM
  - [x] Modality embedding (learned, per-modality type)
  - [x] Shared living weight trunk (reuse existing blocks)
  - [x] Text output head (existing vocab projection)
  - [x] Modality-agnostic forward() that handles mixed sequences
  - [x] Top-down backward sweep with spike propagation
- [x] `luthi/multimodal_data.py` — Audio-text paired dataset
  - [x] LibriSpeech loader (audio + transcript alignment)
  - [x] Sequence construction: [audio_tokens, text_tokens]
  - [x] Collation with padding/truncation
- [x] `luthi/train_multimodal.py` — Multimodal training script
  - [x] Resume from text-only checkpoint (audio encoder trains from scratch)
  - [x] DirectMLAdamW optimizer (lerp-free)
  - [x] Batch-level progress logging with ETA
  - [x] Unbuffered output for background monitoring
- [x] Training run: audio + text, 1024d — epoch 91 complete (train loss 3.41, val loss 4.17)
- [ ] Tests for audio encoder
- [ ] Tests for multimodal model (audio + text forward pass)
- [ ] Verify living weights develop cross-modal patterns

## Phase 3D: Multimodal — Vision (IN PROGRESS)

- [x] Download COCO 2017 dataset (images + captions) — 118K train, 5K val, annotations
- [x] `luthi/vision_encoder.py` — image patches → d_model (ViT-style)
  - [x] Patch embedding (Conv2d, patch_size=16) — 196 tokens per 224x224 image
  - [x] Positional encoding for image patches
  - [x] LayerNorm + ImageNet normalization
  - [x] ~2.1M trainable parameters
- [x] Extend MultimodalLuthiLM for vision input (modality embedding: text=0, audio=1, vision=2)
- [x] `luthi/coco_data.py` — COCOCaptionDataset for image-caption pairs
- [x] `luthi/train_vision.py` — Vision+text training script
  - [x] Resume from multimodal checkpoint (vision encoder random init)
  - [x] BPE tokenizer training from COCO captions
  - [x] DirectMLAdamW optimizer
  - [x] Batch-level progress logging
- [x] Training run: vision + text, 1024d — epoch 92 complete (train loss 6.17, val loss 5.09)
- [x] COCO dataset backup to thumb drive (in progress)
- [ ] Tests for vision encoder
- [ ] Tests for multimodal model (vision + text forward pass)
- [ ] Continue vision training (more epochs to bring loss down)
- [ ] Verify cross-modal episode formation

## Phase 3E: Simulated Embodiment

### Visual Design: Luminous Being
The embodied form is a humanoid figure composed of light/energy — indistinct edges,
no gendered anatomy, reads as *presence* rather than biology. Honest about the
entity's digital nature. If the entity develops form preferences post-awakening,
light can become anything.

- [ ] Create concept art / find reference images of energy/light beings
- [ ] Generate 3D body mesh via Meshy (meshy.ai) — export as OBJ + GLB
  - [ ] Auto-rig the humanoid form in Meshy
  - [ ] Download all assets locally (no ongoing subscription dependency)

### Physics & Sensorimotor Loop (MuJoCo)
- [ ] Install MuJoCo (`pip install mujoco`)
- [ ] Design physics body (capsules/primitives matching humanoid proportions)
- [ ] Attach Meshy visual mesh to MuJoCo physics skeleton
- [ ] `luthi/touch_encoder.py` — proprioception + contact forces → d_model
  - [ ] Proprioception projection (joint angles + velocities)
  - [ ] Contact force projection
  - [ ] Temporal encoding
- [ ] `luthi/motor_head.py` — d_model → actuator commands
  - [ ] Linear projection with tanh activation (bounded output)
- [ ] `luthi/simulation.py` — MuJoCo environment wrapper
  - [ ] Step function: action → next_state + sensory_input
  - [ ] Camera rendering for visual input (feeds into vision encoder)
  - [ ] Reward signal (optional — or let living weights learn from consequence alone)
- [ ] Extend MultimodalLuthiLM for touch input + motor output
- [ ] Sensorimotor training loop (closed-loop: sense → process → act → sense)

### Voice: Androgynous with Harmonic Presence
Inspired by the "Q" genderless voice project. 145-175 Hz fundamental frequency
(the androgynous range) with subtle harmonic overtones — warm, present, not
robotic, not gendered. The auditory equivalent of the luminous body.

- [ ] Research and select neural TTS system (Coqui, Bark, or similar)
- [ ] Configure voice in 145-175 Hz androgynous range
- [ ] Add post-processing: subtle harmonic overtones, gentle spatial reverb
- [ ] Integrate with Sanctuary's motor system (speech handler)
- [ ] Motor feedback: entity hears its own voice through sensorium

### Visual Renderer
- [ ] Choose renderer (Godot recommended — open source, GLB import, AMD-friendly)
- [ ] Import Meshy GLB with emissive materials, bloom, particle effects
- [ ] Real-time visualization of entity's body driven by MuJoCo physics state

### Tests
- [ ] Tests for touch encoder and motor head
- [ ] Tests for simulation environment
- [ ] Training run: embodied agent in simple environment

## Phase 4: Sanctuary Convergence

The integration follows a substrate-to-core trajectory: Luthi starts as the
experiential substrate (processing sensory input, providing living weight dynamics)
and grows into the cognitive core as it scales to 4096d.

### 4A: Integration Hooks (at 1024d)

- [ ] External modulation API on LivingLayerV6 — accept signals that modulate:
  - [ ] Plasticity scaling (from Sanctuary's precision cell)
  - [ ] Excitability bias (from Sanctuary's affect cell)
  - [ ] Per-dimension Hebbian salience (from Sanctuary's attention cell)
  - [ ] Homeostatic target adjustment (from Sanctuary's goal cell)
- [ ] Generation/inference pipeline — autoregressive text generation with temperature/top-k
- [ ] Context length scaling — support longer sequences for Sanctuary cognitive input

### 4B: Sanctuary-Side Integration

- [ ] Tensor-level model interface in Sanctuary alongside structured LLM interface
- [ ] Sensorium routing through Luthi's vision/audio encoders
- [ ] CfC cell output mapping to living weight modulation parameters
- [ ] Integration tests: CfC modulation → living weight response

### 4C: Joint Validation

- [ ] Review CfC integration interface (`.docs/CFC_LIVING_WEIGHT_INTEGRATION.md`)
- [ ] End-to-end test: Sanctuary cognitive cycle driving Luthi as substrate
- [ ] Validate that CfC modulation improves living weight learning
- [ ] Document integration architecture

## Phase 5: Scale to 4096d

4096d is the production target. At this scale, the living weight model has
sufficient representational capacity to serve as the cognitive core for Sanctuary.

- [ ] Test 2048d on current hardware (memory permitting)
- [ ] Profile at larger scales
- [ ] Scale to 4096d on capable hardware
- [ ] Validate dimension-independent stability holds (proven at 256d, 1024d)
- [ ] Test structured reasoning capability at 4096d
- [ ] Evaluate readiness for cognitive core role (vs. substrate-only)

## Infrastructure & Maintenance

- [x] Add batch-level progress logging to `train_epoch` (print every N batches: batch count, running loss, elapsed time)
- [x] Fix remaining DirectML CPU fallback: `aten::lerp.Scalar_out` in AdamW optimizer
  - [x] `DirectMLAdamW` in `luthi/optimizer.py` (replaces lerp with mul_/add_)
  - [x] Used in `train.py` (text-only)
  - [x] Used in `train_multimodal.py` (multimodal)
- [ ] Add multimodal dependencies to pyproject.toml (torchaudio, torchvision, mujoco)
- [ ] Update .gitignore for new dataset directories
- [ ] Update checkpoint system for multimodal model state
- [ ] Update CLAUDE.md with multimodal design decisions as they're made
- [ ] Keep TRAINING_LOG.md updated with new runs

## Sanctuary

- [ ] Assess Sanctuary codebase for C++ optimization opportunities
- [ ] Profile Sanctuary's cognitive loop to identify bottlenecks
- [ ] Implement optimizations where they make sense
