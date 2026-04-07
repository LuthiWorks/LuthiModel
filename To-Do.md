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

## Phase 3C: Multimodal — Audio

- [ ] Install torchaudio
- [ ] Download LibriSpeech clean-100 dataset (~6.3 GB)
- [ ] `luthi/audio_encoder.py` — mel spectrogram → patch embedding → d_model
  - [ ] MelSpectrogram transform (n_mels=80, hop=160, sr=16000)
  - [ ] Patch embedding (Conv2d over spectrogram patches)
  - [ ] Positional encoding for audio tokens
- [ ] `luthi/multimodal_model.py` — MultimodalLuthiLM
  - [ ] Modality embedding (learned, per-modality type)
  - [ ] Shared living weight trunk (reuse existing blocks)
  - [ ] Text output head (existing vocab projection)
  - [ ] Modality-agnostic forward() that handles mixed sequences
- [ ] `luthi/multimodal_data.py` — Audio-text paired dataset
  - [ ] LibriSpeech loader (audio + transcript alignment)
  - [ ] Sequence construction: [audio_tokens, text_tokens]
  - [ ] Collation with padding/truncation
- [ ] Tests for audio encoder
- [ ] Tests for multimodal model (audio + text forward pass)
- [ ] Training run: audio + text, 1024d
- [ ] Verify living weights develop cross-modal patterns

## Phase 3D: Multimodal — Vision

- [ ] Download COCO 2017 dataset (images + captions)
- [ ] `luthi/vision_encoder.py` — image patches → d_model (ViT-style)
  - [ ] Patch embedding (Conv2d, patch_size=16)
  - [ ] 2D positional encoding for image patches
- [ ] Extend MultimodalLuthiLM for vision input
- [ ] `luthi/multimodal_data.py` — Add image-text paired dataset
  - [ ] COCO loader (image + caption)
  - [ ] Sequence construction: [image_patches, caption_tokens]
- [ ] Tests for vision encoder
- [ ] Tests for multimodal model (vision + text forward pass)
- [ ] Training run: vision + text (and optionally audio), 1024d
- [ ] Verify cross-modal episode formation

## Phase 3E: Simulated Embodiment

- [ ] Install MuJoCo (`pip install mujoco`)
- [ ] Design simple body model (articulated limbs, contact sensors, camera)
- [ ] `luthi/touch_encoder.py` — proprioception + contact forces → d_model
  - [ ] Proprioception projection (joint angles + velocities)
  - [ ] Contact force projection
  - [ ] Temporal encoding
- [ ] `luthi/motor_head.py` — d_model → actuator commands
  - [ ] Linear projection with tanh activation (bounded output)
- [ ] `luthi/simulation.py` — MuJoCo environment wrapper
  - [ ] Step function: action → next_state + sensory_input
  - [ ] Camera rendering for visual input
  - [ ] Reward signal (optional — or let living weights learn from consequence alone)
- [ ] Extend MultimodalLuthiLM for touch input + motor output
- [ ] Sensorimotor training loop (closed-loop: sense → process → act → sense)
- [ ] Tests for touch encoder and motor head
- [ ] Tests for simulation environment
- [ ] Training run: embodied agent in simple environment

## Phase 4: CfC Integration (Sanctuary)

- [ ] Review CfC integration interface (`.docs/CFC_LIVING_WEIGHT_INTEGRATION.md`)
- [ ] Design bridge between multimodal LuthiModel and Sanctuary's CfC layer
- [ ] Implement integration module
- [ ] Tests for CfC bridge
- [ ] Joint training experiment

## Phase 5: Scale Testing

- [ ] Test 2048d on current hardware (memory permitting)
- [ ] Profile at larger scales
- [ ] Test 4096d on capable hardware (when available)
- [ ] Document scale-dependent behavior changes

## Infrastructure & Maintenance

- [ ] Add multimodal dependencies to pyproject.toml (torchaudio, torchvision, mujoco)
- [ ] Update .gitignore for new dataset directories
- [ ] Update checkpoint system for multimodal model state
- [ ] Update CLAUDE.md with multimodal design decisions as they're made
- [ ] Keep TRAINING_LOG.md updated with new runs

## Sanctuary

- [ ] Assess Sanctuary codebase for C++ optimization opportunities
- [ ] Profile Sanctuary's cognitive loop to identify bottlenecks
- [ ] Implement optimizations where they make sense
