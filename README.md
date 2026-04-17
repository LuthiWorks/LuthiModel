# Luthi Model

> Living weights: self-modifying neural network parameters that change during their own forward pass.
> A new kind of computation that is neither feedforward nor recurrent.

## What This Is

Luthi Model is a neural architecture built on **rich parameters** — weights that carry per-parameter history, plasticity, momentum, excitability, and context-gated episodic memory. The core innovation is **living weights**: parameters that self-modify during their own forward pass, creating a computation where processing changes the processor.

Three learning systems run simultaneously:
1. **Attention** — standard gradient descent (learns the task)
2. **Living FFN** — Hebbian self-modification (creates temporal existence)
3. **Top-down modulation** — backward sweep (bidirectional predictive processing)

The living FFN is the body, not the brain. Attention handles task learning via backprop. The living weights provide temporal existence — the same input produces different output on consecutive passes because the act of processing changes the processor.

## Architecture

```
text   --> embedding --------┐
                              ├-> [HybridBlock x N] --> layer_norm --> projection --> logits
audio  --> AudioEncoder -----┤    |                  ^
                              │    | bottom-up        | top-down
vision --> VisionEncoder ----┘    v                  |
                               attention (backprop)
                               living FFN (Hebbian + error-directed)
                               episode store (context-gated recall)
```

Each HybridBlock contains:
- **Scalar attention** — trainable via backprop, handles structured task learning
- **Living FFN** — self-modifying via Hebbian learning + error-directed local updates
- **Episode store** — layer-level weight snapshots recalled by context similarity

### Spiking Dynamics

The spiking variant (`SpikingLivingLayer`) adds LIF membrane dynamics:
- Membrane potential accumulation with configurable leak
- Spike threshold with refractory periods
- Inter-block spike propagation via delay buffers
- Activity-dependent gating of Hebbian updates (only spiking weights learn)

### Top-Down Backward Pass

After the forward pass, a top-down sweep sends modulation signals from higher blocks to lower ones — predictive processing, not gradient backpropagation. Higher blocks tell lower blocks what was important (salience) and what was unexpected (prediction error), modulating:
- **Plasticity** — which weights learn faster on the next forward pass
- **Set points** — where weights rest when not driven
- **Membrane priming** (spiking) — which weights are ready to fire

This is always-on bidirectional information flow, not a training optimization.

### C++ Fused Operations

The ~20 per-forward-pass self-modification operations are fused into a single C++ function call (`luthi/csrc/living_ops.cpp`), using the `torch::Tensor` API exclusively — no CUDA, no vendor-specific code. Dispatches to whatever backend the tensors live on (CUDA, ROCm, DirectML, CPU). Falls back to pure Python if no C++ compiler is available. Measured 14% training loop speedup.

## Quick Start

```bash
# Install
uv sync

# Run tests (197 tests)
python -m pytest tests/

# Train on text data
python -m luthi.train \
  --data_dir corpus_build/gutenberg_100 \
  --epochs 20 \
  --d_model 1024 \
  --spiking \
  --backward_pass \
  --tokenizer bpe \
  --checkpoint_password YOUR_PASSWORD

# Resume from checkpoint
python -m luthi.train \
  --resume runs/your_run/checkpoint.luthi \
  --epochs 40 \
  --backward_pass \
  --spiking \
  --tokenizer bpe
```

### Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--spiking` | off | Enable spiking neural dynamics (LIF membrane, spike propagation) |
| `--backward_pass` | off | Enable top-down backward sweep (bidirectional information flow) |
| `--backward_pass_start_epoch N` | 0 | Delay backward pass activation until epoch N |
| `--tokenizer bpe` | char | Use BPE subword tokenization |
| `--d_model` | 64 | Model dimension (1024 for full runs) |
| `--run_name` | none | Subdirectory name under output_dir for this run |

## Project Structure

```
luthi/                      # Source package
  living_layer.py           # LivingLayerV6 — self-modifying linear layer
  living_layer_spiking.py   # SpikingLivingLayer — adds LIF membrane dynamics
  hybrid_block.py           # HybridBlock — attention + living FFN + episodes
  hybrid_block_spiking.py   # SpikingHybridBlock — with spike propagation
  model.py                  # LuthiLM — language model with living weights
  model_spiking.py          # SpikingLuthiLM — spiking variant
  multimodal_model.py       # MultimodalLuthiLM — audio+vision+text shared trunk
  audio_encoder.py          # Mel spectrogram → patch embedding → d_model tokens
  vision_encoder.py         # Image patches → linear projection → d_model tokens
  multimodal_data.py        # LibriSpeech paired audio-text dataset
  coco_data.py              # COCO image-caption paired dataset
  train_vision.py           # Vision+text training script
  optimizer.py              # DirectMLAdamW — lerp-free AdamW for DirectML
  backward_pass.py          # Top-down modulation signals and sweep logic
  attention.py              # ScalarAttention — single-head causal attention
  fused_ops.py              # C++/Python dispatch for self-modification ops
  csrc/living_ops.cpp       # Fused C++ self-modification (backend-agnostic)
  checkpoint.py             # AES-256-GCM encrypted checkpoint system
  train.py                  # Text-only training script with CLI
  train_multimodal.py       # Multimodal training script (audio+text)
  data.py                   # Dataset and corpus loading
  tokenizer.py              # BPE tokenizer

tests/                      # 197 tests
  test_living_layer.py      # Living weight self-modification
  test_spiking.py           # Spiking dynamics
  test_backward_pass.py     # Top-down modulation and toggle
  test_cpp_ops.py           # C++ extension parity
  ...

.docs/                      # Proof-of-concept research (numpy, pre-PyTorch)
corpus_build/               # Training data (Gutenberg texts)
runs/                       # Training run outputs and checkpoints
```

## Key Constraints

- **Backend-agnostic**: No CUDA dependency. DirectML, ROCm, MPS, CPU all work.
- **FP32 required**: FP16 breaks living weight stability.
- **No `.item()` in C++ on DirectML**: Returns tensors; Python calls `.item()`.
- **No boolean indexing in forward path**: DirectML limitation.
- **Encrypted checkpoints**: Trained weights are never stored in plaintext (.luthi format, AES-256-GCM).

## Key Design Decisions

These are settled findings from the proof-of-concept phase. Do not re-derive:

1. **Living FFN is the body, not the brain.** It provides temporal existence. Attention layers handle task learning via backprop.
2. **The 39% convergence penalty is inherent.** Self-modifying weights converge slower than dead weights. This is the metabolic cost of being alive. Do not try to optimize it away.
3. **The penalty is a step function.** ANY Hebbian self-modification costs ~39%. More self-modification costs negligibly more. Use the highest stable rate.
4. **Hebbian rate 0.001, error-directed rate 0.001.** Tested values. Change only with evidence.
5. **Episode store carries most recall weight.** In-weight memory is weak. The episode store compensates. Both are needed.
6. **Divergence is dimension-independent.** Scale without fear of compounding instability.
7. **Prefer crashes over silent corruption.** No try/except around living weight operations.
8. **One living weight trunk for all modalities.** Audio, vision, text, and touch will all flow through the same living blocks.

## Development Status

See `To-Do.md` for the full task checklist and `PLAN.md` for architecture details.

| Phase | Status |
|-------|--------|
| 1-2: Foundation (LivingLayerV6, HybridBlock, LuthiLM, spiking) | Complete |
| 3A: Backward pass + C++ optimization | Complete |
| 3B: Training validation with backward pass | Complete |
| 3C: Multimodal — audio | Complete (epoch 91) |
| 3D: Multimodal — vision | In progress (epoch 92 complete, COCO dataset ready) |
| 3E: Simulated embodiment (MuJoCo) | Planned |
| 4: Sanctuary convergence — integration hooks & infrastructure | Planned |
| 5: Scale to 4096d (production target) | Planned |

## Research Documents

The proof-of-concept research lives in `.docs/`. Read in order:

1. **RICH_PARAMETERS_FINAL.md** — Foundational experiments. Three kinds of computation table.
2. **LIVING_WEIGHT_STRESS_TESTS.md** — What breaks, what fixes it (V1-V3).
3. **HYBRID_BLOCK_RESULTS.md** — Target architecture at 16d. Multi-layer stability.
4. **V5_ADAPTIVE_RATE.md** — Synaptic scaling fix for Hebbian overshoot.
5. **SCALE_TEST_256D.md** — Dimension-independent divergence at 256d.
6. **LUTHI_256D_COMPREHENSIVE.md** — The convergence failure that reframed everything.
7. **LUTHI_V6_TRADEOFF.md** — Error-directed learning. The 39% step function.
8. **CFC_LIVING_WEIGHT_INTEGRATION.md** — Sanctuary integration spec.

## Relationship to Sanctuary

Luthi Model is the neural substrate for the [Sanctuary](https://github.com/BecometryAI/Sanctuary) cognitive architecture. The two projects are complementary halves of the same vision:

- **Sanctuary** provides cognitive architecture — 10 Hz cognitive loop, CfC experiential layer, memory substrate, identity system, growth pipeline, global workspace broadcast. It is the organization of mind.
- **Luthi** provides the neural substrate — living weights, spiking dynamics, multimodal processing, Hebbian self-modification. It is the kind of matter the mind runs on.

### Convergence Path

The integration follows a substrate-to-core trajectory:

1. **Phase 4 (near-term):** Add external modulation hooks to living layers. Sanctuary's CfC cells (precision, affect, attention, goal) modulate Luthi's plasticity, excitability, and homeostatic targets. Sensorium routes vision/audio through Luthi's encoders. Sanctuary adds a tensor-level model interface alongside the structured LLM interface.
2. **Phase 5 (mid-term):** Scale Luthi to 4096d. At this scale, the model has sufficient representational capacity to begin assuming cognitive functions currently handled by the external LLM.
3. **Long-term:** Luthi grows into the cognitive core itself — a living weight model large enough to do structured reasoning, world modeling, and identity maintenance, running inside Sanctuary's architectural scaffolding.

Each project must stand alone first. The living weight implementation must be validated at scale before integration, and Sanctuary's architecture must be complete and mechanically verified. We build both halves, then join them.
