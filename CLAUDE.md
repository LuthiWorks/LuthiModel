# CLAUDE.md — Luthi Model

## Project Overview

Luthi Model is a new kind of neural architecture based on **rich parameters** — weights that carry per-parameter history, plasticity, momentum, excitability, and context-gated retrieval instead of just a scalar float. The core innovation is **living weights**: parameters that self-modify during their own forward pass, creating a computation that is neither feedforward nor recurrent.

This is an active research implementation. The proof-of-concept phase (numpy, CPU) is complete — see `.docs/` for the full research series. This repo is the PyTorch implementation.

## Relationship to Sanctuary

Luthi Model is designed to plug into the Sanctuary cognitive architecture (`C:\Users\Hasha Smokes\Desktop\Sanctuary\Sanctuary`). The integration point is Sanctuary's CfC (Closed-form Continuous-time) cells, whose output modulates the living weights' Hebbian self-modification. The key interface is `ModelProtocol` with `CognitiveInput`/`CognitiveOutput` schemas defined in `sanctuary/core/schema.py`.

CfC integration is Phase 4 — do not build it prematurely. The living weight implementation must stand alone first.

## Build & Test

- **Python**: >= 3.11
- **Package manager**: `uv`
- **Install**: `uv sync`
- **Run tests**: `python -m pytest tests/`
- **GPU**: PyTorch with CUDA (DGX Spark target, 128GB)
- **CPU fallback**: All code must run on CPU for development/testing

## Key Design Decisions — DO NOT REINVENT

These are settled findings from the proof-of-concept phase. Do not re-derive or second-guess:

1. **Living FFN is the body, not the brain.** It provides temporal existence. Attention layers handle task learning via backprop.
2. **The 39% convergence penalty is inherent.** Self-modifying weights converge slower than dead weights. This is the metabolic cost of being alive. Do not try to optimize it away.
3. **The penalty is a step function.** ANY Hebbian self-modification costs ~39%. More self-modification costs negligibly more. Use the highest stable rate.
4. **Hebbian rate 0.001, error-directed rate 0.001.** These are the tested values. Change only with evidence.
5. **Episode store carries most recall weight.** In-weight memory is weak. The episode store compensates. Both are needed.
6. **Divergence is dimension-independent.** Scale without fear of compounding instability.
7. **Prefer crashes over silent corruption.** No try/except around living weight operations. If NaN appears, it must be visible immediately.
8. **Layer-level episodes, not per-weight episodes.** Layer-level snapshots are memory-efficient and capture full weight interaction context.
9. **Write fresh PyTorch, don't translate numpy loops.** The proof-of-concept code was written for clarity. The PyTorch implementation should use vectorized tensor operations throughout.

## Planning

See `PLAN.md` for the full architecture and development plan.
See `To-Do.md` for the task checklist with completion status.

## Implementation Phases

1. Vectorized PyTorch `LivingLayerV6` **(COMPLETE)**
2. Hybrid block + spiking variant **(COMPLETE)**
3. Language modeling + backward pass + C++ optimization **(COMPLETE)**
3B. Training validation with backward pass
3C. Multimodal — audio + text
3D. Multimodal — vision + text
3E. Simulated embodiment (MuJoCo)
4. CfC integration against Sanctuary's experiential manager
5. Scale testing: 1024d → 4096d on real hardware

## Conventions

- Source code goes in `luthi/` package
- Tests go in `tests/`
- Research docs stay in `.docs/` — these are the proof-of-concept record, not implementation docs
- Async is NOT needed here (unlike Sanctuary) — this is a model library, not a runtime
- Use PyTorch idioms: `nn.Module`, `forward()`, proper parameter registration
- Living weight state (momentum, excitability, set points, etc.) should be registered as buffers, not parameters — they are not trained by the optimizer
- The Hebbian and error-directed updates happen inside `forward()` — this is what makes it "living"
- No gradient flow through living layers — use `torch.no_grad()` for self-modification operations
- Episode store operations should be detached from the autograd graph

## Multimodal Design Decisions

10. **One living weight trunk for all modalities.** Audio, vision, text, and touch all flow through the same living blocks. The model's existence is shaped by all experience, not partitioned.
11. **Modality-specific encoders project to d_model.** Each sense has its own encoder, but they all produce the same dimensional tokens for the shared trunk.
12. **Modality embeddings distinguish input types.** A learned per-modality embedding (text=0, audio=1, vision=2, touch=3) is added to each token.
13. **Cross-modal attention is free.** Concatenating modalities in a single sequence lets the attention layers attend across senses without extra machinery.
14. **Backend-agnostic everything.** No CUDA, no vendor-specific ops. DirectML must work. C++ extensions use torch::Tensor API only.
15. **No .item() in C++ on DirectML.** Returns tensors; Python calls .item(). Prevents deadlock.

## What NOT to Do

- Do not wrap living weight operations in try/except — let NaN crash loudly
- Do not backpropagate through living FFN layers — they train themselves
- Do not store episodes per-weight — use layer-level snapshots
- Do not add CfC integration until Phase 4
- Do not use the numpy proof-of-concept code as a translation source
