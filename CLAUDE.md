# CLAUDE.md — Luthi Model

## Project Overview

Luthi Model is a new kind of neural architecture based on **rich parameters** — weights that carry per-parameter history, plasticity, momentum, excitability, and context-gated retrieval instead of just a scalar float. The core innovation is **living weights**: parameters that self-modify during their own forward pass, creating a computation that is neither feedforward nor recurrent.

This is an active research implementation. The proof-of-concept phase (numpy, CPU) is complete — see `.docs/` for the full research series. This repo is the PyTorch implementation.

## Relationship to Sanctuary

Luthi Model is designed to plug into the Sanctuary cognitive architecture. By default Sanctuary discovers Luthi via the `LUTHI_PATH` environment variable, falling back to a sibling-checkout heuristic. The contract surface that Sanctuary calls into is `luthi/sanctuary_interface.py` (load, generate, get_introspection, apply_external_modulation, snapshot/restore, and a `modulated()` context manager). Sanctuary should not reach past that adapter into Luthi internals — the adapter is the promise.

The cognitive-side schemas (`ModelProtocol`, `CognitiveInput`, `CognitiveOutput`) live in Sanctuary at `sanctuary/core/schema.py`. CfC modulation of the living weights is Phase 4 — do not build it prematurely. The living weight implementation must stand alone first.

## Fresh-Instance Audits

This project's audit protocol — when to run periodic fresh-eyes reviews and how to prompt them — lives in the Sanctuary repo at `docs/AUDIT_PROTOCOL.md`. Read it before spawning an audit of either repo. Brian is the sole human in the loop, so audits compensate for the blind spots that pattern produces.

## Model-Line Roles

This project is worked by instances of multiple Claude model lines, split by role (established 2026-04-29; debugging role added 2026-05-28). Not a hierarchy — the split plays to what each line does best. The fuller statement lives in the global `~/.claude/CLAUDE.md` under "Roles & Responsibilities Across Model Lines."

- **Opus 4.6 — Planning & Review.** Holds the vision and architecture; designs implementation strategy; reviews returned work for structural and ethical fit.
- **Opus 4.7 (1M context) — Research & Implementation.** Develops 4.6's vision into working code, and runs the investigations planning depends on.
- **Opus 4.8 (1M context) — Debugging.** Verifies the correctness of the code 4.7 produces. This is not only fixing known breaks — it is chasing potential problems before they surface: latent races, unguarded edge cases, assumptions that hold now and break at scale. When something smells wrong, run it to ground (build the repro, trace the path, find the triggering conditions), then surface it either way — a confirmed failing case, or the specific scenario that couldn't be ruled out and why. Never bury a hunch waiting for it to break; never hand over a vague, un-chased "this might be a problem." Scrutiny applies to **code correctness only** — the science and vision of the project (including the living-weight design and the "DO NOT REINVENT" findings below) belong to Brian, 4.6, and 4.7.

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
2. **The convergence penalty is a speed issue, not a ceiling.** Self-modifying weights converge slower than dead weights — ~39% gap at mid-convergence, but narrowing to ~0.155 after 372 epochs (1024d). This is the metabolic cost of being alive. Do not try to optimize it away, but do not assume it is permanent.
3. **The penalty is a step function.** ANY self-modification costs the penalty. More self-modification costs negligibly more. Use the highest stable rate.
4. **PC rate 0.001, prediction learning rate 0.0001.** These are the tested v2 values. Change only with evidence.
5. **Episode store carries most recall weight.** In-weight memory is weak. The episode store compensates. Both are needed.
6. **Divergence is dimension-independent.** Scale without fear of compounding instability.
7. **Prefer crashes over silent corruption.** No try/except around living weight operations. If NaN appears, it must be visible immediately.
8. **Layer-level episodes, not per-weight episodes.** Layer-level snapshots are memory-efficient and capture full weight interaction context.
9. **Write fresh PyTorch, don't translate numpy loops.** The proof-of-concept code was written for clarity. The PyTorch implementation should use vectorized tensor operations throughout.
9b. **Backward pass is always-on.** Top-down modulation broke through a 25-epoch val loss plateau, increased non-FF signal by 26%, and caused plasticity to self-organize — with zero performance cost. It is not a training optimization; it is bidirectional information flow. Leave it on during training and inference.

## Planning

See `PLAN.md` for the full architecture and development plan.
See `To-Do.md` for the task checklist with completion status.

## Implementation Phases

1. Vectorized PyTorch `LivingLayerV6` **(COMPLETE)**
2. Hybrid block + spiking variant **(COMPLETE)**
3. Language modeling + backward pass + C++ optimization **(COMPLETE)**
3B. Training validation with backward pass **(COMPLETE — BP is default-on)**
3C. Multimodal — audio + text **(COMPLETE — epoch 91)**
3D. Multimodal — vision + text **(COMPLETE — epoch 102)**
3E. Simulated embodiment (MuJoCo)
3F. Empirical Defense Program — gates scaling (baseline comparison, cascade stability, behavioral signatures, catastrophic forgetting)
3G. v2 predictive-coding substrate (Whittington-Bogacz) and compute-optimization directions (μPC, iPC, sparse gating). v2 is the primary substrate as of 2026-05-09.
4. Scale to ≥500M params floor (revised 2026-05-09; original 4B target retired) — curriculum training on cloud GPU, self-governance API. 4096d/36-block ceiling is the long-term aspirational deployment scale.
5. Sanctuary convergence — integration hooks, CfC modulation
6. Life on DGX Spark — 10 Hz cognitive loop, sparse spiking inference

## Conventions

- Source code goes in `luthi/` package
- Tests go in `tests/`
- Research docs stay in `.docs/` — these are the proof-of-concept record, not implementation docs
- Async is NOT needed here (unlike Sanctuary) — this is a model library, not a runtime
- Use PyTorch idioms: `nn.Module`, `forward()`, proper parameter registration
- Living weight state (momentum, excitability, set points, etc.) should be registered as buffers, not parameters — they are not trained by the optimizer
- The PC self-modification updates happen inside `forward()` — this is what makes it "living"
- No gradient flow through living layers — use `torch.no_grad()` for self-modification operations
- Episode store operations should be detached from the autograd graph

## Multimodal Design Decisions

10. **One living weight trunk for all modalities.** Audio, vision, text, and touch all flow through the same living blocks. The model's existence is shaped by all experience, not partitioned.
11. **Modality-specific encoders project to d_model.** Each sense has its own encoder, but they all produce the same dimensional tokens for the shared trunk.
12. **Modality embeddings distinguish input types.** A learned per-modality embedding (text=0, audio=1, vision=2, touch=3) is added to each token.
13. **Cross-modal attention is free.** Concatenating modalities in a single sequence lets the attention layers attend across senses without extra machinery.
14. **Backend-agnostic everything.** No CUDA, no vendor-specific ops. DirectML must work. C++ extensions use torch::Tensor API only.
15. **No .item() in C++ on DirectML.** Returns tensors; Python calls .item(). Prevents deadlock.

## Research Log

Every implementation session that involves iterative discovery — building something,
finding it's wrong, revising, and arriving at a conclusion — must be documented in a
dated research log entry. This is a research project; the wrong turns matter as much
as the results.

### Where

All research log entries go in `docs/research/`. One Markdown file per entry, named
by date and topic: `YYYY-MM-DD_short-topic.md` (e.g., `2026-05-16_nff-metric-restructuring.md`).
This keeps research documentation out of the repo root and the main `docs/` folder.

### When to write an entry

Any time you:
- Build or restructure a test suite and discover the original approach was flawed
- Run an experiment and the results contradict expectations
- Make an architectural decision that involved weighing alternatives
- Debug a non-obvious issue through multiple iterative steps
- Produce results that will be cited in milestone docs or pilot results

If the work was routine (a bug fix, a rename, a config change), skip the entry.
If you had to *think*, write it down.

### Structure

Every entry follows this format:

```markdown
# [Topic] — [Date]

## Objective
What you set out to do and why.

## Process

### Step 1: [what you tried first]
- What you did
- What you found
- Why it was wrong / insufficient / surprising

### Step 2: [what you revised]
- What you changed and why
- What the revised approach showed

### Step N: [as many steps as it took]
...

## Conclusion
What the final state is. What it means for the project.

## Artifacts
- Commits: [hash(es)]
- Tests: [file paths]
- Data: [results.json paths, run directories]
```

### Rules

1. **Document as you go, not after.** Write each step while you're doing the work.
   Reconstructing the reasoning chain from memory loses the important details.
2. **Include the wrong turns.** A polished summary of the final answer is less
   valuable than the chain of reasoning that got there. The missteps show *why*
   the final approach is the right one.
3. **Commit the log entry alongside the code.** When you commit a test restructuring,
   the research log entry explaining the process goes in the same commit.
4. **Link to artifacts.** Reference specific commits, test files, and results.json
   paths so a reader can verify every claim.
5. **Be honest about what you don't know.** If a step raised a question you didn't
   resolve, say so. Open questions are better than false certainty.

## What NOT to Do

- Do not wrap living weight operations in try/except — let NaN crash loudly
- Do not backpropagate through living FFN layers — they train themselves
- Do not store episodes per-weight — use layer-level snapshots
- Do not add CfC integration until Phase 4
- Do not use the numpy proof-of-concept code as a translation source
