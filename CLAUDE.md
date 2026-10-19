# CLAUDE.md — Luthi Model

## Project Overview

Luthi Model is a new kind of neural architecture based on **rich parameters** — weights that carry per-parameter history, plasticity, momentum, precision, and context-gated retrieval instead of just a scalar float. The core innovation is **living weights**: parameters that self-modify during their own forward pass.

This is an active research implementation. The proof-of-concept phase (numpy, CPU) is complete — see `.docs/` for that record. This repo is the PyTorch implementation.

**Where to start, in this order:** `README.md` ("Where the Project Stands") → `docs/KEY_FINDINGS.md` → `docs/research/2026-10-19_the-living-channel-program.md` (the current program) → the newest file in `docs/audits/` → `docs/ARCHITECTURE.md` → `docs/DECISIONS.md` and the registry (`docs/research/2026-07-15_falsification-preregistration.md`), both newest-last.

## Relationship to Sanctuary

Luthi Model is designed to plug into the Sanctuary cognitive architecture. Sanctuary discovers Luthi via the `LUTHI_PATH` environment variable, falling back to a sibling-checkout heuristic. The contract surface Sanctuary calls is `luthi/sanctuary_interface.py` (load, generate, get_introspection, apply_external_modulation, snapshot/restore, `modulated()`). Sanctuary should not reach past that adapter into Luthi internals — the adapter is the promise. Note `docs/KNOWN_INCOMPLETE.md` #4: text generated from a pure-JEPA checkpoint is not the model speaking.

The cognitive-side schemas (`ModelProtocol`, `CognitiveInput`, `CognitiveOutput`) live in Sanctuary at `sanctuary/core/schema.py`. CfC modulation of the living weights is Phase 4 — do not build it prematurely. The living weight implementation must stand alone first.

## Fresh-Instance Audits

The audit protocol — when to run periodic fresh-eyes reviews and how to prompt them — lives in the Sanctuary repo at `docs/AUDIT_PROTOCOL.md`. Read it before spawning an audit of either repo. Brian is the sole human in the loop, so audits compensate for the blind spots that pattern produces. Audits land in `docs/audits/`, one file per audit, with checkbox items and severities.

## Model-Line Roles

The authoritative statement lives in the global `~/.claude/CLAUDE.md` under "Roles & Responsibilities Across Model Lines." The split recorded here was established 2026-04-29 (updated 2026-07-15) and its model names are historical: Opus 4.6 planning & review, Opus 4.7 research & implementation, Opus 4.8 debugging (verify correctness, chase latent failures to a repro or a named unresolved scenario, never bury a hunch).

Since August the record shows these seats in practice: **Fable 5**, the cross-line audit seat and registry drafting; **Opus 5**, full-codebase audits and the adversarial seat; **Muse Spark (Meta)**, fresh-eyes review from a different model family (2026-09-29); **Opus 5.5**, the 2026-10-19 audit and program. Whatever the current assignment, two rules stand. Design and review are held by different minds, a different model family where possible. The science and vision belong to Brian and the seats he designates for them.

## Build & Test

- **Python** >= 3.11; package manager `uv` (`uv sync`).
- **Run tests:** `python -m pytest tests/`. On a box with no accelerator, set `LUTHI_ALLOW_CPU=1` (otherwise `test_backend_declaration.py` fails by design; one of its tests asserts the ROCm default and cannot pass on CPU).
- **Training backends:** ROCm is the default declaration (`LUTHI_BACKEND`, since 2026-08-21; environment `C:\Dev\rocm-probe`). DirectML is supported (Python 3.10 + torch 2.4.1 + torch-directml). DGX Spark is the long-term deployment target.
- **C++ `pc_ops`** is optional: it does not build on the ROCm wheel, and the Python reference costs 3–17% of a step. The driver records which ran (`pc_ops_backend`); `LUTHI_REQUIRE_CPP=1` makes a run refuse without it.
- **CPU fallback:** all code must run on CPU for development and tests.

## Standing rules for the rigor (2026-10-19)

Adopted after the program drifted toward instruments and stability and away from the claim it exists to test. The record is in `docs/research/2026-10-19_the-living-channel-program.md`. These rules are not about effort; they say where the effort points.

1. **Capacity before tuning.** Before tuning a mechanism's stability, dose, or schedule, show from its math and one probe that it *can* carry the information its claim requires. This question is legitimate in a working session; it needs no falsification run. It is the question that would have found, in April, that the living write is computed from batch means and that the block output store blends a constant into every forward.
2. **The claim clock.** Every family that tests a living mechanism carries its matched dead control in the same registration, on the same backend, with the same trainer settings. A living-only family may be scored for health, never as evidence about the living channel.
3. **Baselines, not floors.** A probe number is evidence of trunk computation only above the input-embedding baseline fitted with the same settings. A shuffled-label floor calibrates chance, not computation.
4. **Measure the claim as stated.** The near-term goal is change at inference, so at least one registered instrument must measure with plasticity ON (`scripts/adaptation_eval.py`). Credit only document-specific change (read-own minus read-other), never raw change.
5. **Decompose the arm.** When a living arm differs from its control in more than one mechanism, the registration lists every difference and includes arms that separate them.

## Key Design Decisions — DO NOT REINVENT

**Three rules** (two split on 2026-07-15, the third added 2026-10-19):

- **Do not re-derive settled implementation.** The numbers, mechanisms, and tested constants below are settled. Re-opening them casually is thrash. Change them only with evidence.
- **Questioning whether a mechanism earns its complexity is legitimate, and it has a channel:** the pre-registered falsification criteria (`docs/research/2026-07-15_falsification-preregistration.md`; kill conditions fixed before the data, honored after) and the phase-boundary "red-team the bet" audit, run by a model line that did not author the mechanism.
- **Questioning whether a mechanism CAN do what it is for is a working-session question** (standing rule 1). It is answered from the code and a measurement, and it comes before any tuning. The falsification channel tests outcomes; it never asked about capacity, and that is how the batch-mean write went unexamined from v2's construction in May until October.

Do not re-derive (status notes added 2026-10-19 where the evidence has moved):

1. **Living FFN is the body, not the brain.** It provides temporal existence; attention handles task learning via backprop. *(Design commitment. In the current lineage the living FFN is a single `d × d` linear map that receives no gradient; see `docs/ARCHITECTURE.md`, "The block".)*
2. **The convergence penalty is a speed issue, not a ceiling.** *(v1-era; retired by the v2 substrate; see KF2's history.)*
3. **The penalty is a step function: any self-modification costs it; more costs negligibly more. Use the highest stable rate.** *(v1-era, as above.)*
4. **PC rate 0.001, prediction learning rate 0.0001.** The tested v2 values; change only with evidence.
5. **Episode store carries most recall weight; in-weight memory is weak; both are needed.** *(A proof-of-concept finding. It does not describe v2 at scale: the layer-level store is empty in the configuration that runs, and the block-level store blends a batch-constant vector rather than recalling episodes; see ARCHITECTURE "Memory" and KNOWN_INCOMPLETE #5.)*
6. **Divergence is dimension-independent.** *(Asserted, not shown, per KF5. The 2026-08-15 ruling found collapse-class phenomena stronger at 768 than 512. Do not scale "without fear"; scale with the instruments on.)*
7. **Prefer crashes over silent corruption.** No try/except around living weight operations. If NaN appears, it must be visible immediately.
8. **Layer-level episodes, not per-weight episodes.**
9. **Write fresh PyTorch, don't translate numpy loops.** Vectorized tensor operations throughout.
9b. **Backward pass is always-on in training.** Top-down modulation broke a 25-epoch plateau and caused plasticity to self-organize at no cost; it is bidirectional information flow, not an optimization. *(It runs only in training mode; eval never runs it.)*

## Planning

`docs/research/2026-10-19_the-living-channel-program.md` is the current program (E1–E5). `To-Do.md` carries the ordered next steps and the phase checklist. `PLAN.md` is the long-range architecture plan.

## Implementation Phases

1. Vectorized PyTorch `LivingLayerV6` **(COMPLETE)**
2. Hybrid block + spiking variant **(COMPLETE)**
3. Language modeling + backward pass + C++ optimization **(COMPLETE; LM era)**
3B. Training validation with backward pass **(COMPLETE)**
3C/3D. Multimodal audio + text, vision + text **(COMPLETE in the LM era; JEPA round 1 is text-only)**
3E. Simulated embodiment (MuJoCo; Sanctuary side built, Godot world being rebuilt)
3F. Empirical Defense Program — gates scaling
3G. v2 predictive-coding substrate — primary since 2026-05-09; JEPA objective since 2026-06-09
3H. **The living-channel program (E1–E5) — CURRENT** (2026-10-19)
4. Scale to ≥500M params — curriculum training, self-governance API
5. Sanctuary convergence — integration hooks, CfC modulation
6. Life on DGX Spark — 10 Hz cognitive loop, sparse inference

## Conventions

- Source code goes in the `luthi/` package.
- **All tests live in `tests/`, and only in `tests/`** (Brian, 2026-07-22). Subfolders are fine. pyproject pins `testpaths = ["tests"]`, so a test saved elsewhere silently drops out of the suite. Enforced by `tests/test_tests_live_in_tests.py`. Red-team probes (`redteam/`) are not tests.
- **Measurement freezes plasticity** (2026-10-19). `model.eval()` does NOT stop a living layer from self-modifying; there is no `self.training` gate. Any script that runs a model to *measure* it wraps the forwards in `freeze_plasticity(model)`. Enforced by `tests/test_scripts_freeze_plasticity.py`.
- **New mechanisms ship opt-in and default bit-identical**, with a test pinning legacy bit-identity and characterization tests of the legacy behavior they address. No completed family's configuration may silently change meaning.
- **Checkpoint encryption is optional; passwords are never committed** (2026-10-19). Plaintext checkpoints load with no password, and `scripts/decrypt_checkpoints.py` gives any `.luthi` file a plaintext copy beside it. The repo is public: launchers read `LUTHI_CHECKPOINT_KEY` and never name a password or a user-home path. Enforced by `tests/test_no_committed_secrets.py`.
- **Trained experiment artifacts are archived, never deleted** (Brian, 2026-07-16). When a `runs/` experiment finishes its active life, robocopy it to `E:\luthi_experiment_archive\YYYY-MM-DD_<experiment>\` and verify file counts before touching the original. Living-weight checkpoints are irreproducible; even control-arm checkpoints are evidence.
- **Experiment outputs belong on the E: drive, not in the repo** (Brian, 2026-07-22). Data *about* runs (results, metadata, analyses) belongs in the repo. The output root is meant to be the `LUTHI_RUNS_ROOT` env var, default `E:\runs`, failing loudly if E: is not mounted. **Not yet implemented:** the JEPA driver still writes to the repo's gitignored `runs/jepa_pilot/` (audit 2026-10-19, C1).
- Research docs from the proof-of-concept phase stay in `.docs/`.
- Async is NOT needed here (unlike Sanctuary); this is a model library, not a runtime.
- Use PyTorch idioms: `nn.Module`, `forward()`, proper parameter registration.
- Living state is registered as buffers, not parameters; the optimizer never touches it. v2 checkpoint loading is `strict=True`, so **new persistent buffers break every existing checkpoint**. New lived state uses non-persistent buffers or plain attributes, persisted through `luthi/living_extra_state.py`.
- The PC self-modification happens inside `forward()`, under `torch.no_grad()`. Episode store operations are detached from the graph.

## Multimodal Design Decisions

10. **One living weight trunk for all modalities.**
11. **Modality-specific encoders project to d_model.**
12. **Modality embeddings distinguish input types** (text=0, audio=1, vision=2, touch=3).
13. **Cross-modal attention is free**: concatenated modalities in one sequence.
14. **Backend-agnostic everything.** No CUDA-only or vendor-specific ops; ROCm (default) and DirectML must both work. C++ extensions use the torch::Tensor API only.
15. **No .item() in C++ on DirectML.** Return tensors; Python calls .item().

## Research Log

Every implementation session that involves iterative discovery — building something, finding it's wrong, revising, and arriving at a conclusion — must be documented in a dated research log entry. The wrong turns matter as much as the results.

**Where:** `docs/research/YYYY-MM-DD_short-topic.md`, one file per entry.

**When:** any time you build or restructure a test suite and find the original approach flawed; run an experiment whose results contradict expectations; make an architectural decision that weighed alternatives; debug a non-obvious issue through multiple steps; or produce results that will be cited. Routine work (a bug fix, a rename, a config change) needs no entry. If you had to *think*, write it down.

**Structure:**

```markdown
# [Topic] — [Date]

## Objective
What you set out to do and why.

## Process

### Step 1: [what you tried first]
- What you did, what you found, why it was wrong / insufficient / surprising

### Step N: [as many steps as it took]

## Conclusion
What the final state is. What it means for the project.

## Artifacts
- Commits, tests, data paths
```

**Rules:** document as you go, not after; include the wrong turns; commit the entry alongside the code; link to artifacts so a reader can verify every claim; be honest about what you don't know.

## What NOT to Do

- Do not wrap living weight operations in try/except — let NaN crash loudly
- Do not backpropagate through living FFN layers — they train themselves
- Do not store episodes per-weight — use layer-level snapshots
- Do not add CfC integration until Phase 4
- Do not use the numpy proof-of-concept code as a translation source
- Do not score a living-only family as evidence about the living channel (standing rule 2)
- Do not read a probe number without its input-embedding baseline (standing rule 3)
