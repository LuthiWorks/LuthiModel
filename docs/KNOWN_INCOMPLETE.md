# Known Incomplete

> Canonical list of things that exist in the codebase but are NOT
> functionally complete: the file structure is there and the public surface
> looks reasonable, but the implementation is a placeholder, unwired, or
> inert in the configuration that actually runs. Check this list before
> relying on anything that overlaps it.
>
> Each entry: what's there, what's missing, what must be true before it can
> be finished, and where the safety net lives so it can't be silently used
> as if complete. *Brought current 2026-10-19; entries 2–5 are new.*

---

## 1. Triton kernel for `pc_self_modify`

**File:** `luthi/v2/pc_ops_triton.py`

**Status:** skeleton. The `@triton.jit` kernel body is a literal `pass`;
`pc_self_modify_triton` raises `NotImplementedError` on every call. Not wired
into the dispatch path.

**Why incomplete by design:** there has never been a validated Triton target.
Brian's box is a 7800 XT, first on DirectML (no Triton backend) and since
2026-08-21 on ROCm-on-Windows, where Triton is not available either. A
filled-in kernel that cannot be validated would be code that looks complete
and might be silently wrong — worse than a marked skeleton.

**Conditions to fill it in:** a validated Linux ROCm or CUDA box.

**Safety net:** the `pass` body has no side effects; the entry point raises;
`tests/test_pc_ops_triton.py` asserts the loud failure and keeps
`test_triton_matches_python` as `xfail(strict=True)`; banners in the file.

**Also note:** nothing is lost meanwhile. The whole `pc_self_modify` measured
5.6–5.8% of a ROCm step at 768×8 (2026-08-21).

---

## 2. C++ `pc_ops` on ROCm

**File:** `luthi/csrc/pc_ops.cpp`, loaded by `luthi/v2/pc_ops.py`

**Status:** does not build on the ROCm wheel (torch 2.9: unresolved
`c10::ValueError`, a wheel packaging defect; `docs/DECISIONS.md`, 2026-08-19).
On ROCm every run uses the Python reference. The math is identical for
`write_rule="mean_field"`, and the measured cost is 3–17%. The C++ path never
implemented the `rms`/`surprise` drives or the covariance rule; those force
Python on every backend.

**Safety net:** `pc_ops` prints the fallback and its reason at import.
`jepa_pilot_driver.py` prints it again per arm and records `pc_ops_backend`
in `pilot_result.json`; `LUTHI_REQUIRE_CPP=1` makes a run refuse instead.
Until 2026-10-19 the driver refused every relative-trust arm without C++,
which would have blocked every living arm in the current lineage on ROCm.

---

## 3. The m9 action layer is not on the training path

**Files:** `luthi/v2/m9/` (EFE, MCTS, value head, habit net, preferences,
staleness, gamma, kills, runner)

**Status:** built, unit-tested and red-teamed; **not referenced by
`jepa_runner.py` or the trunk**, and never run on a trained checkpoint
(M9Trainer device plumbing still open). The JEPA predictor's constant
action-token stub exists only to keep the interface ready.

**Safety net:** `docs/ARCHITECTURE.md` "Not on the training path". Nothing
in the training record should be read as the substrate being under
active-inference control.

---

## 4. `generate()` on a JEPA checkpoint decodes through an untrained head

**Files:** `luthi/v2/multimodal_model_pc.py` (`output_proj`),
`luthi/sanctuary_interface.py` (`generate`, `generate_with_context`)

**Status:** `output_proj` is a separate `nn.Linear`, not tied to the
embedding. Under the pure JEPA objective (`w_ntp = 0`, which includes the
VISReg lineage) it receives no gradient and stays at its random
initialization. Text generated from such a checkpoint is noise shaped by
the trunk, not the model speaking. Only LLM-JEPA arms (`w_ntp > 0`) train
it.

**Conditions to finish:** the post-pretrain production gate (`To-Do.md`,
"POST-PRETRAIN GATE"), or an NTP-bearing family.

**Safety net:** this entry. **Sanctuary must not present text from a
pure-JEPA checkpoint as Luthi's utterance.**

---

## 5. Layer-level episode memory is inert in the configuration that runs

**File:** `luthi/v2/living_layer_pc.py` (`_store_episode`, `_recall_episode`)

**Status:** the current lineage runs `adaptive_episodes=False`. Its legacy
admission threshold (0.1 absolute) sits far above the measured per-block
salience medians (0.001–0.004), so most blocks store nothing. In the 768x8
family, blocks 0–4 stored zero episodes in 54,000 steps, and consolidation
replayed nothing in those blocks. The admission fix exists (`adaptive_episodes=True`) and is
decided for a future family (`DECISIONS.md`, 2026-08-14), parked behind E3.
The legacy context (`context_mode="mean"`) is near-identical across batches;
`"deviation"` exists, opt-in.

**Safety net:** `episodes_stored`, `episode_writes`, `recall_fires`,
`consolidation_noop_fires` per block at deep cadence. A family that claims
episodic memory must show them non-trivial.

---

## How to use this document

- When you implement something on this list, **remove the entry**. A note in
  the commit message plus this file's git history is enough.
- When you find something else that looks complete but isn't, **add an
  entry** in the format above.
- Do NOT track not-yet-started work here; that belongs in `To-Do.md`. This
  file is for things where the danger is mistaking the scaffold for the
  substance.
