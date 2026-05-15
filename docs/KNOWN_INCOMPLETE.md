# Known Incomplete

> Canonical list of things that exist in the codebase but are NOT
> functionally complete — file structure is there, the public surface
> looks reasonable, but the implementation is a placeholder. Future
> instances must check this list before relying on anything that
> overlaps it.
>
> Each entry: what's there, what's missing, what conditions need to
> be met before it can be finished, and where the safety net lives so
> it can't be silently used as if complete.

---

## 1. Triton kernel for `pc_self_modify`

**File**: `luthi/v2/pc_ops_triton.py`

**Status**: Skeleton. The `@triton.jit` kernel body is a literal `pass`.
The Python entry point `pc_self_modify_triton` raises `NotImplementedError`
on every call. NOT wired into the dispatch path in `luthi/v2/pc_ops.py`.

**Why incomplete by design (not by accident)**:
Brian's hardware is a 7800 XT via DirectML. Triton has no DirectML
backend, and ROCm-on-Windows for consumer RDNA3 cards is patchy
enough that we can't reliably run Triton kernels on the dev box.
Without a GPU to validate against, filling in the kernel body means
shipping code that *looks* complete but might be silently wrong on the
first machine that actually runs it. That's worse than a clearly-marked
skeleton — it would violate the "prefer crashes over silent corruption"
principle the README is built on.

**Conditions to fill it in**:
- ROCm + the 7800 XT under Linux/WSL2, OR
- DGX Spark (Phase 7 deployment hardware, CUDA), OR
- A borrowed/cloud NVIDIA box for one-off validation.

**Safety net** (so this can't get silently used as if complete):
1. The kernel body is `pass` — calling the kernel directly produces no
   side effects.
2. `pc_self_modify_triton` raises `NotImplementedError` with a message
   pointing back to the Python path.
3. `tests/test_pc_ops_triton.py::test_triton_not_implemented_raises_loud`
   asserts the loud-failure contract on every test run.
4. `tests/test_pc_ops_triton.py::test_triton_matches_python` is
   `xfail(strict=True)` — if anyone fills in the kernel and the math
   doesn't match the Python reference, CI fails noisily.
5. Big ASCII banner in the file's module docstring. Big banner at the
   kernel-body `pass`. No quiet way to mistake the file for complete.

**What gets unblocked when this lands**:
- Sparse PC gating gets a fast path. Right now the C++ kernel skips
  when `sparse_gate is not None`, falling back to Python — so sparse
  gating works but does NOT get the C++ acceleration that makes the
  rest of pc_self_modify fast at production scale.
- Phase 7 (DGX Spark deployment) gets the 10 Hz cognitive loop
  bandwidth budget the architecture is designed for.

**Reference implementation** (the bit-identity spec):
`luthi/v2/pc_ops.py::_pc_self_modify_python`. Section labels (a-k) in
the Python reference match the structure in the kernel skeleton.

---

## How to use this document

- When you implement something on this list, **remove the entry**.
  Don't just edit it to say "done." A short note in the relevant
  commit message + this file's git history is enough.
- When you discover something else that looks complete but isn't,
  **add an entry**. Use the format above: what's there, why it's
  incomplete, what unblocks it, what the safety net is.
- Do NOT use this file to track "TODO" items that aren't yet started.
  Those belong in `To-Do.md`. This file is specifically for
  half-implemented things where the danger is mistaking the scaffold
  for the substance.
