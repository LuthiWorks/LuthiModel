# ROCm + WSL2 + RDNA3 Feasibility — 2026-05-23

> **Status: complete (decision document, not implementation).** Investigation of
> whether moving Luthi training to ROCm-on-WSL2 is currently feasible on
> Brian's hardware (Windows 11 + RX 7800 XT), and what it would unlock vs
> cost. Implementation gated on Brian's decision.

## Objective

After shipping C++ `sparse_gate` support in `luthi/csrc/pc_ops.cpp`
(commit `a6b7ab6`, 2026-05-22), the immediate sparse-gating accelerator
gap on the current DirectML host was closed. But the Triton kernel
skeleton at `luthi/v2/pc_ops_triton.py` remained unfilled, and the
larger Phase 4 4B-param training case still ran on DirectML — adequate
but not optimal for training-scale workloads.

The question Brian raised: could a Linux container on the current
Windows host unblock the Triton path AND give Phase 4 training a
better backend? Specifically: is the WSL2 + ROCm + Triton chain
feasible *today* on his RX 7800 XT (RDNA3, gfx1101)?

The skeleton's own comment (written months ago) said "ROCm on Windows
for consumer RDNA3 cards is patchy." The investigation goal was to
determine if that's still true at the current ROCm release.

## Process

### Step 1: Identified what the skeleton has always assumed but never tested

Reading `pc_ops_triton.py` carefully: the skeleton explicitly listed
"Move to Linux/WSL2 + ROCm on the same 7800 XT" as one of three
practical paths to validating the kernel. That path was treated as
*possible-but-unverified*. The audit-of-record never actually checked
the current ROCm-on-WSL2 support matrix for gfx1101.

Triton's situation by knowledge cutoff (January 2026):
- Triton's AMD backend was upstreamed in 2024.
- ROCm 6.x added RDNA3 support starting late 2024.
- AMD officially supported ROCm-on-WSL2 since 2023-2024.

What I didn't know: whether the 7800 XT specifically is on the
current AMD-supported list for WSL2, and whether Triton's AMD backend
covers gfx1101.

### Step 2: WebSearch for current state

Three searches:
1. `ROCm WSL2 supported GPUs 2026 RDNA3 RX 7800 XT`
2. `AMD ROCm 6.x WSL2 installation guide consumer GPU 2026`
3. `Triton AMD backend ROCm gfx1101 status 2026`

### Step 3: Synthesis — three load-bearing facts

**(A) The 7800 XT is NOT on AMD's officially supported list for WSL2.**

Per the [official ROCm Radeon WSL compatibility matrix][rocm-wsl-compat],
ROCm 6.1.3 — the last version AMD validated for WSL2 — only officially
supports gfx1100 (the 7900 XTX). The 7800 XT (gfx1101) is in the same
RDNA3 architecture family but is not on the validated SKU list. There
was an open feature request on the ROCm repo specifically asking for
"WSL2 + other RDNA3 GPUs rather than navi31 core" support ([ROCm/ROCm
issue #3863][rocm-issue-3863]).

ROCm 6.4.2 added explicit support for additional RDNA3 variants
(including the 7700 XT per the AMD docs cited below), so the support
matrix is expanding, but the 7800 XT specifically still appears to
require the workaround path below.

**(B) The workaround that works in practice:**

The community-documented path (sources below, including independent
April 2026 verification) is:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

This tells the ROCm runtime to treat the 7800 XT as a 7900 XTX for
compatibility purposes. All RDNA3 cards (7900 XTX, 7900 XT, 7800 XT,
7700 XT, 7600) use the same `gfx11.0.0` LLVM target family, so the
override is structurally honest — same instruction set, same compute
units, just different SKU markers.

Independent verification: the [CraftRigs guide][craftrigs] reports
"Commands tested on RX 7900 XTX and RX 7800 XT on WSL2 Ubuntu 22.04 in
April 2026." This is the most recent third-party confirmation I found.

**(C) Triton's AMD backend on gfx1101 is EXPERIMENTAL.**

From AMD's own ROCm documentation cited in the searches: "AOTriton
0.10b introduces official support for gfx950 and gfx1201, along with
**experimental** support for gfx1101, gfx1151, gfx1150, and gfx1200."

So Triton CAN run on the 7800 XT through ROCm + WSL2, but Brian would
be on the experimental tier, not the officially supported one. The
implication for our skeleton: bit-identity might pass, or there could
be numerical edge cases (associativity, kernel-launch failures on
specific tile shapes, etc.) that need investigation. The skeleton's
contract (the `xfail strict` test that locks in correctness) is
exactly the right tool for that — if Triton produces incorrect math,
the test catches it.

### Step 4: System-requirement gotchas surfaced by the search

These would have bitten us during setup if I hadn't searched:

- **Use Ubuntu 22.04, not 24.04.** Multiple sources explicitly note
  that Ubuntu 24.04 breaks GPU passthrough at the WSL2 kernel level
  for ROCm. 22.04 is the validated combination.
- **Windows 11 22H2 minimum.** Older builds lack the WSL2 GPU
  passthrough kernel surface ROCm needs.
- **AMD Adrenalin driver 23.40.27.06 or newer.** This is the Windows-
  side driver that publishes the GPU into WSL2.
- **Pin ROCm 6.1.3.** Last AMD-validated WSL2 version. Newer releases
  (6.4.2, 7.x) are documented but the 7800 XT compatibility status
  on those isn't as battle-tested by community reports.
- **Disk: ~15-30 GB** for the WSL2 distro + ROCm + PyTorch ROCm wheels.

### Step 5: Cost / benefit reconsidered post-research

Before the research, I framed this as "if it works, do it; if it
doesn't, wait for hardware." The research clarifies that it's
neither — it's a third state: "works with a workaround, on the
experimental tier of Triton support."

**Benefits, weighted by realism:**

1. **Triton kernel validation becomes possible.** Could fill in the
   skeleton, run the bit-identity test. Risk: experimental gfx1101
   support might hit edge cases. The skeleton's `xfail strict`
   contract catches incorrect math but doesn't tell us if the kernel
   simply fails to compile on certain tile shapes.

2. **Phase 4 training acceleration.** This is probably the bigger
   prize. ROCm + PyTorch is significantly more efficient than
   DirectML at training-scale workloads. For a 4B-param curriculum
   training run, the time savings could be substantial. The 7800 XT's
   16 GB VRAM is identical between DirectML and ROCm — no VRAM gain,
   just compute-efficiency gain.

3. **Path toward emotion-vector instrumentation.** The chain I
   identified earlier (emotion-vector ← curriculum training ←
   production-scale model + corpus + efficient kernels) gets one of
   its blockers softened: training-scale efficient kernels become
   more accessible.

**Costs:**

1. **Setup time**: half a day for clean install + validation. Not
   overwhelming.
2. **Maintaining a dual backend**: Sanctuary still uses DirectML
   (`DirectMLAdamW`, the DirectML-aware C++ dispatch). Luthi training
   on WSL2 + ROCm would be a parallel environment, not a replacement.
   The two environments run in different shells, different Python
   installs.
3. **Experimental-tier risk on Triton specifically**. PyTorch + ROCm
   itself is well-supported and the workaround is community-proven.
   Triton on gfx1101 is the part most likely to surprise.
4. **The C++ extension already covers the immediate need.** Sparse PC
   gating works fast on the current setup. The ROCm migration is for
   *future* capability, not present unblock.

## Conclusion

**Feasible. Not officially supported. Workaround-dependent. Worth it
if Phase 4 training is on the near horizon.**

Concretely:

- ✅ PyTorch + ROCm on WSL2 with the 7800 XT works with the
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` workaround. Third-party tested
  April 2026.
- ⚠️ Triton + ROCm on gfx1101 is at the **experimental** tier. AMD
  officially supports gfx950 / gfx1201; gfx1101 is in the experimental
  list. May work; may have rough edges.
- ✅ Setup is bounded: half a day for clean install + smoke test,
  given the documented pin (ROCm 6.1.3, Ubuntu 22.04).
- ❌ Not a fix for the *immediate* sparse-PC-gating problem (already
  solved by `a6b7ab6`). This is forward-looking investment.

**Recommendation:**

If Phase 4 (curriculum training on the 4B model) is the next major
training run, the WSL2 + ROCm setup is worth doing first because:

1. ROCm-accelerated PyTorch on the 7800 XT is likely meaningfully
   faster for training-scale workloads than DirectML, and Phase 4 is
   the project's most compute-intensive step.
2. The Triton kernel can be validated as a parallel track — if it
   works (experimental-tier permitting), the sparse PC path gets
   GPU-native acceleration; if it doesn't, the C++ extension is the
   fallback and nothing's lost.
3. The setup is reversible (`wsl --unregister Ubuntu`) and
   non-destructive to the Windows side.

If Phase 4 is not in the near plan, the C++ extension I shipped today
covers the immediate need and this work can wait.

**Decision belongs to Brian.** This document is for him to read and
decide; I haven't taken any action on the system.

## What I'd do next if Brian says go

1. Spawn a focused setup session: WSL2 install → Ubuntu 22.04 → ROCm
   6.1.3 install with `HSA_OVERRIDE_GFX_VERSION=11.0.0` → PyTorch
   ROCm wheels → small smoke test that a tensor lives on the GPU and
   a matmul produces a sane result.
2. Run the existing Luthi test suite under PyTorch+ROCm. Anything
   that passes there confirms the environment is sound for the
   project's existing tests.
3. Fill in the Triton kernel against the skeleton's a-k structure.
   Run `tests/test_pc_ops_triton.py::test_triton_matches_python` —
   if it passes, the kernel is bit-identical; if it fails strict,
   the kernel math needs work; if it crashes, that's experimental-
   tier reality and we have data.
4. Benchmark Luthi training (small config) on the new path vs
   DirectML to quantify the speedup.

The setup commands themselves require admin/sudo and interactive
responses; that's Brian at the keyboard. I can document the exact
command sequence in advance.

## What's NOT in this document (and shouldn't be)

- I have not run `wsl --install`. The system state is unchanged.
- I have not tested the workaround on Brian's hardware.
- I have not run any benchmark. The "ROCm is faster than DirectML for
  training" claim is community-consensus; concrete numbers for the
  7800 XT + Luthi specifically would come from the post-setup
  benchmark step.

## Artifacts

- **WebSearches:** three queries on 2026-05-23.
- **Sources** (canonical AMD docs + recent third-party guides):
  - [AMD ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
  - [WSL support matrices by ROCm version (AMD Radeon docs)][rocm-wsl-compat]
  - [Install Radeon software for WSL with ROCm (AMD docs)](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html)
  - [WSL How-to guide (AMD Radeon Ryzen docs)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)
  - [PyTorch compatibility (AMD ROCm docs)](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)
  - [GPU hardware specifications (AMD ROCm docs)](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
  - [Install Triton for ROCm (AMD Radeon docs)](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-triton.html)
  - [ROCm/ROCm Issue #3863 — WSL2 + other RDNA3 GPUs feature request][rocm-issue-3863]
  - [ROCm/ROCm Discussion #2599 — RX 7800 XT support](https://github.com/ROCm/ROCm/discussions/2599)
  - [ROCm/ROCm Discussion #3607 — WSL2 + ROCm 6.1 + RX 6800 setup notes](https://github.com/ROCm/ROCm/discussions/3607)
  - [CraftRigs guide — ROCm on WSL2 step-by-step (2026)][craftrigs]
  - [Till Code — AMD ROCm in WSL2 PyTorch limitations](https://tillcode.com/amd-rocm-in-wsl2-pytorch-installation-limitations/)
  - [ROCM HIP AMDGPU WSL2 Setup gist](https://gist.github.com/tonykero/8ceb62868378ee11e36b07f975731d26)
- **Related project files:**
  - `luthi/v2/pc_ops_triton.py` (the skeleton this investigation is in
    service of)
  - `luthi/csrc/pc_ops.cpp` + `luthi/v2/pc_ops.py` (the C++ path that
    closed the immediate gap, commit `a6b7ab6`)
  - `tests/test_pc_ops_triton.py` (the bit-identity contract that
    would gate a real Triton implementation)

[rocm-wsl-compat]: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html
[rocm-issue-3863]: https://github.com/ROCm/ROCm/issues/3863
[craftrigs]: https://craftrigs.com/guides/rocm-on-wsl2-amd-gpu-setup-that-actually-works/
