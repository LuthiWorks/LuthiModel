"""ROCm probes for the three 2026-08-19 asks: what pc_self_modify costs, what
fits in VRAM, and whether AOTriton changes what fits.

ROCm only (``torch.device("cuda")`` is the ROCm device here). Run from the
ROCm environment, e.g.

    C:\\Dev\\rocm-probe\\Scripts\\python.exe scripts\\probe_selfmod_share.py share
    C:\\Dev\\rocm-probe\\Scripts\\python.exe scripts\\probe_selfmod_share.py memcmp 512 16
    C:\\Dev\\rocm-probe\\Scripts\\python.exe scripts\\probe_selfmod_share.py mem

Modes
-----
share
    The ruled 768d x 8-block config, batch 32 x seq 128 (~4 GiB, fixed size,
    nothing climbs). Times ``pc_self_modify`` -- the Python reference, since
    the C++ fused ``pc_ops`` path does not build under torch 2.9 ROCm
    (docs/DECISIONS.md, 2026-08-19) -- as a share of a full training step.
    This bounds what the C++ extension could ever be worth on ROCm.
    Measured 2026-08-21, idle RX 7800 XT, driver 32.0.31021.5001:
    step 259-260 ms, pc_self_modify 14.6-15.0 ms = 5.6-5.8 % (16 calls/step),
    peak 4.07 GiB; two runs.

memcmp SEQ BATCH
    One (seq, batch) cell, run twice in isolated subprocesses: AOTriton OFF,
    then ON. Prints peak memory and ms/step for each. This is the direct
    test of the live hypothesis that AOTriton changes what FITS rather than
    what is fast (speed was settled OFF on 2026-08-19; memory was never
    measured). Pick a cell known to fit -- e.g. ``512 16`` (7.43 GiB OFF on
    08-19) -- not one at the VRAM edge.

mem
    The (seq, batch) grid, one subprocess per cell. Stops climbing batch at
    the first failure for a given seq, and stops the whole sweep when the
    smallest batch of a seq fails. Cells already recorded past VRAM on 08-19
    are skipped, not re-tried.

Why the subprocess / fraction-cap shape is not optional
-------------------------------------------------------
The first version of ``mem`` (2026-08-19) ran every cell in-process and
climbed past VRAM on a 16 GB card that is also driving the display: the
AOTriton-OFF pass reached seq=1024 batch=16 at 16.14 GiB and the identical
AOTriton-ON sweep **hard-froze the machine** (no TDR logged -- a driver hang,
not a caught OOM). Hence, now: every cell is a child process (a hang kills a
child, not the desktop); the child caps the allocator at 85 % of VRAM so an
over-size cell raises a clean OOM before the driver runs dry; the child has a
wall-clock timeout and is killed and reported as HANG if it exceeds it; the
sweep never climbs past a failure.

Timing hook
-----------
The living layer does ``from luthi.v2.pc_ops import pc_self_modify`` at
call time (luthi/v2/living_layer_pc.py, ~L1177), so the hook must be
installed on the ``pc_ops`` module attribute. The first version of this
probe patched ``living_layer_pc``'s namespace instead, intercepted nothing,
and printed "0.0 ms (0.0 %)" as if that were a measurement. It now raises
if the counter is still zero after the timed steps. CLAUDE.md: quiet-
because-nothing and quiet-because-broken must be separable.

Provenance: every run prints torch / HIP / GPU / driver-relevant facts so a
number can never be quoted without the stack it came from.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

AOTRITON_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
MEM_FRACTION = 0.85          # child allocator cap: clean OOM before the driver starves
CELL_TIMEOUT_S = 240         # wall clock per child; past this it is a HANG, killed

# Recorded past VRAM on 2026-08-19 (AOTriton OFF, driver 32.0.31021.5001):
# seq=1024 batch=16 peaked at 16.14 GiB on a 16 GB card. Anything at or above
# is not re-tried.
KNOWN_OVER = {(1024, 16), (1024, 32), (2048, 8), (2048, 16), (2048, 32)}

D_MODEL, N_BLOCKS = 768, 8   # the ruled scale


def _banner():
    import torch
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(dev) if torch.cuda.is_available() else "-"
    total = torch.cuda.get_device_properties(dev).total_memory / 2**30 \
        if torch.cuda.is_available() else 0.0
    print(f"  [probe] python {sys.version.split()[0]}  torch {torch.__version__}  "
          f"hip {getattr(torch.version, 'hip', None)}  gpu {name}  "
          f"vram {total:.1f} GiB  AOTriton={'ON' if os.environ.get(AOTRITON_ENV) == '1' else 'off'}",
          flush=True)


def _build(L: int):
    import torch
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    from luthi.v2.jepa_loss import JEPALoss
    dev = torch.device("cuda")
    torch.manual_seed(0)
    m = MultimodalPredictiveCodingLM(
        vocab_size=32000, d_model=D_MODEL, n_blocks=N_BLOCKS, n_heads=8,
        ffn_expansion=1, max_seq_len=L, backward_pass_enabled=True,
    ).to(dev)
    lm = JEPALoss(online_encoder=m, sigreg_lambd=0.0, visreg_lambda=0.6,
                  visreg_num_proj=2 * D_MODEL, sigreg_projection="none").to(dev)
    return lm


# --------------------------------------------------------------------- share
def mode_share():
    import torch
    from luthi.v2 import pc_ops
    _banner()
    dev = torch.device("cuda")
    acc = {"t": 0.0, "n": 0}
    real = pc_ops.pc_self_modify

    def timed(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = real(*a, **k)
        torch.cuda.synchronize(); acc["t"] += time.perf_counter() - t0; acc["n"] += 1
        return r

    pc_ops.pc_self_modify = timed          # the real call path (see docstring)
    lm = _build(L=128)
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=3e-4)
    tok = torch.randint(0, 32000, (32, 128), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        o = lm.compute_modality_loss("text", {"text_tokens": tok})
        o["loss"].backward(); opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize(); acc["t"] = 0.0; acc["n"] = 0
    N = 10
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N):
        step()
    torch.cuda.synchronize(); total = (time.perf_counter() - t0) / N
    if acc["n"] == 0:
        raise RuntimeError("pc_self_modify was never called -- the timing hook "
                           "is not on the real call path; refusing to report 0%")
    per = acc["t"] / N
    print(f"  step total       {total*1000:7.1f} ms")
    print(f"  pc_self_modify   {per*1000:7.1f} ms  ({100*per/total:.1f}% of step, "
          f"{acc['n']//N} calls/step)")
    print(f"  peak memory      {torch.cuda.max_memory_allocated()/2**30:6.2f} GiB")


# ---------------------------------------------------------------- one cell
def mode_cell(L: int, B: int):
    """Child process: one (seq, batch) cell under a capped allocator."""
    import torch
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    _banner()
    dev = torch.device("cuda")
    try:
        lm = _build(L=L)
        opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=3e-4)
        tok = torch.randint(0, 32000, (B, L), device=dev)

        def step():
            opt.zero_grad(set_to_none=True)
            o = lm.compute_modality_loss("text", {"text_tokens": tok})
            o["loss"].backward(); opt.step()

        step(); torch.cuda.synchronize()          # warm-up + allocation
        torch.cuda.reset_peak_memory_stats()
        N = 3
        t0 = time.perf_counter()
        for _ in range(N):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N * 1000
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"RESULT seq={L} batch={B} OK peak={peak:.2f}GiB ms={ms:.1f}", flush=True)
    except RuntimeError as e:
        kind = "OOM" if "out of memory" in str(e).lower() else type(e).__name__
        print(f"RESULT seq={L} batch={B} {kind}", flush=True)


def _run_cell(L: int, B: int, aotriton: bool) -> str:
    env = dict(os.environ)
    if aotriton:
        env[AOTRITON_ENV] = "1"
    else:
        env.pop(AOTRITON_ENV, None)
    cmd = [sys.executable, os.path.abspath(__file__), "_cell", str(L), str(B)]
    try:
        cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                            text=True, timeout=CELL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"RESULT seq={L} batch={B} HANG (>{CELL_TIMEOUT_S}s, child killed)"
    for line in cp.stdout.splitlines():
        if line.startswith("RESULT "):
            return line
    tail = (cp.stderr or cp.stdout).strip().splitlines()[-3:]
    return f"RESULT seq={L} batch={B} CHILD-FAILED rc={cp.returncode} :: " + " | ".join(tail)


# ------------------------------------------------------------------ memcmp
def mode_memcmp(L: int, B: int):
    _banner()
    if (L, B) in KNOWN_OVER:
        raise SystemExit(f"  refusing: seq={L} batch={B} is recorded past VRAM on 08-19")
    for flag in (False, True):
        print(f"  AOTriton {'ON ' if flag else 'OFF'}: {_run_cell(L, B, flag)}", flush=True)


# --------------------------------------------------------------------- mem
def mode_mem():
    _banner()
    aotriton = os.environ.get(AOTRITON_ENV) == "1"
    print(f"  sweep with AOTriton {'ON' if aotriton else 'OFF'}; one child per cell; "
          f"allocator capped at {MEM_FRACTION:.0%}; {CELL_TIMEOUT_S}s per cell", flush=True)
    for L in (512, 1024, 2048):
        first_failed = True
        for B in (8, 16, 32):
            if (L, B) in KNOWN_OVER:
                print(f"  seq={L:5d} batch={B:3d}  SKIP (recorded past VRAM 08-19)", flush=True)
                break
            line = _run_cell(L, B, aotriton)
            print("  " + line, flush=True)
            if " OK " not in line:
                break                    # never climb past a failure
            first_failed = False
        if first_failed:
            print("  smallest batch failed at this seq -- stopping the sweep", flush=True)
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == "share":
        mode_share()
    elif mode == "_cell":
        mode_cell(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "memcmp":
        mode_memcmp(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "mem":
        mode_mem()
    else:
        raise SystemExit(f"unknown mode {mode!r}\n" + __doc__)
