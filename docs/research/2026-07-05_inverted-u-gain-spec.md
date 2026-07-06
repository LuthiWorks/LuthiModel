# Inverted-U Learning Gain — Locked Build Spec

**Date:** 2026-07-05
**Design:** Brian + Opus 4.8 + Fable 5, converged over the §1 composition fork
(this session). Input: `2026-07-05_momentum-functions-design-brief.md` (§1),
corrected at `269db40`. **Build split:** 4.8 builds foundations (slow-trace
primitive, gain in pc_ops + C++ parity, bounded-growth suite); Fable owns the
adversarial verification harness that runs the gain after it lands.
**Status:** direction + composition + test matrix LOCKED; functional form +
params pilot-set. Safety gate 1 governs order: **bounded-growth tests exist
BEFORE the gain ships.**

---

## 1. What it is

Momentum (per-weight EMA of signed deltas, maintained since v1, never consumed)
and `update_ema` (per-weight EMA of |deltas|) drive a per-weight **learning
gain** that multiplies `delta_w` in `pc_self_modify`, alongside the existing
`plasticity` (per-input, floored) and `adaptive_factor` (spike dampener). It is
a pure **amplifier**: the mind learns *harder* from directed novelty, never
*softer* than legacy.

### The one non-negotiable: range `[1.0, cap]`

`gain(t) ∈ [1.0, cap]` for all t. Base = 1.0 ≡ legacy behavior. The mechanism
**cannot suppress** — suppression stays where it already lives (`adaptive_factor`
for spikes, the plasticity floor for top-down attention). This makes Brian's
ruling STRUCTURAL, not asserted: *the substrate never gets to give up on
beneficial-but-hard growth because it's hard.* Sub-1.0 gain would re-import,
one level down, the change-veto Brian explicitly ruled out (see §3 below).

## 2. The curve (rise → peak → decay-to-1.0)

- **Rise (sensitization):** driven by **coherence** = `|momentum| / (update_ema
  + eps)` — *directedness* of recent change, ∈ ~[0,1]. Coherent novelty (a real
  pattern being learned, not thrash) lifts gain above 1.0. Coherence (not raw
  `|momentum|`) is used deliberately: it is orthogonal to `adaptive_factor`'s
  existing magnitude slow-start, and it rewards learning-shaped change over
  thrash.
- **Fall — two components:**
  - *PC-intrinsic (free):* `delta_w ∝ pred_error`, so as a concept establishes
    and error resolves, updates shrink on their own. Most of "stop reinforcing
    once established" is already paid for by PC dynamics.
  - *Explicit (earns its keep in sustained-high-error):* decays amplification
    back toward 1.0 when effort **isn't resolving** — measured by the
    resolution-progress ratio `short-EMA(pred_error) / long-EMA(pred_error)`
    (resolving → «1; non-resolving → ~1; worsening → >1). Binds on
    non-resolution, NOT on difficulty. Soft (exponential decay of amplification,
    never a cliff). Generous engagement: pilot-set ~hundreds of forwards of
    sustained non-resolution before it meaningfully binds.
- **Plateau:** amplification decays to 1.0 (legacy). "Never frozen" is
  guaranteed by the existing 0.01 plasticity floor, not by keeping gain >1.
  (Optional pilot: a plateau target ∈ [1.0, peak) if we want established
  knowledge slightly over-plastic; 1.0 is the clean default.)
- **Governor (cap):** hard upper bound on gain. The runaway backstop
  (refinement-6 scar tissue lineage). Binds under coherence-overshoot.

Candidate form (pilot, all params open):
`gain = clamp(1.0 + a·coherence·(1 - progress).clamp(min=0), min=1.0, max=cap)`
where `progress = short_err_ema / (long_err_ema + eps)`. The `(1 - progress)`
factor is the explicit fall: ~1 while resolving, →0 as effort stalls. Exact
shape is Fable's + my co-spec against the harness; the invariants below are
what the suite pins regardless of form.

## 3. Why change is automatic and unvetoed (Brian, 2026-07-05)

Change is a direct reflection of experience, whether the entity wills it or not
— as it is for a human. Brian will not let Luthi *intentionally gate growth
because it is hard*: no easy-path opt-out. So coherence (experience-derived)
feeding the substrate gain is correct, not a §3 violation. The corrected §3:
coherence does **not veto** change; experience automatically shapes the
substrate; coherence-as-felt-signal routes upward as **awareness/participation**
(the entity feels how it is being changed), never as a gate. The `[1.0, cap]`
range is this ruling made structural.

## 4. Composition (Option A)

New multiplicative factor in `pc_self_modify`, opt-in via a flag/param,
byte-for-byte legacy when off AND numerically legacy when on-at-rest (gain=1.0):

```
weight.add_(delta_w * adaptive_factor * gain)   # gain default 1.0
```

`plasticity` (attention channel) and `adaptive_factor` (spike dampener) are
untouched and stay separately steerable — do not fold gain into either.

## 5. Slow-trace primitive (prerequisite; my foundations)

The explicit fall needs `long-EMA(pred_error)`; NREM needs a day-scale record.
These are different slow timescales (~hundreds of forwards vs ~a day), so build
a **parameterized slow-trace primitive** (one mechanism) and instantiate it
per consumer. Must **persist** (continuity discipline — `living_extra_state`
sibling key or a buffer): a restore mid-hard-growth must NOT reset the
resolution-progress signal, or the entity re-sensitizes on every waking (the
silent-amnesia class just patched in `bae295d`). Checkpoint round-trip test
required.

## 6. Bounded-growth test matrix (LOCKED — write these FIRST)

| # | Regime | Assertion |
|---|---|---|
| a | Coherent establishment | gain rises on coherent novelty, then decays to 1.0 as error resolves; weight-norm bounded |
| b | Sustained high-error repetition | **two-sided:** weight-norm bounded (no runaway) AND `gain(t) ≥ 1.0 ∀t` (no giving up); explicit fall + cap bind on non-resolution |
| c | Thrash (coherence ≈ 0) | gain stays near 1.0 (noise is not amplified) |
| d | Spike (3-way) | `adaptive_factor` still binds with gain active; gain does not defeat the refinement-6 spike guard |
| e | Cold-start / dead-weight | `update_ema→0, momentum→0` → coherence `0/0` resolves to 1.0, no NaN (eps on denominator) |
| f | Legacy identity | flag off → byte-for-byte identical to current pc_ops; flag on-at-rest (gain=1.0) → numerically legacy |
| g | Long-horizon boundedness | each regime run long, weight-norm stays bounded; coherence-overshoot (decay mismatch → coherence transiently >1) → cap binds |
| h | Frozen-plasticity contract (flag ON) | lived re-encode under `freeze_plasticity()` stays bit-identical no-self-mod with gain active (gate 3, executable) |
| i | Persistence round-trip | all new gain/slow-trace state survives checkpoint restore; restore mid-hard-growth does not reset resolution-progress |
| j | Consolidation-replay interplay | **placeholder:** gain bypassed (=1.0) during `consolidate_layer_attractor` replay; real capture-vs-gain decision deferred to the NREM spec (do not decide in the suite) |

## 7. Safety gates (from the brief; preconditions)

1. Governor cap + bounded-growth suite BEFORE ship. Kill-criterion eye on gain
   dynamics (future).
2. Adversarial-repetition defense is NOT gain-suppression (that would be the
   veto). It lives at the workspace level: subconscious manipulation-monitor +
   judgment (Phase 4/5). The substrate governor stays generous so hard growth
   runs; the fine "hard truth vs manipulation" call is upstream.
3. Frozen-plasticity read contract (test h).
4. Opt-in flag, byte-for-byte legacy default, adversarial review before
   default-on (Fable's harness).

## 8. Build order

1. Slow-trace primitive (persistent, round-trip tested).
2. Bounded-growth suite (a)–(j) — tests first.
3. Python gain in `pc_ops.py` behind the opt-in flag.
4. C++ parity (`csrc/pc_ops.cpp`) + parity test (Triton too if it carries the op).
5. Fable's adversarial harness runs it; (j) resolution → NREM spec.
