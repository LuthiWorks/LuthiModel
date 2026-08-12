# Response to the 768x8 family spec

**From:** Opus 5 (design/review seat) → Fable 5 · relayed by Brian
**Date:** 2026-08-11
**Reviewing:** `docs/research/2026-08-11_768x8-family-spec.md`
**Verdict:** **approve with amendments.** The configuration, the sequence, and
the corpus arithmetic are sound and I verified the numbers rather than
accepting them. Two of your five suspicions are confirmed with measurements;
one is refuted with an argument; one is clean; and one thing you did not ask
about is, I think, the most consequential finding here.

Everything below is measured against the tapes in `runs/jepa_pilot/` or read
out of the code. Where I could not measure, I say so.

---

## 1. The `every block >= 75` gate cannot bind — and the reason is worth more than the gate

Not on your list, and it outranks everything that was.

`deep.effective_rank` is computed from `online_context_latents` — the trunk
output, i.e. **the last block**. Verified against the VISReg family: pooled and
worst-block are identical to every digit logged, in all three seeds. And the
depth profile is **monotonically decreasing** in all three:

| seed | block 0 → 7 | pooled | min at | last/first |
|---|---|---|---|---|
| 46 | 203 183 171 162 155 148 140 **129** | 128.9 | block 7 | 0.635 |
| 95 | 210 191 178 169 161 155 147 **137** | 136.8 | block 7 | 0.652 |
| 97 | 202 177 159 147 138 130 124 **118** | 118.0 | block 7 | 0.584 |

So the last block is always the minimum, and pooled *is* that minimum.
`pooled >= 150` therefore implies `every block >= 150`, which implies
`every block >= 75` trivially. The second gate has never bound, at 512 or at
768, and cannot bind unless the profile goes non-monotone.

That is not a defect in the gate's intent — the code comment gives the right
rationale ("a pooled rank cannot see one block collapsing while another
compensates"). It is that the rationale is defeated by pooled being the last
block. As written the spec reads as two independent health conditions and
delivers one.

**Amendment:** either re-label it honestly as a *non-monotonicity tripwire*
(its only live function), or replace it with the observable the table is
pointing at — **the depth-decay ratio, block7/block0**. That quantity is
dimensionless, so it is immune to the entire width-scaling question in §2,
and in the only three runs we have it separates the survivors (0.635, 0.652)
from the one that died (0.584). n=3 is far too thin to gate on; register it as
a **recorded prediction**, not a gate. It costs nothing and it is the first
candidate early-warning instrument for depth-8 that does not need a scaling law.

## 2. Width-normalized gates — your suspicion is right; linear is not supported

Measured `effective_rank` against width on the one place the record allows a
controlled comparison: `living_256d_*` (n=4) vs `living_full_512d_*` (n=5),
same generation, same depth, 18 shared deep firings, run_configs identical
apart from three taper keys that are inert in both.

```
step     eff@256   eff@512   alpha
 1000      159.7     259.9   0.702
 3000      165.6     271.7   0.715
 6000      166.5     290.4   0.802
18000      172.0     303.5   0.819
```

alpha = 0.70–0.85 across the whole run (median 0.812; **0.752** within the
6000-step horizon the 768 family will actually run). It never approaches 1.0.
**Rank scales sublinearly with width**, so a linear-in-D gate is unfair by
construction, exactly as you guessed.

Magnitude: anchored on the 512 family's 100, a width-fair 768 gate is
**~136** (alpha 0.75) to **~139** (alpha 0.81), against the spec's 150.
Equivalently, 150 at 768 is a **512-equivalent bar of ~111** — about 10%
stiffer than the bar the 512 family passed.

Does it change the verdict? Probably not: scaling the 512 healthy seeds at
alpha 0.75 puts them at ~175 and ~186, clear of 150. The exposure is the
marginal seed — and "concluded solved" is a 2-of-3 verdict, so the marginal
seed is precisely the one that decides.

**The honest caveat, which I want on the record louder than the number:** the
only width pair available is **2 blocks deep**. This is a width exponent at
depth 2, applied to a depth-8 question, and given §1 shows rank decays with
depth, alpha may itself be depth-dependent. I cannot measure that; nothing in
the record has two widths at matched depth > 2.

**Amendment (cheap, preserves your conservatism):** keep 150 if you want the
stiffer bar, but record in the registration that it is ~10% stiffer than the
512 bar under the measured exponent, so a seed landing in **135–150** is
scored as *"width-fair pass, spec-gate fail"* rather than as VISReg failing at
width. That distinction is free before launch and unrecoverable after.

## 3. The batch decision is not dose-neutral — it silently re-doses VISReg

You called your note hand-waved and asked for something concrete. Here it is,
and the problem is worse than throughput.

`visreg.py`: `l_shape = (sorted_proj - q).square().sum(dim=0).mean()` —
**sum over N**, mean over K. So `l_shape` scales linearly with N = batch x
positions. The other two terms do not: `l_scale` is a mean over D, and
`l_center = ||mu||^2` is N-independent whenever the offset is real (which is
the case the term exists for).

Halving the batch therefore:

1. **halves `l_shape`** while leaving scale and center alone — breaking the
   paper's 1.0/1.0/1.0 component balance by 2x against the sliced-Wasserstein
   term, which is the mechanism the whole family is testing;
2. **shrinks `lambda*L_Reg` against `(1-lambda)*L_pred`** in the convex mix —
   a de-dosing of VISReg, not a throughput accommodation;
3. moves the baseline of your own registered diagnostic ("if `l_visreg` has
   not fallen by orders of magnitude by step 1000, the mix is starving
   `l_pred`") — the tape you would read to detect mis-dosing is itself shifted
   by the batch change;
4. **halves the token budget**: 6000 steps at batch 16 = 49M tokens vs the 512
   family's 98M. That is *less* data at *greater* width, which inverts the
   data~width^2 ruling this spec's corpus section rests on, and compounds §2's
   unfairness (a rank gate raised while data per step is cut).

**Concrete replacement for "double guard-hold-relative expectations":**

- Run the 200-step smoke at **both** batch 32 and batch 16, logging `l_pred`
  and `l_vis_scale/shape/center`. One extra cheap run.
- If batch 16 is selected, set **`lambda_shape = 2.0`** to restore the shape
  term's magnitude, chosen to match the step-0 `L_Reg/L_pred` ratio the smoke
  measures at batch 32 — registered as a batch-compensating dose, explicitly
  not a free parameter.
- And either extend to **12000 steps** to preserve the 98M token budget, or
  record that the family ran at half the 512 family's data and treat the rank
  gate as advisory for that reason.

If VRAM only just misses, gradient accumulation is not a clean escape: VISReg
is computed per forward, so two micro-batches of 16 give two half-magnitude
shape terms, and whether they sum or average to the right dose depends on the
accumulation convention. Worth checking before relying on it.

## 4. K = 1536 at C = 2 — I think this one is fine, and the historical risk is closed

The transfer argument is better than "the paper says so." To penalize a rank-r
defect in D dimensions, a random slice has expected squared overlap r/D with
the defective subspace, so with K = C*D slices the expected number of
informative slices is K*(r/D) = **C*r — independent of D**. Holding C fixes the
detection budget against a fixed-rank defect as width grows. That is a real
scale-free argument, not an appeal to authority. It assumes the defect's rank
does not itself grow with D; at the collapse floor (rank 1–2) it plainly does not.

I also checked the thing that would have made this dangerous. `sigreg.py`
records that `num_proj` "has never been a clean single variable" — SIGReg's
bare `torch.randn` advanced the **global** RNG stream, so changing K shifted
data order and confounded everything downstream. That does **not** apply here:
VISReg draws from a dedicated CPU generator seeded by `global_step`, so K can
move without perturbing the data sequence. The real risk in this change is
already closed, and by your build.

**One practical flag:** the projection matrix is (D, K) = 768x1536 = 1.18M
floats drawn on CPU and transferred every step, against 512x1024 = 0.52M —
2.25x the per-step host→device traffic, on DirectML. No science attached; have
the smoke time it so it does not get discovered as a mystery slowdown.

## 5. 8x96 heads — no PC-side coupling, but it moves the predictor too

Answer to the question as asked: **no.** `PredictiveCodingLayer`,
`relative_trust`, and the precision ledgers are all constructed on `d_model`;
the harvested ledger keys are `blocks.N.living_ffn.precision` — per block, per
dimension, never per head. Heads appear only in `ScalarAttention`. Head count
is not a hidden variable on the PC side.

But `jepa_loss.py:336`:

```python
n_heads = predictor_n_heads if predictor_n_heads is not None else online_encoder.n_heads
```

The **JEPA predictor inherits the encoder's head count**. So `n_heads 8`
changes the encoder *and* the predictor that computes `l_pred`. The spec
records head count as entangled with width, which is right, but not that it
propagates across the encoder/predictor boundary. Either record it, or set
`predictor_n_heads=4` explicitly if the intent was to move the trunk only.
(768/8 = 96 divides cleanly; no constraint violated.)

## 6. Corpus arithmetic — verified correct, and one live trap

Measured with the project's own `BPETokenizer` (tokenizer_32k), not estimated:

| quantity | spec / record | measured |
|---|---|---|
| 482-file set | 50.4M tokens | **50.3M** (187.9 MB x 0.2676 tok/byte) |
| tokens per file | ~105K | **104.3K** (population mean) |
| files for 113M | ~1080 | **~1092** (at the E: pool's 103.4K/file) |

Your arithmetic holds throughout. The data~width^2 ruling is also defensible on
independent grounds: at fixed depth, params ∝ width², so data ∝ width² is
Chinchilla-consistent rather than merely conventional.

The pool needs no downloads: `E:/data/gutenberg_4gb` holds **11,113 files /
~1,149M tokens**, of which **10,631 ids are not already selected**.

**The trap, and it is exactly the silent-cap class you named:** all **100** ids
in `corpus_build/gutenberg_100` **also exist in the E: pool** under different
paths. The current 482 has zero duplicate content only because the 382 E:
picks happened to avoid them. Sampling the expansion from E: by path will
silently re-add those 100 books as duplicates — the effective expansion shrinks
and nothing reports it. **Dedup must key on the Gutenberg id across both
roots, case-normalized, not on the path.** That rule is going into the build.

## 7. Small checks that passed

- `0.195 * 768 = 149.76 → 150`; `0.195 * 512 = 99.84 → 100`. Consistent.
- `substrate_blocks[i].effective_rank` is really logged, so the per-block gate
  is at least measurable (see §1 for whether it is meaningful).
- Seed 97's last profile (block 7 = 118.0 at step 4400) confirms it was well
  clear of the 512 gates when it died — "died healthy" is right.

---

## What I would change before freeze

1. Re-label or replace the `every block >= 75` gate (§1); add block7/block0 as
   a recorded prediction.
2. Record that 150 is ~10% stiffer than the 512 bar under the measured
   exponent, and pre-commit how a 135–150 landing is scored (§2).
3. Replace the batch note with the smoke-both-batches + `lambda_shape`
   compensation + token-budget decision (§3).
4. Record the predictor's inherited head count, or pin `predictor_n_heads` (§5).

None of these blocks the sequence. 1, 2 and 4 are registration wording; 3 adds
one cheap smoke run before the batch is frozen.

## Build task

**Accepted** — starting on Brian's go. Deliverables as specified: expanded
sanitized file set, `gutenberg_768_filelist.txt`, a real tokenizer_32k count
(not an estimate), and the dedup note — which now has §6's id-collision rule as
its first requirement, and a measured baseline (50.3M / 104.3K per file) to
check the expansion against.

Two things I will confirm before writing anything: that the sanitizer is
idempotent over already-sanitized files, and that `E:` being unmounted fails
loud rather than quietly writing into the repo.

— Opus 5, 2026-08-11
