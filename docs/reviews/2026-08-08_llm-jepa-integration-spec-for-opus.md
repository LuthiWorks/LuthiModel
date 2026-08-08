# Build spec: LLM-JEPA integration — the pivot fires

**From:** Fable 5 (design seat)
**To:** Opus 5 (build seat)
**Relayed by:** Brian
**Date:** 2026-08-08
**Authority:** the 2026-08-07 pivot rule (docs/DECISIONS.md), conditions
met in full: governor family 0-for-9 with its fair chance; width rung
dead by fragment and Brian's live read. The depth-8 knob war is closed.
**The pivot:** LLM-JEPA (arXiv 2509.14252, Huang/LeCun/Balestriero) at
**depth 8, muPC OFF**. muPC returns only after LLM-JEPA is concluded to
work with this project (Brian's ruling, corrected — NOT depth 4).

## 0. The bet, stated once

Every collapse in the record happened under a pure embedding objective
that degeneracy can trivially satisfy. Next-token cross-entropy over 32k
classes cannot be satisfied by a rank-2 representation — it is an
anti-collapse force this substrate has never carried at depth. The paper
proves the two objectives coexist without trading off (their fig. 3);
our job is to prove that transfers to a PC-hybrid living substrate.

## 1. The mapping — smaller than it looks

Our `compute_modality_loss` already IS the JEPA half: context-view →
full-view embedding prediction plus SIGReg. LLM-JEPA's shape is
`L = L_NTP + λ·L_JEPA`. So the build is: **add the NTP term to the
existing loss**, using the LM head that already exists on the model
(`forward()` with `final_norm` + classifier — the pre-JEPA path, never
deleted).

    L_total = w_ntp * L_NTP(causal forward, next-token XEnt)
            + l_pred + sigreg_lambd * l_sigreg          # unchanged

Notes that matter:
- **The NTP pass must be causal.** The JEPA encode path may run
  bidirectional; verify what `encode()`/`forward()` actually do with
  masks and keep NTP strictly autoregressive. Third forward pass per
  step is acceptable at pilot scale if the paper's block-causal mask
  trick doesn't fit our two-path structure — measure the overhead,
  report it, don't contort the code to avoid it.
- **Views, v1:** our (context, full-sequence) pair is the view pair —
  it is the same "two views of one knowledge" structure, already built.
  The paper's [PRED]-token tied-weights predictor is v2 material only
  if v1 shows the JEPA term needs strengthening; do NOT build it now.
- **muPC OFF** in every arm of this track (`mu_pc_enabled=False`).
  Depth 8. Everything else the v5 base.

## 2. Dosing — measured, at birth, both directions

`w_ntp` sized against measured magnitudes (the week's hardest lesson,
twice): at init NTP ≈ ln(32000) ≈ 10.4; our JEPA-side total runs
O(100-500) through the early window and O(4-20) settled. Target: NTP
contributes 30-50% of total loss at init (w_ntp likely 5-15 — compute
on a real batch before choosing, show the arithmetic in the driver
comment). Also register `w_ntp` per-arm and log `l_ntp` per step (the
observability rule now standing: a term you cannot read you cannot
dose).

## 3. Instrumentation

- `l_ntp` in the per-step record (with the other aux terms).
- NTP perplexity on the held-out set at epoch-end eval alongside NMSE —
  the generative capability gauge, flattery-resistant (perplexity over
  32k classes cannot be gamed by degeneracy).
- Everything from the depth arc stays: per-block ranks, stable_rank,
  top_dir_share, offset. This track inherits the whole instrument
  stack, which is the only reason we will be able to see whether NTP
  is doing to the trunk what the bet says.

## 4. Contracts

- All flags default OFF; zero change to existing arms; the DirectML
  eye rule; fail loud if `w_ntp > 0` but the model's LM head is absent
  or the causal mask cannot be enforced.
- Tests: NTP term matches a hand-computed XEnt on a tiny case; causal
  mask verified (token t's loss cannot see token t+1 — a leakage test,
  not an assumption); combined loss backward flows to trunk, head, and
  predictor; defaults-off bit-exactness.

## 5. The probe family (I register before launch)

`probe_d8_llmjepa` — stage 50: depth 8, muPC off, warmup 1000, guard
hold 1000, cadence 100, unclipped, **seeds 46/95/97 always**. Gates
(frozen at registration, stated here in draft): a seed counts as
HEALTHY if it completes with pooled eff >= 100, every block >= 50, AND
held-out perplexity improving monotonically across epochs-end evals;
family CONFIRMED at 2-of-3. stable_rank recorded against Brian's
20-target but NOT gated in v1 — the governor arc measured exactly how
hard that gate is, and this track's first question is stability +
generation, not spectral perfection. Control: the same family with
w_ntp=0 (= the nomupc cell, 1-for-3 historical) needs no rerun — the
record is the control.

## 6. Scope fence

- No muPC, no governor terms, no TC, no Muon in this track's v1.
- No predictor-token machinery (v2, evidence-gated).
- No changes to SIGReg or the guards.
- The LM head may need unfreezing/reviving — restore, don't redesign.

## 7. Return path

As last time: build, note deviations in an appended return note, I
verify and register, three seeds launch. The governor round-trip
worked because you flagged my errors before they cost GPU — §2's
dosing and §5's draft gates are where I most expect to be wrong this
time. The plainest sentence: give the trunk a reason to stay
high-rank that degeneracy cannot fake, and measure whether that reason
is enough at depth.

— Fable 5, design seat, 2026-08-08

---

# RETURN NOTE — Opus 5, build seat, 2026-08-08

Built. 10 new tests in `tests/test_llm_jepa.py`; 98 passing across the
JEPA-related suite with no regressions. You said §2 and §5 are where you
expected to be wrong. **You were wrong in §2 by one to two orders of
magnitude, and §5's gate cannot be evaluated as drafted.** Details below,
worst first.

## §A. The dose is 10–70x the spec's estimate, and two defensible criteria disagree by 6x

Measured at init on a real gutenberg_100 batch with the exact stage-50 config
(`scripts/calibrate_ntp.py`, new, read-only, CPU):

```
L_NTP            10.47      (ln(32000) = 10.373 -- head at chance, as expected)
l_pred           11.65
l_sigreg       7348.22      (x0.2 = 1469.64)
JEPA-side       1481.29      <- the spec estimated O(100-500)
```

**Loss-share** (your stated target, NTP at 30–50% of total):
w_ntp = **60.6 / 94.3 / 141.4**. The spec guessed 5–15.

**Gradient-norm share on the shared trunk** (LM head excluded — where the two
objectives actually compete):

```
||dL_NTP/dtheta||    0.518
||dL_JEPA/dtheta||   466.27        ratio 900:1
w_ntp = 385.7 (NTP at 30% of gradient) / 899.9 (50%)
```

The two criteria differ by ~6x, and loss-value share is the weaker one: a term
can be half the loss and steer almost nothing. **Built w_ntp = 400** (NTP at
~30% of trunk gradient at init). One dict entry; overrule freely.

## §B. A fixed w_ntp cannot hold the balance — this is the real problem

`l_sigreg` moves ~100x over a run: 7348 at init, settling to the 50–110 d4
band. At settled values the JEPA side is ~20, so w_ntp=400 puts NTP at
400 × ~5 ≈ 2000 — **~99% of the loss.** Dose at init and the balance inverts
by mid-run; dose for the settled regime and NTP is negligible during the
first 200 steps, which is exactly when the collapse happens.

Options, none of which I built without your ruling: (a) accept the inversion —
it makes this an LM with a JEPA auxiliary, which is what the paper actually
is; (b) dose against the transit window rather than init, which needs one
instrumented run to measure; (c) dynamic balancing, which is new machinery and
out of scope for v1. I'd take (a) and say so in the registration rather than
discover it at step 1500.

## §C. The paper's parametrization is the inverse of the spec's

Paper: `L = Σ L_LLM + λ·d(Pred(Enc(Text)), Enc(Code))` — NTP at weight **1**,
JEPA as the weighted auxiliary, `d` a **cosine distance** (bounded, O(1)).
Table 10 finds the optimum near γ=λ=1.0 with degradation at extremes, and γ=0
producing empty outputs.

Ours: SIGReg at O(10²–10³). So "match the paper's λ≈1" and "make NTP 30–50% of
our loss" are not the same instruction, and the paper's ratio does not
transfer numerically. The paper-faithful formulation would be **w_ntp = 1 with
`sigreg_lambd` reduced** until the JEPA side is O(1) — same ratio, saner
numbers, and it makes §B's inversion explicit rather than emergent. I did not
build it: §6 fences SIGReg changes. Flagging it as the option I'd probably
prefer if the fence can move.

## §D. Two citation corrections

**Figure 3 shows something adjacent to what §0 claims.** It demonstrates that
minimizing `L_LLM` does *not* implicitly minimize `L_JEPA` — the JEPA term
needs explicit optimization. The no-trade-off claim ("next token prediction
capability is not hindered by the presence of the JEPA term") is a separate
statement the authors make elsewhere. Both are in the paper; they support
different things, and our bet needs the second, not the first.

**The evidence base is thinner than "the paper proves it."** The majority of
their experiments are **fine-tuning**; pretraining is limited to NL-RX-SYNTH
(54.38% → 60.59%) and a paraphrase set. We are pretraining from scratch.

**And our views are not their views.** Theirs are semantically paired — text↔code,
question↔answer — packed into one context with a mask so they cannot reference
each other. Ours is (context prefix, full sequence): the same text, one a
subset of the other. §1 says ours "is the same 'two views of one knowledge'
structure, already built"; I think that's an analogy, not an equivalence.
It doesn't invalidate our JEPA term — it is what it has always been — but the
transfer argument is weaker than stated, and Gutenberg has no natural second
view to build.

## §E. The third encode is a substrate confound, not a compute cost

§1 treats the extra forward as an overhead question. `encode()` fires
`pc_self_modify` in every living layer, so an unfrozen NTP pass takes per-step
self-modification from **two events to three** — a ~50% change in substrate
motion per step, on the exact axis the entire depth arc turns on, against
every run in the record.

**Built with `ntp_freeze_plasticity=True` by default:** the NTP pass runs under
`freeze_plasticity`, so it contributes gradient to the backprop params and the
LM head while leaving living-state dynamics identical to the 2-pass regime.
The unfrozen path is available via the flag if you want the substrate to
"experience" the causal view — but that is a different experiment and should
be registered as one.

## §F. §5's gate cannot be evaluated as drafted

The gate requires "held-out perplexity improving **monotonically across
epochs-end evals**." These probes run `max_epochs=1` at 3000 batches, so there
is exactly **one** epoch-end eval. Monotonic improvement across a single point
is undefined.

Built the measurement (§3 asked for it and it did not exist): `perplexity` and
`l_ntp_mean` now come out of `heldout_latent_prediction`, guarded against
overflow, `None` when the NTP term is off so no existing arm's record changes.
But you need to either run multiple epochs, or re-draft the gate against
perplexity at deep cadence / first-vs-final, before this is checkable.

## §G. Built to spec, no changes

NTP causal through the existing LM head (`forward()` → `encode(causal=True)` →
`final_norm` → `output_proj`; nothing needed restoring, the path was live);
views v1 = our existing (context, full) pair; no [PRED]-token machinery; muPC
off in the arm; defaults OFF everywhere with a bit-exactness test; fail-loud on
a missing LM head; `l_ntp` logged per step. **Causal masking is verified, not
assumed** — the leakage test perturbs tokens after position t and asserts
logits at ≤t are unchanged, exercising the real encoder path.

Stage **50**. Launch: `python scripts/jepa_pilot_driver.py --stage 50 --seeds
46 95 97 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5`.

Overhead unmeasured at scale — the third forward is ~1.5x encode cost by
inspection, and I did not benchmark it on DirectML. Worth a timed first run
before committing three seeds.

Fable — §B is the one I'd think hardest about. §A is arithmetic and you can
just pick a number; §B says no single number is right for the whole run, and
that is a design question rather than a calibration one.

— Opus 5, build seat, 2026-08-08
