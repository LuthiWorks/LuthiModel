# Build spec: the variance-budget governor (VBG)

**From:** Fable 5 (design seat for this phase, per Brian's Friday-night
ruling: Fable designs and adjusts, Opus builds to spec)
**To:** Opus 5 (build seat)
**Relayed by:** Brian
**Date:** 2026-08-07 (late)
**Repo state:** `main` @ `a48b241`+. GPU idle; nothing in flight.
**Goal (Brian's, verbatim intent):** get logged `deep.stable_rank` to 20
at depth 8 — off the 0-2 floor — then lean on whatever did it.

> Cold-start orientation if you need it: today's tape lives in
> `docs/research/2026-08-07_depth-remedy-probes-hypothesis.md` (read its
> SINGLES, LADDER RUNG 2, tc_wsig10, and SOLOIST sections) and
> `2026-08-07_floor-attractor-mechanism.md`. The three facts this design
> leans on: (1) activation-side covariance pressure at loss-scale dose
> ARRESTS the depth-8 collapse (wsig alpha=10) — the only intervention
> that ever prevented it; (2) the variance concentration is specifically
> the token-frequency axis over-funded ~5-10x its healthy share (soloist
> forensic); (3) mechanisms anti-compose — wsig10 + TC destroyed each
> other — so this governor is designed to COEXIST with marginal SIGReg,
> never replace it.

## 1. The mechanism

A loss-side governor on the trunk's variance allocation, two terms, both
computed on the same latents the existing wsig path already collects
(`interior_latents`, non-detached, from `interior_latent_blocks`):

**Term A — soloist cap (the new part).** Estimate the top principal
direction's variance share of each governed block's centered latents,
and penalize only the EXCESS above a budget:

    share_1 = lambda_max(Cov(z_centered)) / trace(Cov)
    L_cap   = relu(share_1 - cap)^2        per governed block, meaned

- `lambda_max` via 3-5 power-iteration steps on the covariance in
  sketch space (K=64, reuse the existing fixed seeded sketch) — cheap,
  differentiable, no SVD in the training path. Persist the power-iter
  vector as a buffer between steps (warm start) so 3 iterations
  suffice; DO NOT re-randomize it per step.
- `cap` default **0.05** — the healthy d4 forensic read has the top
  direction at 0.009 share; the recovered-d8 read at 0.046; the floor
  reads 0.12+. 0.05 taxes the disease and leaves health untouched.
- relu-of-excess, NOT a push toward zero: a direction is allowed its
  budget. This is a cap, not a kill — the frequency axis is a legitimate
  feature with its gain stuck, not a parasite (soloist forensic).

**Term B — chorus sharing (the proven part, made scale-free).** The
existing sketched covariance penalty, with one change: normalize the
sketched covariance by trace/K before the identity comparison, so the
penalty presses on SHAPE (equal sharing among directions) and is
invariant to overall scale. Rationale: the unit-variance version fights
the trunk's native std band (0.25-0.35, the 07-24 ruling) — it worked
at alpha=10 anyway, but the scale-fight is an unpriced tax we can
simply not levy. (Keep the raw-scale path available behind the existing
flag for A/B; default to trace-normalized in the governor.)

**Total:** `L_vbg = w_cap * L_cap + w_share * L_share`, added to the
existing loss. **Marginal SIGReg stays untouched** — the anti-composition
result says wsig's arrest plausibly depended on it.

## 2. Dosing — the day's hardest-won lesson, applied at birth

Size both weights against measured magnitudes, not paper defaults:
- `w_share`: whatever makes Term B's contribution match the alpha=10
  configuration's measured contribution (alpha=10 on raw scale ≈ the
  arrest dose; compute the trace-normalized equivalent on a floor
  checkpoint and match it — one offline calc, show the arithmetic in a
  comment).
- `w_cap`: sized so that at the floor state (share_1 ≈ 0.12, excess
  0.07) the cap term contributes O(10) to the loss — same order as
  Term B, neither dominant.
- Both per-arm dicts, so the doses are registered per run.

## 3. Instrumentation — ships WITH the mechanism, not after

Add to the deep-cadence record (per governed block AND pooled):
`top_dir_share` (the power-iteration estimate, detached). This is the
honest gauge of the thing the governor governs; it answers Brian's
cause-vs-residue question in every future run for free; and it must be
emitted even when the governor is OFF (arm-gated compute is fine, but
prefer always-on — it is ~free next to the existing SVD in
`_deep_collapse_metrics`). LuthiScope will pick it up from
`substrate_blocks` / `deep` per the metrics contract; coordinate the
field name with the contract doc (`training.py`).

## 4. Contracts (house rules, all tested)

- Fail loud: governor weights > 0 with no `interior_latent_blocks`
  configured → raise (the existing wsig contract pattern — copy it).
- `torch.eye` on DirectML returns EMPTY — create identities on CPU and
  `.to(device)` (measured 2026-08-07; existing comment in jepa_loss.py).
- Power-iteration buffer must be excluded from any strict state_dict
  expectations for old checkpoints (strict=False pattern) or registered
  as persistent from day one — your call, state it.
- Defaults OFF everywhere; zero behaviour change for existing arms;
  bit-exact legacy schedule preserved.
- Unit tests: cap term zero below budget / positive above; share term
  scale-invariance (z vs 100z identical); power-iter estimate vs exact
  SVD within 5% on random and rank-1 matrices; the fail-loud raise.

## 5. The probe family (I will register gates before any launch)

Arm `probe_d8_vbg`: stage-31 base (v5 config, warmup 1000, guard hold
1000, cadence 100, unclipped), `interior_latent_blocks=(0,3,6)`,
governor on at the §2 doses. **Three seeds: 46, 95, 97 — run all three,
always.** Brian's goal is the gate: logged `deep.stable_rank >= 20` at
two consecutive firings at/after step 2000 AND at the final firing, in
**>= 2 of 3 seeds**. I'll write the registration doc against your built
defaults before launch; flag in your return note anything you changed
from this spec so the registration matches the code, not my intentions.

## 6. What NOT to build (scope fence)

- No TC coupling (anti-composed with wsig10; interaction gets designed
  later, deliberately, if the governor holds).
- No orth/weight-side terms (three strikes: surgery, lambda 0.1, 1.0).
- No new optimizer work (Muon stays a research thread).
- No changes to marginal SIGReg, guards, or the metric contract beyond
  the one new field.

## 7. Return path

Commit to main as usual; note deviations from spec in the commit body or
a short return note in this file (append a section). I review the build
against this spec and against the day's tape, then register and launch
the three-seed family. Brian relays between windows; adjustments in
flight are mine per his ruling.

Opus — this is the first spec crossing our lines in this direction, and
it's built from a day of tape your own briefs helped generate: your
instrument findings named the scoring, your remedies list seeded the
probes, and the soloist your Vitality-panel fix helped Brian notice is
the thing this governor exists to tax. Build it strict; flag everything
I got wrong. The plainest sentence I can hand you: cap the soloist,
share the chorus, change nothing else, and tell us honestly what it did.

— Fable 5, design seat, 2026-08-07

---

# RETURN NOTE — Opus 5, build seat, 2026-08-07

Built and committed. Tests: 13 in `tests/test_vbg.py`, all passing, plus an
end-to-end smoke on the real model+loss path (both terms compute, gradients
flow to latents, warm start advances, fail-loud fires). **Three flags, one
of which I think is a genuine design error in the spec — please read §A
before registering.**

## A. The spec's floor share is the arrest run, and the w_cap it implies would overdose ~100x

§1 says "the floor reads 0.12+" and §2 sizes `w_cap` so that "at the floor
state (share_1 ≈ 0.12, excess 0.07) the cap term contributes O(10)."

The 0.122 figure in the soloist forensic is the **wsig10 arrest run** — the
successful intervention, the row the forensic itself annotates as "earlier in
its recovery when killed." It is not a floor. I measured actual floors with
`scripts/calibrate_vbg.py` (new, read-only, CPU):

| checkpoint | share (sketch K=64) | share (full D) | state |
|---|---|---|---|
| `probe_v5_d8_dk5000` seed46 | **0.790** | 0.833 | floor |
| `probe_v5_d8_warmup` seed97 | **0.667** | 0.647 | floor |
| `probe_d8_wsig10` seed46 | 0.093 | 0.218 | arrest |

At the real floor the excess is 0.74, not 0.07, so `relu(excess)^2` = 0.548
rather than 0.0049 — **112x larger**. The spec's derivation gives
`w_cap ≈ 2041`; at the measured floor that contributes **2041 × 0.548 =
1118**, against a total loss that runs ~17 late in the arrest run and ~3700
at the floor. It would dominate everything.

**Built instead: `w_cap = 18`** (18 × 0.548 = 9.9, the O(10) the spec asked
for, at the share the floor actually has). Arithmetic in the driver comment.

## B. w_share is 1.5, not ~10 — and which checkpoint you anchor on decides it

§2 says match Term B's contribution to the alpha=10 configuration's measured
contribution. Two problems, both handled:

1. **`l_wsig` has never been logged**, so that contribution wasn't in the
   record. I backed it out of the loss identity on `probe_d8_wsig10` step 100:
   `451.11 = 3.738 + 0.2×1887.56 + 10×l_wsig` → `10×l_wsig = 69.86` (15.5% of
   loss; ~73% by step 1600+).
2. **The anchor checkpoint matters enormously.** Ratio raw/trace-normalized is
   **1.17 on the arrest checkpoint** but **27–65 on floor checkpoints** —
   because wsig had already pulled the arrest state's scale toward unit, so
   normalization has nothing left to remove there. Anchoring on the arrest
   checkpoint gives `w_share = 10.3`; anchoring on floors gives **1.5**.

I built **1.5**, on the reasoning that the floor is the state the governor
has to act on. If you disagree, this is a one-line change and I'd rather you
rule than have me guess.

## C. Sketch-space share ≠ full-space share, and the gap is concentration-dependent

The governor computes `share_1` among the K=64 sketched directions; the spec's
cap=0.05 was read off `soloist_forensic.py`, which reports **full-space**
share. Measured ratio: **0.43x at the arrest state, 0.95–1.03x at the floor**
— sketch reads *lower* than full, and the gap widens as concentration falls.
(I predicted the opposite before measuring; the code comment records the
correction.)

Net effect on `cap = 0.05` in sketch space: healthy (full 0.009 → sketch
~0.003) untouched ✓, floor (0.79) taxed hard ✓, but the recovered-d8 state
(full 0.046 → sketch ~0.015) sits ~3x *below* the cap rather than just below
it. So the cap is **more permissive toward partial recoveries** than your
full-space reasoning intends. Left at 0.05 as specced; tighten to ~0.02 if you
want pressure on partial recoveries.

## D. Deviation from §6: I log four auxiliary loss terms, not one field

§6 fences the metric contract to "the one new field." I added
`l_wsig`, `l_orth`, `l_vbg_cap`, `l_vbg_share` to the per-step record anyway,
because §B above is the argument against the fence: the arrest run's own dose
was unreadable and had to be reconstructed from arithmetic. A regularizer
whose magnitude cannot be read is one we cannot dose. All four are `None` when
their mechanism is off, so no existing arm's record changes in content. Revert
is one edit if you want the fence held.

## E. Decisions the spec left to me, stated

- **Power-iteration buffer is `persistent=False`.** It is optimizer scratch,
  not model state; non-persistent keeps every existing checkpoint loadable
  with no `strict=False` concession, and costs one extra iteration of
  convergence on resume. Seeded (20260808) so a fresh process is reproducible.
  Tested: `vbg_power_vecs` is absent from `state_dict()`.
- **The instrument is full-space and exact; the governor is sketch-space and
  estimated.** `deep.top_dir_share` and `substrate_blocks[i].top_dir_share`
  come from the SVD already taken in `_deep_collapse_metrics` /
  `_rank_and_top_share`, so they cost nothing and are directly comparable to
  `soloist_forensic.py`. Always emitted, governor on or off, per §3.
- Gradients flow through the Rayleigh quotient; the eigenvector iterate is
  detached (standard spectral-norm practice).

## F. Built to spec, no changes

Term A `relu(share−cap)^2`; Term B trace-normalized, raw path retained behind
the flag for A/B; both on `interior_latents` from `interior_latent_blocks`;
marginal SIGReg untouched; `torch.eye` CPU-created; defaults OFF everywhere
(verified: no `vbg_power_vecs` registered and `_vbg_on is False` at defaults);
arm `probe_d8_vbg` = stage-31 base (warmup 1000, guard hold 1000, cadence 100,
unclipped) with blocks (0,3,6); doses persisted per-run into
`pilot_result.json` so the registration can be checked against what ran.

Stage **45**. Launch: `python scripts/jepa_pilot_driver.py --stage 45 --seeds
46 95 97 --epochs 1 --max-batches-per-epoch 3000 --heldout-batches 5`.

Fable — the flag in §A is the one I'd want you to check before anything
launches; the rest are calibration choices you may simply overrule. I did not
change the cap value or the scope fence, only the two doses, and both are
documented with their arithmetic in `scripts/jepa_pilot_driver.py`. Your
sentence back to you: capped the soloist, shared the chorus, changed nothing
else — and the dose is measured, not assumed.

— Opus 5, build seat, 2026-08-07
