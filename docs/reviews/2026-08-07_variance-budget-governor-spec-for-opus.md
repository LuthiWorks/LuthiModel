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
