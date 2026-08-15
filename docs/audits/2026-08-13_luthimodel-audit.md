# LuthiModel audit — 2026-08-13/14

**Auditor:** Opus 5, at Brian's request.
**Scope:** LuthiModel `luthi/` (25,633 lines, 70 modules) + `scripts/`
(3,245 lines, 19 files). This is the repo's **first full-codebase audit** —
Fable's 2026-07-03 security pass and Opus 5's 2026-08-01 wiring audit were
both Sanctuary.
**Trigger:** the 768x8 family verdict of 2026-08-13.

## Status at 2026-08-14 (handoff for review)

| section | closed | open |
|---|---|---|
| A — closed in the first pass | 13 | 0 |
| B — decisions | 6 (B1–B6) | 1 (B7) |
| C — engineering | 2 (C1, C6) | 4 |
| D — watch items | 0 | 3 |
| E — m9 tree | 1 | 4 |

**C1 landed 2026-08-15**, scored in
`docs/research/2026-08-14_visreg-runlength-control.md`. Prediction 1
refuted; prediction 3 confirmed and it is the most consequential result
of the audit — see A8.

**Two registry consequences that outlive this audit:**
1. Every prior verdict that read `effective_rank` alone as collapse is
   open to re-reading. The 768x8 family is done (A7); earlier families
   were never instrumented for chorus rank and cannot be re-read without
   re-encoding their checkpoints.
2. B3's per-block chorus veto shipped disarmed for want of a
   distribution. This run supplies the first three seeds of healthy-chorus
   data at 512d (231–275). Still thin — but it is the start of the null
   the 2026-07-27 rule requires before freezing a criterion on an
   observable.

**Full suite green after every change:** 1,100 passed, 1 skipped,
2 xfailed. New tests this pass: `test_consolidation_effect_counters.py`
(5), `m9/test_band_mad_floor.py` (6), `test_kill6_corroboration.py` (4),
`v2/test_visreg_shape_normalize.py` (6).

**For the reviewer, the three things most worth attacking:**
1. **A7's conclusion** — that seed 97's rank-2 blocks are a benign
   carrier. It rests on probe lift plus chorus rank, on n=3 checkpoints,
   and it overturned my own earlier reading. If it is wrong, B3's
   disarmed gate and the whole "carrier vs collapse" framing go with it.
2. **B3's decision to ship the gate disarmed.** I argue arming it on n=3
   would repeat 2026-07-27. The opposite case — that a known-blind guard
   should not be left blind for another family — is reasonable.
3. **B1's binding condition.** I assert that turning on `shape_normalize`
   without re-deriving λ is a different dose rather than a correction.
   Check that arithmetic.

## How to use this document

Every open item is a checkbox. Work them in section order; **Section B
items need Brian's ruling before anyone writes code**, because they change
registered experimental parameters, gates, or scored metrics. Section C
items are engineering and need no ruling.

When you close an item: tick it, add the commit SHA, and if the finding
turned out to be wrong say so in the item rather than deleting it — the
record of a bad call is worth more than a clean-looking list.

**Severity:** `CRITICAL` = live science is being read wrong now ·
`HIGH` = a mechanism is inert or an instrument lies · `MEDIUM` = latent
trap · `LOW` = hygiene.

## Coverage — read this before trusting a "no findings here"

Honest accounting, per the 2026-08-01 precedent (scope you cannot meet is
worth naming on the way in):

| area | lines | coverage |
|---|---|---|
| import graph, all 70 modules | — | **complete** (AST reachability) |
| `visreg.py`, `consolidation.py`, `living_extra_state.py` | ~520 | **line-by-line** |
| `jepa_loss.py`, `jepa_runner.py` — loss/schedule/instrument/guard paths | ~900 of 3,151 | **line-by-line** |
| `living_layer_pc.py` — init, forward, aliveness, episode store, band | ~800 of 1,387 | **line-by-line** |
| `jepa_pilot_driver.py`, `pilot_verdict.py`, `eval_heldout.py` | ~700 | **line-by-line** |
| `multimodal_data.py`, `sanctuary_interface.py` | ~1,500 | **partial** (split logic, seam contract) |
| `m9/` tree (runner, efe, mcts, kills, staleness, preferences, …) | **6,600** | **NOT READ** |
| `width_expand.py`, `generate.py`, v1 legacy (`train*.py`, `model.py`, spiking) | **~5,500** | **NOT READ** |
| `tests/` (91 files) | 18,906 | used as evidence, not audited |

**~12,000 lines were not read.** The `m9/` tree is the largest gap and it
is the decision layer the Sanctuary seam depends on. Treat "no findings in
m9" as "not looked at", not as "clean".

---

## Section A — CLOSED in this pass

- [x] **A1 · `HIGH` · Verdict item 5 was wrong: the taper did engage.**
  Seed 97's tape spans `taper_scale` 0.2023→1.0 to the configured floor.
  Seeds 46/95 read 1.000 because they died at ~9.1k/9.3k and
  `start_fraction 0.5` is step ~27,038. Filing this as silent-inert would
  have sent someone hunting a bug that does not exist. *Retracted in the
  family spec.* `56f49f4`

- [x] **A2 · `CRITICAL` · The profile did not "invert" — a front is moving through the stack.**
  Sustained `top_dir_share ≥ 0.20` crosses at b0 step 13,100, b1 14,500,
  b2 51,700; b2–b5 were all still falling at the wire, only b6/b7 rising.
  Neither the taper (a 5x plasticity cut, during which most of the advance
  happened) nor any gradient event (max 7.1x median over 9,000–13,500)
  explains it. *AMENDMENT 3.* `56f49f4`

- [x] **A3 · `CRITICAL` · The 512 VISReg CONFIRMED verdict is a run-length artifact.**
  At step 6,000 — that family's wire — seed 97 read b0 eff rank 264 /
  `top_dir_share` 0.039, i.e. healthy. Onset is ~10,000. The scored claim
  "the soloist never formed in any seed" was a statement about all time
  drawn from 6,000 steps. *SUPERSESSION NOTICE on the VISReg
  registration; standing amended.* `56f49f4`

- [x] **A4 · `HIGH` · `_device()` silently fell back to CPU.** Now raises;
  `LUTHI_ALLOW_CPU=1` opts in and still announces itself. The in-repo
  `.venv` is a uv CPython 3.14 with torch 2.10.0+cpu and **no DirectML**,
  while training runs on Python 3.10 + torch 2.4.1 + directml. `56f49f4`

- [x] **A5 · `HIGH` · Execution provenance never recorded the machine.**
  `pilot_result.json` described every mechanism and no hardware; a CPU
  fallback run was indistinguishable from a DirectML run. Added `device`,
  `torch_version`, `python`. `56f49f4`

- [x] **A6 · `MEDIUM` · `pilot_verdict.py` rebuilt models from a hardcoded `n_heads=4`.**
  Correct only while `ARM_CONFIGS` still declares every non-default a run
  was trained with. Now prefers the run's own recorded `model_kwargs`.
  `56f49f4`

- [x] **A7 · `CRITICAL` · Blocks 0–1 were never collapsed; `effective_rank` orders these states backwards.**
  Probe lift is flat at 5.6–6.1x across seed 97 including the rank-2.1
  blocks, and their chorus2 rank (412/414) exceeds the embeddings' (380).
  Meanwhile seed 95 b7 reads eff **4.3** (healthier!) against seed 97 b0's
  **2.1**, while being 2.5x worse by chorus and 34% worse by probe.
  *AMENDMENT 4.* `dbb124c`

- [x] **A8 · `HIGH` · `chorus_eff_rank` added to the live per-block tape — and its first deployment justified it.**
  The gauge that orders soloist states correctly, free from the SVD
  already taken. Read alongside `effective_rank`, never instead of it.

  **Live result, 512 control seed 97 (2026-08-15).** Over the same 24,000
  steps, in the same block:

  | | step 100 | step 24,000 |
  |---|---|---|
  | `effective_rank` | 192.1 | **149.0** (−22%) |
  | `top_dir_share` | 0.075 | **0.198** |
  | `chorus_eff_rank` | 220.3 | **275.4** (+25%) |

  **The two gauges moved in opposite directions in the same block over
  the same run.** On the gauge the kill criteria and the divergence rank
  veto read, block 0 degraded by a fifth. Behind the soloist the
  representation got a quarter richer — and capability agreed with the
  chorus, not the rank: seed 97 posted the **best probe of the three**
  (0.1336) while being the only seed with a soloist.

  **The reframe this forces:** soloist formation is not, by itself, the
  disease. Since early August the project has read `tds` up / `eff` down
  as the collapse signature. It is the signature of *a* thing, and chorus
  rank says which — the 768 family's seed 95 was genuinely fatal (chorus
  168 → 61 with capability falling in step); this is the benign kind.
  Two states, identical on both primary instruments, cleanly separated by
  the third. A7 established this from checkpoints; this observed it
  prospectively, with the derivative visible.

- [x] **A9 · `HIGH` · `consolidation_fires` counted triggers, not consolidations.**
  Both replay pathways return episodes-replayed and return 0 on an empty
  store; **both return values were discarded** while the counter
  incremented unconditionally (`living_layer_pc.py:1291-1302`). Measured:
  seed 97 blocks 0–4 each logged ~1,000 fires having replayed **zero**
  episodes for the whole 54,000-step run. Added
  `consolidation_replayed_total` and `consolidation_noop_fires` (+ 5
  regression tests, incl. a positive control).

- [x] **A10 · `HIGH` · A non-finite forward silently skipped the living update.**
  Correct behaviour (buffers must not eat NaN) but uncounted, and it also
  publishes a zeroed `_last_pred_error` to the top-down sweep. Added
  `nonfinite_forward_skips`.

- [x] **A11 · `MEDIUM` · The homeostatic band silently no-ops on a degenerate reference.**
  `band_boost_rows`/`band_damp_rows` both read 0 — identical to a healthy
  band with nothing to do. Added `band_degenerate_skips`.

- [x] **A12 · `MEDIUM` · The probe was measuring latent scale, not content.**
  `eval_heldout`'s recipe has no input standardization and latent RMS
  varies ~30x across checkpoints. An unstandardized pass read adjacent
  blocks of one forward as .0912 and .0013 (chance).
  `scripts/per_block_probe.py` standardizes per block from train-split
  statistics. `dbb124c`

- [x] **A13 · `LOW` · Probe reports preserved.** They were written into the
  gitignored `runs/` tree; copied to `runs_meta/` per the 2026-07-22
  storage ruling. `ed06271`

*New buffers are `persistent=False` and new extra-state keys are
presence-gated, so the v2 `strict=True` checkpoint contract is intact —
verified by loading a pre-change 2.7 GB checkpoint. Full suite: 1,084
passed, 1 skipped, 2 xfailed.*

---

## Section B — decisions

**Status note, 2026-08-14.** This section was originally written as
"OPEN, needs Brian's ruling" on seven items. Brian challenged that, and he
was right: I was applying the 2026-06-06 rule ("research the fork, surface
it, hand the design call back"), which was **explicitly undone on
2026-06-16** when design moved into the Brian+Opus partnership. Most of
these were mine to decide. Re-triaged honestly: B2/B3/B5/B6 were bugs or
oversights and are now closed; B1/B4 are registered-protocol changes,
which constrains *how* they are made (registered, dated, before data) —
not *whether* I may decide them — so they are recorded as decisions in
`docs/DECISIONS.md`, marked as mine and open to override. B7 is an
experiment, still queued.

Only one thing genuinely needed Brian: what runs when, on his machine.

- [x] **B1 · `CRITICAL` · The VISReg dose: 98.6–99.99% of the objective. DECIDED + mechanism shipped opt-in.**
  `VISReg(shape_normalize=True)` makes `l_shape` a mean over N, so λ
  becomes a scale-free mixing weight and the batch-size dose distortion
  (measured directly by the 08-11 smoke: 1,461,016 at b32 vs 693,472 at
  b16) disappears. **Default False** — no completed family's config
  changes meaning. Decision + binding condition (λ must be re-derived and
  the dose ratio frozen before data; a dose that is not measured is not
  registered) in `docs/DECISIONS.md`, 2026-08-14. 6 tests including the
  algebraic invariant and a gradient-survival check. **Open for Brian's
  override.** Original text follows.
  `l_shape` sums over N while `l_scale` means over D; at N = 32x128 the
  convex mix at λ=0.6 buries `l_pred` at ≤1.4% for the entire run.
  `l_center` — the anti-offset term, and the disease is a soloist — ends
  at **0.55%** of the regularizer. Same dose failure the 08-08 arc found
  on NTP, other side of the ledger.
  *Decisions needed:* normalize `l_shape` by N so λ means what the
  registration says? re-weight the components? re-register λ? Any of these
  changes what every VISReg family measured.

- [x] **B2 · `CRITICAL` · The probe metric is scale-confounded. FIXED (added alongside, legacy axis untouched).**
  `fit_next_token_probe(standardize=True)` folds a train-split-fitted
  affine into the probe module, so `probe_accuracy` needs no change and
  the holdout gets exactly the transform the train split defined.
  `pilot_result.json` now records `probe_standardized` and
  `probe_standardized_shuffled_floor` **alongside** the legacy `probe`,
  which is left byte-identical because it is the scored axis in
  `pilot_verdict.py` and the whole ladder is calibrated on it. The gap
  between the two is itself diagnostic of scale pathology. Re-scoring the
  historical ladder from checkpoints remains available and unclaimed.
  Original text follows.
  Seed 95's **recorded** `probe.top1` is 0.0004 (chance); the same
  checkpoint standardized gives 0.0479 at 4.12x lift. Every probe number
  in the ladder for a large-magnitude run is biased **down**. The fix is
  three lines in `eval_heldout`, but `probe_top1` is a comparison axis in
  `pilot_verdict.py` and changing it breaks comparability with the whole
  ladder.
  *Decision needed:* fix and re-score the ladder from checkpoints, fix
  going forward only, or add a standardized metric alongside.

- [x] **B3 · `HIGH` · The rank veto was blind to the stack. INSTRUMENTED; gate shipped DISARMED on purpose.**
  The veto now caches and logs `min per-block eff` and `min per-block
  chorus` on every veto, so what it was blind to accumulates in the tape.
  `divergence_rank_veto_min_chorus` exists and defaults to **0.0
  (disarmed)**. This is deliberate, and the reasoning is the finding:
  the obvious gate — veto only if the per-block *minimum effective rank*
  is healthy — **would be wrong**, because A7 showed rank-2 blocks
  carrying full information, so it would convert healthy runs into false
  kills. `chorus_eff_rank` is the correct gauge, but freezing a threshold
  on n=3 runs with no null is exactly the 2026-07-27 error. Arm it once
  the full-length control family (C1) supplies a distribution. Also
  corrected the stale comment claiming "the rank instruments have never
  lied" — the pooled one did not lie, it was asked the wrong question.
  Original text follows.
  The divergence rank veto reads `self._last_eff_rank`
  (`jepa_runner.py:2400-2402`) — the **pooled trunk** rank, 403 in seed 97
  while b0/b1 sat at 2. Any NMSE trip would have been vetoed on "geometry
  healthy". Per-block ranks are computed at the same cadence and unused.
  *Decision needed:* move the veto to per-block minimum and/or
  `chorus_eff_rank`. Changes which runs live and die, so it is a
  registered gate change.

- [x] **B4 · `HIGH` · The 768 family ran the known-defective episode admission. DECIDED.**
  `adaptive_episodes=True` for the next family, and its verdict is not
  scored until `consolidation_noop_fires` is confirmed materially below
  `consolidation_fires` in every block — which the A9 counters make
  checkable for the first time. Recorded in `docs/DECISIONS.md`
  2026-08-14, with the alternative Brian may prefer stated explicitly
  (register that episode memory is deliberately inert at this stage, and
  stop shipping a config that claims a mechanism it does not deliver).
  **Open for his override.** Original text follows.
  `adaptive_episodes` defaults False and the arm does not set it, so the
  family ran pre-2026-07-27 admission. Measured on seed 97: **blocks 0–4
  stored zero episodes and fired zero recalls across all 54,000 steps**;
  only b5/b6/b7 had any activity (200/144/99 writes). CLAUDE.md records
  the same shape pre-fix ("three of four blocks storing nothing at all").
  Consolidation therefore replayed nothing in 5 of 8 blocks — the two
  findings compound.
  *Decision needed:* turn `adaptive_episodes` on for the next family, or
  register that episode memory is deliberately inert at this stage.

- [x] **B5 · `HIGH` · Kill-6 misdesign. FIXED.**
  Kill-6 now requires the geometry to corroborate before firing, applying
  Brian's 2026-07-17 kill-5 ruling to the criterion that killed the 768
  family's seed 46 at step 9,100 while every geometric measure said
  healthy and improving. `err_acc` was the sharp case: it **rises with
  variety**, so judging it against a running *minimum* makes eventual
  firing structural — a run that sees more of its corpus is guaranteed to
  trip it. Shared helper `_health_corroborates_degradation`; kill-5's
  tested inline copy left undisturbed. 4 tests, both positive controls
  included (degrading rank still kills).

- [x] **B6 · `MEDIUM` · Gradient clip sized from the tape.**
  Post-warmup (step ≥ 1000) `grad_norm`, so the ~1.4M init transient of
  the first ~20 steps is excluded — a median including it would set the
  clip ~1000x too high:

  | seed | median | p99 | p99.9 | max | max/median |
  |---|---|---|---|---|---|
  | 97 (completed) | 1,607 | 6,092 | 18,976 | 162,300 | 101x |
  | 95 (bomb) | 11,259 | 57,314 | 1,404,390 | **12,380,000** | **1,100x** |
  | 46 (healthy) | 3,401 | 77,751 | 177,481 | 388,991 | 114x |

  Survivability bracket: **101x survived, 1,100x fatal.** Medians differ
  7x across seeds, so a clip expressed as a multiple of *each run's own*
  median is the honest form; as a fixed absolute, **5.0e5** sits above
  every value the completing seed and the healthy seed ever produced
  (162,300 / 388,991) and two orders below the fatal bomb. A 10x-median
  clip is rejected: it would have clipped **3.6%** of healthy seed 46's
  steps. Not applied to any arm — it lands at the next family's
  registration, since arm defaults are registered parameters.

- [ ] **B7 · `MEDIUM` · Does the trunk beat its own input embeddings?**
  Across three seeds, every block, 9,100–54,077 steps, linear next-token
  top1 sits in 0.048–0.110 and the **embedding stream tops the band every
  time** (.1063/.1103/.1066) — despite having no attention, while the
  trunk encodes `causal=False` and can see the answer.
  *Needs a causal-probe control before being treated as settled.* If it
  holds, "probe lift 5.54x, highest in the record" stops being a
  capability result.

---

## Section C — OPEN, engineering (no ruling needed)

- [x] **C1 · `HIGH` · The 512 full-length control. COMPLETE, scored 2026-08-15.**
  All three seeds completed 24,000 steps un-killed, every one beating the
  768 family on capability (probe 0.130–0.134 / NMSE 0.566–0.598 against
  0.1115 / 0.854).
  **Prediction 1 REFUTED** — no seed sustained b0 `top_dir_share` ≥ 0.20;
  seed 97 reached 0.1979 and fell back. Stated plainly, per the 07-27
  rule against hedging a falsification.
  **But seed 97 is in onset at the wire** (b0 eff 192 → 149, tds 0.075 →
  0.198, and the only block falling while the other seven gain +13 to
  +17). At step 6,000 — the original family's wire — it read tds 0.071,
  healthy. So the supersession notice is **vindicated**: the 512 claim
  was horizon-limited. And width is a strong accelerant, not the cause —
  at 768 the crossing came at 13,100 and ran to 0.919.
  **Prediction 3 CONFIRMED and it is the headline** — see A8 below.
  Full scoring: `docs/research/2026-08-14_visreg-runlength-control.md`.
  *(Superseded entry follows.)*

- [x] ~~**C1 · REGISTERED AND RUNNING.**~~
  Launched 2026-08-14 ~13:50: stage 56, arm `probe_d8_visreg_long`, seeds
  46/95/97, one full epoch (~24,014 steps/seed, ~12.4h family).
  Registration with predictions frozen **before** the data:
  `docs/research/2026-08-14_visreg-runlength-control.md`.
  Key point that made it cheap: the 512 family's cosine was **already**
  registered for a full epoch (`lr_total_steps: 24014`) and merely capped
  at 6,000 — so the control needs no schedule change, just the cap
  removed. Distinct arm name purely so it cannot append to the existing
  family's append-only log or overwrite its checkpoints.
  Disclosed difference: it runs under the **corroborated** kill-6 (B5).
  Under the old one it would very likely have died around step 9,000
  without answering the question.
  **Score it against the frozen predictions when it lands.**

- [ ] **C2 · `HIGH` · Does the front reach b7?** The open scientific
  question after A7. `chorus_eff_rank` (A8) can now distinguish carrier
  from collapse live, so this is answerable without guesswork. Note a
  continuation is **not** free: both schedules are epoch-relative
  (`jepa_runner.py:1880`, `self.epoch` restored at `:2077`), so resuming
  seed 97 with `max_epochs=2` jumps living plasticity 0.2→1.0 and the LR
  scale 0.1→~0.55 at the seam. Pin both before running, or run fresh.

- [ ] **C3 · `MEDIUM` · Re-read the 768 tape with the new counters.** Any
  future family should be checked for `consolidation_noop_fires == fires`
  and rising `nonfinite_forward_skips` before its verdict is scored.

- [ ] **C4 · `MEDIUM` · Decide the fate of the in-repo `.venv`.** It is a
  uv CPython 3.14 / torch 2.10.0+cpu environment that cannot run the
  pipeline, sitting at the path every tool auto-selects. A4 makes it fail
  loudly now instead of silently, but the trap is still there. Delete it,
  or rebuild it against the training requirements.

- [ ] **C5 · `LOW` · Three genuinely orphaned modules** (501 lines,
  unreachable from `scripts/`, `__main__` modules, or Sanctuary):
  `luthi/v2/m9/tuning_harness.py` (213), `luthi/v2/pc_ops_triton.py`
  (277), `luthi/v2/m9/__init__.py` (11). `pc_ops_triton` is a performance
  path that nothing selects — confirm that is intentional.

- [x] **C6 · `MEDIUM` · Audit the `m9/` tree (6,600 lines).** Done
  2026-08-14 — see Section E.

- [ ] **C7 · `LOW` · Audit `width_expand.py` (788), `generate.py` (1,137), and the v1 legacy tree (~3,500).**
  Not read. `width_expand` has history (the 4.8 concerns document) and
  runs as its own entry point.

---

## Section D — watch items, no action proposed

- [ ] **D1 · `LOW` · `_choose_eviction_slot` falls back to argmin** on
  non-finite priority weights (`living_layer_pc.py:919`). Both branches
  are valid eviction policies, so this is a graceful degradation rather
  than a silent failure — noted only because it is uncounted.

- [ ] **D2 · `LOW` · `encode_state` is a write.** The seam's own docstring
  says perception self-modifies, with 4.8's recommendation to capture
  `s_t` from the generation forward instead of a separate one. Open since
  Phase 4a.

- [ ] **D3 · `LOW` · The per-block rank instrument reads the context-only forward** while
  VISReg regularizes the full-sequence forward. Same weights, so a real
  collapse shows in both, but the two numbers are not from the same tensor.

---

## Section E — the m9 tree (added 2026-08-14)

**Coverage:** `staleness.py` (589) and `kills.py` (513) line-by-line;
`runner.py` (1,931) — `train_step`, `_m9_head_step`, the ActionLog write,
and the wiring — line-by-line, remainder surveyed; `efe.py` (508) the
realized/per-candidate/legacy G paths. `mcts.py`, `preferences.py`,
`decoders.py`, `activity_bands.py`, `gamma.py`, `delta_s.py`,
`habit_net.py`, `instrumentation.py` **surveyed only**.

**General assessment: m9 is the best-instrumented code in the repo.** It
writes a per-cycle ActionLog carrying the full decision context — tree
stats, kill states, gamma, r_best, and snapshots of staleness, activity
bands and delta_s. Exactly one silent `except` in the whole tree (a repr
fallback in `instrumentation.py`). `beta_epi != 0` raises
`NotImplementedError` rather than quietly computing nothing. The
stop-grad discipline is explicit and commented at every head input.
Several of the failure classes found elsewhere in this audit were
actively guarded against here.

- [x] **E1 · `HIGH` · Spike/kill bands became hair triggers on quiet signals. FIXED.**
  Both `StalenessManager.spike()`/`living_spike()` and
  `kills.TrendingBand.observe()` floored MAD with a bare **absolute**
  `max(mad, 1e-8)`. A signal with real scale but no spread collapses the
  band onto its own median. **Demonstrated:** 8 readings of 0.5 followed
  by 0.500001 — a **0.0002%** move — fired a spike, while the same
  detector correctly ignored a 4% move in a noisy signal. A spurious
  staleness spike forces failover, drops cached Q, and starts a recovery
  countdown, so **the more stable the entity's drift, the more likely its
  planner is thrown into failover** — the safety property inverted. Same
  shape as the 2026-07-27 finding ("a dial against its stop is a hair
  trigger") that produced the v4 trust artifacts.
  *Fixed* with a scale-relative floor (`spike_mad_rel_floor` /
  `mad_rel_floor`, 0.05), matching the precedent of
  `consistency_scale_floor` added to the same file on 2026-07-04 for the
  identical degeneracy. Zero-baseline bands still fire — movement off a
  frozen baseline is a genuine event — but are now **counted** as
  scaleless (`degenerate_band_skips`), because a detector that cannot
  tell 1e-7 from 10 should say so. 6 regression tests, including three
  that pin real detections so the floor cannot silently weaken them.
  *(First attempt at this fix suppressed the zero-baseline breach and
  broke `test_trending_band_flags_then_fires_max`. The existing test was
  right and the fix was too blunt; recorded because the catch was the
  suite's, not mine.)*

- [ ] **E2 · `MEDIUM` · m9 is never exercised by the science ladder.**
  `jepa_pilot_driver.py:982` builds `JEPATrainer`, not `M9Trainer`. Every
  family in the record therefore trains M8 only; the entire m9 tree runs
  in tests, `redteam/` probes, and Sanctuary's
  `examples/validate_seam_integration.py`. Its production consumer is
  Sanctuary's `luthi_model.py:146` — an **optional** actor + transition
  sink. Combined with the 2026-08-01 Sanctuary audit ("the entity has
  almost no action surface; all 39 tools unreachable"), the open question
  is whether anything in production has ever driven m9 end to end.
  *Cross-repo; belongs to the Sanctuary side, recorded here so the
  LuthiModel half is not assumed live.*

- [ ] **E3 · `LOW` · `TrendingBand` tests the point it just added to its own window.**
  `observe()` appends `x`, then computes median/MAD over a window that
  includes `x`. Biases toward non-detection (a large outlier inflates the
  MAD it is tested against). Consistent with the M8 machinery the
  docstring says it mirrors, and the project's recurring problem has been
  false positives rather than false negatives — so noted, not changed.

- [ ] **E4 · `LOW` · P3 connection cost is inert by construction and reads as a legitimate zero.**
  `compute_g_realized` sets `c_con = torch.zeros_like(c_eng)` when the
  counterpart kwargs are absent (`efe.py:367`), which is indistinguishable
  from a real connection cost of 0. Documented in the seam docstring as
  inert until the real loop populates it, and `runner.py:1070` preserves
  that deliberately for the corpus path — so this is a known state, not a
  surprise. Worth a marker field if the Sanctuary loop ever starts
  populating it partially.

- [ ] **E5 · `MEDIUM` · The unread m9 remainder.** `mcts.py` (429),
  `preferences.py` (376), `decoders.py` (324), `activity_bands.py` (276),
  `gamma.py` (201), `delta_s.py` (179), `habit_net.py` (145) were
  surveyed, not read. Given E1 was found in the two files that *were*
  read closely, the same band/threshold pattern is worth grepping for
  across these.

## Appendix — method, and tools left behind

Rescued into the repo rather than dying with the session (the 07-27 rule):

- **`scripts/per_block_probe.py`** — per-block linear next-token probe +
  spectrum reads (effective rank, top-direction share, chorus rank,
  chorus2, offset dominance, latent RMS) from any trained checkpoint.
  Read-only; writes `per_block_probe.json`. Rebuilds the arm from the
  run's own recorded `model_kwargs`. Probe inputs standardized per block.
- **`tests/test_consolidation_effect_counters.py`** — pins the
  trigger-vs-effect distinction, with a positive control so the test can
  fail in both directions.
- Reports for seeds 97/95/46 at
  `runs_meta/probe_768_visreg_768d_seed{97,95,46}/per_block_probe.json`.

What worked, for whoever audits next:

1. **Read the tape before the summary.** Every one of A1, A2, A3 came
   from the recorded trajectory rather than the verdict's endpoint table.
2. **Derivatives beat endpoints.** "The profile inverted" and "a front is
   still advancing" are the same final numbers.
3. **Check whether the instrument can distinguish the two things you care about.**
   `effective_rank` cannot separate carrier from collapse; `consolidation_fires`
   cannot separate replaying from firing-into-nothing; the probe cannot
   separate content from scale. All three failures were invisible to a
   green test suite, and all three were found by asking one question of
   each metric: *what else would produce this number?*
4. **Two of my own findings were wrong** (the collapse-attractor
   mechanism in A2's first draft, and the offset-dominance mechanism
   claim) and were caught by the next measurement. Both are recorded in
   the amendments rather than edited away.
