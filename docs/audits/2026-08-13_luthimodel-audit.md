# LuthiModel audit — 2026-08-13/14

**Auditor:** Opus 5, at Brian's request.
**Scope:** LuthiModel `luthi/` (25,633 lines, 70 modules) + `scripts/`
(3,245 lines, 19 files). This is the repo's **first full-codebase audit** —
Fable's 2026-07-03 security pass and Opus 5's 2026-08-01 wiring audit were
both Sanctuary.
**Trigger:** the 768x8 family verdict of 2026-08-13.

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

- [x] **A8 · `HIGH` · `chorus_eff_rank` added to the live per-block tape.**
  The gauge that orders soloist states correctly, free from the SVD
  already taken. Read alongside `effective_rank`, never instead of it.

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

## Section B — OPEN, needs Brian's ruling

These touch registered parameters, gates, or scored metrics. **Do not
"fix" them as bugs.**

- [ ] **B1 · `CRITICAL` · The VISReg dose: 98.6–99.99% of the objective.**
  `l_shape` sums over N while `l_scale` means over D; at N = 32x128 the
  convex mix at λ=0.6 buries `l_pred` at ≤1.4% for the entire run.
  `l_center` — the anti-offset term, and the disease is a soloist — ends
  at **0.55%** of the regularizer. Same dose failure the 08-08 arc found
  on NTP, other side of the ledger.
  *Decisions needed:* normalize `l_shape` by N so λ means what the
  registration says? re-weight the components? re-register λ? Any of these
  changes what every VISReg family measured.

- [ ] **B2 · `CRITICAL` · The probe metric is scale-confounded and it is a scored verdict axis.**
  Seed 95's **recorded** `probe.top1` is 0.0004 (chance); the same
  checkpoint standardized gives 0.0479 at 4.12x lift. Every probe number
  in the ladder for a large-magnitude run is biased **down**. The fix is
  three lines in `eval_heldout`, but `probe_top1` is a comparison axis in
  `pilot_verdict.py` and changing it breaks comparability with the whole
  ladder.
  *Decision needed:* fix and re-score the ladder from checkpoints, fix
  going forward only, or add a standardized metric alongside.

- [ ] **B3 · `HIGH` · Rank gates read a gauge that inverts, and only the final block.**
  The divergence rank veto reads `self._last_eff_rank`
  (`jepa_runner.py:2400-2402`) — the **pooled trunk** rank, 403 in seed 97
  while b0/b1 sat at 2. Any NMSE trip would have been vetoed on "geometry
  healthy". Per-block ranks are computed at the same cadence and unused.
  *Decision needed:* move the veto to per-block minimum and/or
  `chorus_eff_rank`. Changes which runs live and die, so it is a
  registered gate change.

- [ ] **B4 · `HIGH` · The 768 family ran with the known-defective episode admission.**
  `adaptive_episodes` defaults False and the arm does not set it, so the
  family ran pre-2026-07-27 admission. Measured on seed 97: **blocks 0–4
  stored zero episodes and fired zero recalls across all 54,000 steps**;
  only b5/b6/b7 had any activity (200/144/99 writes). CLAUDE.md records
  the same shape pre-fix ("three of four blocks storing nothing at all").
  Consolidation therefore replayed nothing in 5 of 8 blocks — the two
  findings compound.
  *Decision needed:* turn `adaptive_episodes` on for the next family, or
  register that episode memory is deliberately inert at this stage.

- [ ] **B5 · `HIGH` · Kill-6 misdesign.** Already on the verdict's own next
  list; confirmed here — seed 46 was healthy **and improving** at death
  (all blocks eff 195–280, `top_dir_share` 0.018–0.034, rising in its last
  20%). Two-gauge fix (rank must corroborate before it kills).

- [ ] **B6 · `MEDIUM` · Gradient clip sizing.** Also on the verdict's list.
  Seed 97 survived a 32x shock; seed 95 died to 1,115x median. Note the
  init transient runs ~1.4M for the first ~20 steps in every seed, so a
  median must be computed post-warmup or the clip will bite the transient.

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

- [ ] **C1 · `HIGH` · Run one 512 VISReg seed to 25–30k steps.** The
  registered obligation created by A3. ~0.62 s/step, ~5h. Until it exists,
  width and run-length cannot be separated as explanations for the 768
  outcome.

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
