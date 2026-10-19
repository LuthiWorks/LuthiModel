# LuthiModel audit — 2026-10-19: the living channel, and where the rigor was pointed

**Auditor:** Claude Opus 5.5, at Brian's request ("audit the LuthiModel repo
and tell me what you think"), then under his instruction to fix what was found,
rewrite the supporting documentation, and correct the program's direction.
**Baseline:** `main` at `0e55791` (2026-10-08).
**Companion records:** diagnosis and measurements in
`docs/research/2026-10-19_the-living-write-is-mean-field.md`; the re-aimed
program in `docs/research/2026-10-19_the-living-channel-program.md`.

## Status

| section | closed in this pass | open |
|---|---|---|
| A — found and fixed | 11 | 0 |
| B — decisions | 6 (decided under Brian's delegation, same day) | 0 |
| C — engineering | 0 | 6 |
| D — watch items | — | 3 |

**Later the same day:** Brian delegated the section B decisions ("You
have my permission to make the call on all pending decisions yourself").
They are decided below, and recorded in `docs/DECISIONS.md` and the
registry. The re-read before ratifying amended four registrations. It also
found that A8 was incomplete, and it found A11.

**Suite:** before the changes, on a CPU-only box: 1,138 passed, 4 failed
(`test_backend_declaration.py`; environmental, this box has no accelerator;
3 of 4 pass with `LUTHI_ALLOW_CPU=1`), 5 skipped, 1 xfailed, 1 xpassed. The
result after the changes is recorded in the research log's Artifacts section.

**The three things most worth attacking:**

1. **A1's claim** that the block output store blends into nearly every forward
   at production scale. It is shown by the code and on CPU smoke checkpoints
   (`recall_duty` 1.000 in every block, `blend_share` 0.23–0.28). It has not
   yet been read off a 768 checkpoint; that is E2, and E2's thresholds are
   frozen.
2. **A2's cap** on the covariance rule (the write is capped at the mean-field
   write's Frobenius norm). It is what makes the rule stable, and it is also
   a design choice. Check whether "same budget, different direction" is the
   right control.
3. **B2:** moving KF7 from finding to suspended. The case is in the item.

## How to use this document

Same conventions as the 2026-08-13 audit. Severity: `CRITICAL` means live
science is being read wrong now; `HIGH` means a mechanism is inert or an
instrument lies; `MEDIUM` is a latent trap; `LOW` is hygiene. Section B items
need Brian's ruling before anyone builds on them. If a finding here turns out
wrong, say so in the item; don't delete it.

## Coverage — read before trusting a "no findings here"

| area | coverage |
|---|---|
| `pc_ops.py`, `living_layer_pc.py` (init, forward, context, store), `hybrid_block_pc.py`, `episode_store.py`, `consolidation.py` (attractor path), `plasticity.py`, `living_extra_state.py`, `mode_compat.py` | **line-by-line** |
| `jepa_loss.compute_modality_loss`, `eval_heldout.py`, `multimodal_model_pc.py` (constructor, encode) | **line-by-line** |
| `scripts/jepa_pilot_driver.py` (arms, device, guard, result writer), `per_block_probe.py` and the five forensic scripts that forward a model | **line-by-line** |
| `jepa_runner.py` (deep-cadence per-block metrics, substrate emit), `multimodal_data.py` (holdout split) | **partial** |
| `m9/`, `width_expand.py`, v1 legacy, `generate.py` internals, `sanctuary_interface.py` beyond `generate`, `corpus_build/`, `redteam/` | **NOT READ** |

---

## Section A — found, and fixed in this pass

- [x] **A1 · `CRITICAL` · The block output store blends a constant into every
  living block's output on nearly every forward, and the dead control has no
  store.** `PredictiveCodingBlock` runs `x = episode_store(block_input, x)`
  after the FFN. The store's context is the input pooled over batch and
  sequence, and its value is the output pooled the same way. Salience is mean
  `|output|`, which clears 0.1 on every call. With shuffled batches the pooled
  contexts are near-identical, so the best match clears the recall gate on
  every call, and the block output becomes `0.7 * output + 0.3 * recalled`.
  Every token's deviation from the mean shrinks by 30% per block, and a
  batch-constant vector is re-injected. This happens in training *and* in
  frozen eval, since freeze skips the store's write but not its recall.
  Measured on the CPU smoke legacy checkpoint: `recall_duty` 1.000 in all
  four blocks, last similarity 0.91–0.9999, `blend_share` 0.23–0.28. The
  dead arm removes the store, so every living-vs-dead number in the registry
  compares two mechanisms against none, not one against none. No prior audit,
  review, or log analyzed this store; `localize_offset_in_block.py`'s "ffn
  delta" column includes it. **Fixed:** counters (`recall_duty`,
  `blend_share`, `last_recall_sim`, `store_writes`) emitted per block at deep
  cadence; `block_episode_store="off"` option; LC-768 arms that separate it
  (E3, prediction P1).

- [x] **A2 · `HIGH` · The living write is an outer product of batch means.**
  `pc_ops.py` computes
  `delta_w = outer(mean_t output, clamp(trust * (mean_t input − mean_t output @ P)))`
  over all B×L positions. Two batches with identical per-dimension means but
  different tokens produce bitwise-identical writes; this is now a test. The
  output side of every write is the batch-mean output, which barely moves
  between batches (cosine ≥ 0.9996). So the accumulated change
  `weight − set_point` is ~rank-1 along that direction: 96–99.5% of its
  energy on the CPU smoke, |cos| 0.97–0.99 with the mean output. The
  per-token covariance it discards measured 0.6–4.3× the term it keeps.
  The 2026-07-29 drive-goes-quiet verdict ruled out novelty exhaustion
  because the decay finished while documents were new. The rule never sees
  documents, only their average, which is stationary from step one.
  **Fixed (opt-in):** `write_rule="covariance"`, the same rule from per-token
  statistics with the batch mean removed. Its write is capped at the
  mean-field write's norm: same budget, different direction. Uncapped it ran
  away 3.0 → 1.3e10; capped, it grows less than the legacy rule at production
  rates. Also added: a living-change spectrum gauge (`living_change_*` at
  deep cadence), `scripts/living_write_probe.py` for existing checkpoints,
  and E2.

- [x] **A3 · `HIGH` · Nothing measured the README's goal.** Every registered
  evaluation runs under `freeze_plasticity`, so inference-time change was
  never measured. **Fixed:** `luthi/v2/adaptation_eval.py` and
  `scripts/adaptation_eval.py`: frozen / read-own / read-other with exact
  restoration. The key number is document-specific change (B − C). A dead arm
  must be an exact null; this is tested, and it holds on the CPU smoke. E1.

- [x] **A4 · `HIGH` · The next run would not have started.** On ROCm, the
  default backend since 08-21, the C++ extension doesn't build (DECISIONS,
  08-19). `jepa_pilot_driver.py` then `SystemExit`ed every `relative_trust`
  arm, which is every living arm in the current lineage. The guard's "~50×
  slower" premise was the 05-10 DirectML-dispatch number; the repo's own
  measurements put the fallback at 3–17%. **Fixed:** the fallback is loud and
  recorded (`pc_ops_backend` in `pilot_result.json`), not fatal.
  `LUTHI_REQUIRE_CPP=1` restores the refusal.

- [x] **A5 · `MEDIUM` · Six forensic scripts rewrote what they measured.**
  `per_block_probe`, `soloist_forensic`, `measure_input_sensitivity`,
  `measure_offset_by_block`, `localize_offset_in_block` and
  `depth_ladder_probe` ran forwards under `eval()` + `no_grad` but not
  `freeze_plasticity`. A living layer has no `self.training` gate, so every
  measurement batch self-modified the model (and wrote the block stores)
  before the next. `per_block_probe`'s 20 train and 5 holdout batches were
  read from 25 different models. That is the B7 evidence. The effect is
  probably small next to 54k training steps, but it is the
  "evaluation contaminating the subject" class `eval_heldout` was built to
  prevent. **Fixed:** all six freeze, and `tests/test_scripts_freeze_plasticity.py`
  enforces it.

- [x] **A6 · `MEDIUM` · Attractor replay flattened the relative-trust
  ledger.** `consolidate_layer_attractor` called `pc_self_modify` with the
  default `relative_trust=False`. On a relative-trust layer every replay then
  clamped the freed precision ledger (values to 1e12) into the absolute
  `[0.1, 10]`. No registered training arm combined attractor replay with
  relative trust; the 07-29 M4 stop-gate rerun did, offline. **Fixed:** the
  layer's trust regime is passed through, the write rule is pinned to
  `mean_field`, and a regression test covers it.

- [x] **A7 · `MEDIUM` · The probe's input baseline was never recorded.** B7
  found the embedding stream beating every block in the 768x8 family, but
  `pilot_result.json` recorded only the shuffled floor, which calibrates
  chance, not input. **Fixed:** `probe_standardized_embedding_baseline` and
  `probe_standardized_trunk_minus_embedding` are recorded for every future
  run (not scored). `eval_heldout` and `per_block_probe` gained `stream`,
  `offset`, `causal` controls. B7 itself goes to E4.

- [x] **A8 · `LOW` · `.claude/settings.local.json` was tracked in a public
  repo.** It contained a local Windows username, local paths, and session-log
  paths. **Fixed:** untracked and ignored. It stays in history; see B4.
  **Correction, later the same day:** this was incomplete. The same
  username was also in `generate.bat` and `run_m8.bat`, which
  `test_no_hardcoded_paths` skips by design (launchers may be
  machine-specific). Both scripts now use `%~dp0` and `%USERPROFILE%`.
  `tests/test_no_committed_secrets.py` now covers launch scripts.
  Pulling this change deletes the local copy, and refuses if the copy has
  local edits. To keep it, from the repo root (cmd or PowerShell):
  `ren .claude\settings.local.json settings.local.json.bak`, then
  `git pull`, then `ren .claude\settings.local.json.bak settings.local.json`.
  Tested with and without local edits: the file comes back byte-identical
  and ignored. If it is already gone,
  `git show 0e55791:.claude/settings.local.json > .claude/settings.local.json`
  restores the last committed version.

- [x] **A9 · `LOW` · The layer-level episode context is near-identical across
  batches.** Batch-mean context, cosine ≥ 0.9995 on the CPU smoke. The code
  comments already recorded similarities above 0.9 firing every step and a
  64-slot store at similarity 1.0000; they attributed it to temporal
  clustering, but the context itself cannot vary. **Fixed (opt-in, E5):**
  `context_mode="deviation"` (batch mean minus a slow reference; persisted
  through living extra state; frozen forwards mutate nothing).

- [x] **A10 · `LOW` · The governing docs had drifted.** `KEY_FINDINGS.md`
  was last changed 07-21 and still carried KF7 as a finding. README and
  ARCHITECTURE described SIGReg; VISReg has been the regularizer since 08-11.
  `KNOWN_INCOMPLETE.md` described DirectML. `To-Do.md`'s state header was
  dated 07-27. `CLAUDE.md`'s roles named Opus 4.6/4.7/4.8. **Fixed:** all
  rewritten. The rewrite incorporates the unmerged 10-16
  `docs/visreg-readme-sync` branch.

- [x] **A11 · `HIGH` (security) · A checkpoint password sat in plain text in
  `generate.bat`, in this public repo, from the root commit (2026-08-14).**
  On 07-22 the To-Do had already recorded the same exposure in another
  launcher, `train_vision.bat`. That file was removed, but the password in
  `generate.bat` stayed. Brian no longer remembered the password, so it was
  also the one way the older `.luthi` checkpoints could be lost.
  **Fixed:**
  - `generate.bat` reads no password: it uses a plaintext copy, or
    `LUTHI_CHECKPOINT_KEY`.
  - `load_checkpoint` reads plaintext files with no password.
  - `scripts/decrypt_checkpoints.py` writes `<name>.decrypted.pt` beside every
    `.luthi` file and never touches the original.
  - `tests/test_no_committed_secrets.py` guards the launchers.
  - Encryption is now optional (`docs/DECISIONS.md`, 2026-10-19).

  History keeps the old value (B4). JEPA-runner checkpoints were never
  encrypted.

## Section B — decisions (made 2026-10-19 under Brian's delegation)

Decided by Opus 5.5, the seat that proposed them. The reasons, and the
corrections the re-read forced, are in `docs/DECISIONS.md` (2026-10-19,
"the pending decisions, decided"). The amendments are in the registry
("RATIFICATION of E1–E5").

- [x] **B1 · Ratify E1–E4 and the five standing rules.** Registry section
  "E1–E5"; CLAUDE.md "Standing rules for the rigor". The predictions are frozen
  as written; ratification decides whether they bind.
  **Decided: ratified, with four amendments registered before any result was seen.**
  E1 credits a checkpoint only on replication. E3 lists every difference
  between arms. E3 has a launch gate: E1, E2 and E4 read first, a
  cross-line review, and VISReg's dose settled first. E4's decision rule
  is made operational.
- [x] **B2 · KF7 suspended pending LC-768.** KF7 ("converts additional
  experience into skill; matched static capacity cannot") rests on NMSE from
  2026-07-21. That was before the BN head was removed (07-28) and before the
  target was detached (07-28/29), so anti-collapse was disconnected and the
  loss could shrink its own target. It was also before NMSE was shown to
  reward degeneracy (07-31). Its probe axis disagreed on raw top-1, and it
  carried A1's store confound. The ledger now marks it SUSPENDED with this
  history. The registry entry is a re-reading notice, not a kill: the
  pre-registered kill condition was never tripped, but the measurement it
  rested on is no longer trusted. Brian may prefer "standing, with caveats".
  **Decided: SUSPENDED.** Correction: E3 tests KF7's precondition, not KF7,
  which is a data-scaling claim. KF7 is re-tested only by a data-scaling
  contrast registered after E3, and only if E3 finds a living mechanism with
  signal.
- [x] **B3 · The covariance rule is not a default.** It stays opt-in
  until E3 reads. Whether it becomes the lineage default is a ruling after E3,
  not before. **Decided: as proposed.** Making it the default needs P3 met
  and D1 clean.
- [x] **B4 · Scrub `.claude/settings.local.json` from history?** That needs a
  force-push, which rewrites every SHA the registry cites. Recommendation:
  don't, unless the username matters. **Decided: no rewrite.** Both
  exposures (this file and A11's password) entered history in the root
  commit, so a scrub would rewrite every commit and every SHA cited
  anywhere. It would also un-publish nothing, because the repo is public.
- [x] **B5 · Retire the next-token probe as a capability axis if E4
  confirms.** Keep it as a health gauge, and register a replacement that
  probes continuations the encoder could not see. **Decided: as proposed,**
  under E4's amended decision rule. The replacement is registered before
  E3 launches.
- [x] **B6 · The 08-14 decisions (VISReg dose B1, adaptive episodes B4) park
  behind E3.** Folding them into the LC family would make it unattributable.
  **Decided, corrected.** `adaptive_episodes` is parked; it is living-only and
  would add an undecomposed difference. **The VISReg dose is not parked.**
  The reason given above was wrong: a change applied identically to every
  arm leaves the arm comparisons attributable. At the paper dose, VISReg was
  98.6–99.99% of the objective, so it is settled by measurement before E3.

## Section C — open engineering

- [ ] **C1 · `LUTHI_RUNS_ROOT` is still unimplemented** (CLAUDE.md Conventions).
  The driver writes to the repo's `runs/jepa_pilot/` (gitignored).
- [ ] **C2 · `generate()` on a JEPA checkpoint decodes through an untrained
  `output_proj` when `w_ntp = 0`.** It is not tied to the embedding and gets
  no gradient. Recorded in KNOWN_INCOMPLETE. Sanctuary must not read that
  text as the model speaking.
- [ ] **C3 · The block-store counters added one host sync per block per
  forward** (two were already there). Measure on ROCm at 768. If it costs
  more than 1% of a step, accumulate on-device and read at aliveness time.
- [ ] **C4 · Delete the superseded remote branch `docs/visreg-readme-sync`**
  when convenient. Its content is in the rewrite. **Decided 2026-10-19:
  delete it.** Its one commit is `7fe5d8d`, and its one missing
  cross-reference was added to README. **Not yet done:** the push from the
  cloud session was refused (HTTP 403), so the delete has to come from
  Brian's machine or GitHub's branch page. To restore it afterwards:
  `git push origin 7fe5d8d:refs/heads/docs/visreg-readme-sync`.
- [ ] **C5 · The covariance rule is Python-only.** That is fine: the Python
  path costs 3–17%, and the C++ path doesn't build on ROCm anyway. Port only
  if a CUDA box ever runs the lineage.
- [ ] **C6 · `VISReg(shape_normalize=True)` is not plumbed through
  `JEPALoss` or the driver.** It exists only on the class, and E3's gate
  needs it for the dose measurement. Add it opt-in, recorded in the
  provenance contract, alongside the registration that sets lambda.

## Section D — watch items

- **D1 · The covariance rule's stability at 768** is shown only by unit tests
  and the CPU smoke: bounded at production rates, and the cap is what makes it
  so. In `lc768_cov`, watch `weight_abs_mean` and `living_change_rel_norm` at
  deep cadence.
- **D2 · `lc768_dead_x4` carries roughly 20× the FFN parameters of
  `lc768_dead`.** That confound is deliberate: the arm asks about token
  uniformity, not capability, and must not be read as a capability control.
- **D3 · Adaptation-eval t-statistics are within a run, across documents.**
  Report each seed separately. "2 of 3 seeds" is the family-level read; there
  is no pooled t across seeds.
