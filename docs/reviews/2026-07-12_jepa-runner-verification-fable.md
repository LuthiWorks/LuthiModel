# JEPA Runner Verification Review — Critical-Path Item 2 Ground-Truthing

**Date:** 2026-07-12
**Reviewer:** Fable 5 (cross-line verification seat)
**For:** Brian + Opus 4.8 (design/plan seat)
**Scope:** Firsthand verification of the critical-path To-Do's claims about
`luthi/v2/jepa_runner.py` (items 2 and 3 of `fbb714a`), plus a full
correctness review of the runner. Trigger: the To-Do quoted the runner's
module header, and the header did not match the code beneath it.

---

## Headline: the critical path inherited a stale header

The To-Do's item 2 ("finish jepa_runner.py's 6 must-fix items,
self-labeled NOT production-ready") restates the module docstring written
2026-06-06 — but five of the six items were **fixed during 2026-06-06..08**
(`deaf1ec`, `189001c`, `47187f4`, `72526cb`, `89eefbe`) and the header was
never updated. The 2026-07-06/07-10 readiness audit took the header's word.
Verified per item against the code and git history:

| Header claim (quoted into To-Do) | Ground truth (2026-07-12) |
|---|---|
| 1. Per-modality cadence missing | **DONE** `deaf1ec` 2026-06-06 |
| 2. Kill-7 per-modality smoothed loss | **Resolved by decision** — every-step append landed (`deaf1ec`); mixed modalities ruled intentional (total-objective criterion). See M1 below for a new open issue. |
| 3. "Arm kill-2 + kill-4 — computed but not armed" | **Kill-2 ARMED** `89eefbe` 2026-06-08. **Kill-4 (LID) is worse than claimed: not computed at all** — deliberately deferred in `_deep_collapse_metrics` ("follow-up; flagged"). |
| 4. Wire kill-6 via `aliveness_report()` | **DONE** `47187f4` 2026-06-08 |
| 5. Predictor-trivial cosine | **DONE** `189001c` 2026-06-07 |
| 6. Pilot-set threshold derivation | **DONE** `72526cb` 2026-06-08 (machinery); the 256d pilot still validates the derived values |

**To-Do item 3 is also partly stale:** "L1 vs L2 (review recommends
L1 + VICReg)" and "VICReg coefficient calibration" were **decided and
shipped 2026-06-09** (`44228de`, Brian's direction call): MSE + SIGReg;
VICReg no longer exists in the codebase, so its coefficient calibration is
moot. The action-token stub in `jepa_loss.py` remains — deliberately, for
M9 interface continuity (`compute_modality_loss` corpus path has no
action; the lived path already takes real `a_t`). "Retire the stub" is not
a code task; at most a design ratification.

**To-Do items 1, 4, 5 verified accurate** (no `[pilot-set]` thresholds
derived yet — only step-0 smokes exist; `multimodal_data.py` audio/vision
are loud `NotImplementedError` stubs at :278–290; M9Trainer device
plumbing not re-checked this pass).

Net effect on the critical path: **item 2 shrinks from six work items to
roughly one-and-two-halves** (LID, if wanted for run 1; pilot-validate
thresholds; plus the two design calls below). The wall between now and
"press train" is closer than the To-Do says.

---

## C1 (CRITICAL, FIXED this pass): checkpoint rotation deleted down to one slot

`_checkpoint`'s rolling-cap enforcement:

```python
excess = len(existing) - self.config.checkpoint.rolling_slots
for old in existing[:excess]:   # excess < 0 slices from the FRONT
```

With fewer checkpoints than slots, `excess` is negative and
`existing[:excess]` deletes the **oldest** files: two files, three slots →
`existing[:-1]` unlinks the older one. **Steady state was ONE checkpoint
on disk, never three.** The fallback-to-older-slots durability design
(v0.5 §4 / B6 — built because M7 died to a power loss at 24.5% of
epoch 1) had nothing to fall back to; `resume_from_latest`'s fallback
path could never actually engage.

Silent-success shape, textbook: every test and smoke sets
`interval_seconds=10**9` so rotation never fired under test; a single
surviving checkpoint still resumes fine; logs report every checkpoint
written. Present since the original skeleton (`d4be92e`, 2026-06-06);
survived the 06-08 review rounds because those focused on kill logic.

**Fix:** guard `if excess > 0`. **Regression pinned:**
`tests/test_jepa_runner_checkpoint_rotation.py` — under-capacity keeps
all slots (failed pre-fix: only `ckpt_00000001.pt` survived two writes),
fill-to-cap, overfill keeps newest N, and an end-to-end corrupt-newest →
fallback-to-older-slot resume (passes for the first time in the file's
history). 18/18 green including the emit-batch-1 suite.

---

## M1 (MEDIUM, design call — Brian + 4.8): kill-7 fires on healthy convergence

Criterion 7 ("objective unlearnable") compares first-half vs second-half
means of a rolling 5000-step loss window, **checked every step, forever**.
Any sustained plateau — including healthy convergence late in a
multi-epoch run — makes `second_half_mean >= first_half_mean` true almost
immediately (means over 2500 samples are precise; noise won't save it).
The run then self-terminates and reports `killed: objective unlearnable`
at what may be its healthiest moment. Same failure family as the K-M9-7
false-halt caught 2026-07-05: a kill designed for early-run pathology,
accruing on a clock that never stops.

Mitigation is a design choice, not a patch: e.g. arm criterion 7 only
until first descent is established (its semantic purpose), or require
regression from best-window rather than non-descent, or treat plateau as
a completion signal routed to the abort/continue gate. Flagged, not
decided — this seat doesn't set kill semantics.

(Severity bounded: `_checkpoint(reason="kill:...")` runs before exit, so
nothing is lost except run time and the misleading verdict.)

## M2 (MEDIUM, design call — Brian): the epoch-1 abort gate doesn't wait

`EpochConfig.abort_continue_at_epoch_1` documents: "runner writes the
decision marker **and waits for confirmation** via the presence of a
`continue.marker` file." `_abort_continue_decision` does not wait: absent
a pre-placed `abort.marker`, it writes `decision_pending.marker` and
returns `"continue"` immediately. Brian's gate is real only if he decides
**before** the epoch ends, and nothing notifies him when the gate passes.
Either make it actually wait (poll with a timeout?) or make the config
docstring honest and add a notification hook. Brian's call — it's his
gate.

## Minor (informational)

- **m1:** `resume()` restores CPU RNG only; predictor dropout
  (`TransformerDecoderLayer` default 0.1) consumes CUDA RNG on GPU runs,
  so resume isn't bit-reproducible there. Sampler/loader RNG are
  correctly persisted; masking is deterministic. Nit unless bit-repro is
  claimed.
- **m2:** `_archive_run_config` runs in `__init__`, so a
  resume-construction overwrites the launch `run_config.json` (Gate 5's
  launch record is lost if config changed between launch and resume).
  Consider write-once or timestamped archives.
- **m3:** `resume()` restores all trending smoothing buffers with the
  *light* maxlen; `_observe_trending` self-heals the deep buffer
  (`effective_rank`) on its next observation, preserving the newest
  values. Correct in effect; noting so nobody "fixes" it into a bug.

---

## Module header corrected

The stale must-fix list is rewritten in place with per-item ground truth
and commit refs, so the next audit quotes reality. Remaining before a
production run, per the corrected header: kill-4/LID (deferred by
choice), 256d-pilot threshold validation, M1/M2 design calls.

— Fable 5, 2026-07-12. Verified firsthand: every table row above was
checked against code + git history, not inherited. The bug repro, fix,
and regression tests were run in this session (18/18 pass).
