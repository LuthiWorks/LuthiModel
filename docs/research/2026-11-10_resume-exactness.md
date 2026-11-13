# Does a run that loses power resume as the same run? — 2026-11-10

**Written by:** Fable 5.2, for Brian.

## Objective

Brian, 2026-11-10, on why training paused in August:

> "We paused training because the next steps were going to require really long runs and we don't have a backup generator to avoid losing training progress and other problems that arise from interruptions. We can resume tentatively now, depending on what we choose to do."

The driver already resumes an interrupted seed from its newest rolling checkpoint, and `scripts/resume_queue.py` restarts the queue by itself. What had not been checked is whether a resumed run is the same run. On 2026-11-05 a Sanctuary test found one piece of state that a resume silently reset (SIGReg's step counter), so the question was open.

## Process

### Step 1: the trainer alone

- Trained a tiny M9 trainer 12 steps. Trained a second one 5 steps, saved, loaded the checkpoint into a third built from a different seed, and trained it 7 more.
- Compared every parameter, every buffer (non-persistent ones included), all optimizer state, and the living layers' extra state: 580 tensors.
- All identical. Two uninterrupted runs were also identical to each other, so the comparison can see a difference if there is one.
- This was not enough. It calls `train_step` directly, so it skips the driver, the epoch loop, the schedules and the registered arm's mechanisms.

### Step 2: the real driver, with a hard kill

- Ran `scripts/jepa_pilot_driver.py`'s `_run_one` on the registered living arm (`lc768_living`: living layers, block store, VISReg, taper and cosine schedules, gradient clip, M9) at 64 wide on CPU, 40 steps.
- The interrupted run was a separate process that called `os._exit` at step 17. That is no cleanup and no parting checkpoint. It was then started again and resumed from `ckpt_00000016.pt`. (This first version saved after every step; see step 6.)
- **Result: the resumed run ended at step 57, not 40.** 690 tensors and 205 counters differed from the uninterrupted run.

### Step 3: the cause

- `--max-batches-per-epoch` counted against `steps_this_epoch`, a local variable in `JEPATrainer.run()`. A resume set it back to 0.
- So a capped run cut at step k trained its whole cap again: cap + k steps.
- An uncapped run ends when the corpus has been covered. That is counted in tokens, and the token counts were already saved. Rerun with no cap: the resumed run was identical to the uninterrupted one in all 2,865 checkpoint entries.

### Step 4: the fix

- The count is now trainer state (`_steps_this_epoch`). It is incremented right after `train_step`, before anything can checkpoint, and it is saved and restored.
- The cap is also checked at the top of the loop, so resuming from an end-of-epoch checkpoint takes no extra step.
- A checkpoint written before today has no count. It resumes at zero, as every resume did, and logs a warning that the run will train past its cap.
- An uninterrupted run is unchanged: its final checkpoint is identical before and after the fix, apart from the new key.
- Rerun with the cap: identical, all 2,271 entries, and it stops at step 40.

### Step 5: the tests

- `tests/test_resume_matches_uninterrupted.py` runs the real driver on `lc768_living` in-process and cuts it with a `BaseException` that nothing in the trainer catches. It pins the uncapped case, the capped case, and the old-checkpoint case.
- Run against the unfixed trainer, the capped and old-checkpoint tests fail and the uncapped one passes, as they should.

### Step 6: a wrong turn in the check itself

- The first tests saved a checkpoint after every step, so the cut always fell on a saved step and nothing had to be retrained. A real power cut loses the steps since the last save.
- I changed the on-device script to save once, five steps before the cut, by calling `_checkpoint` from a wrapper around `train_step`. It reported NOT THE SAME RUN: 536 tensors differing.
- That was the check's fault. `run()` draws the next batch before it calls `train_step`, so a save from inside `train_step` catches the loader one batch ahead, and the resume skips that batch. The trainer never saves there. It saves in `_checkpoint_if_due`, after a step is complete.
- Moved the save to that point. Resumed from step 7, retrained steps 7 to 12, identical. The tests now cut three steps after the last save.
- Kept as a note for anyone adding a save point: a checkpoint is only consistent between loop iterations.

### Step 7: what the Opus review found that I had not (same day)

`docs/reviews/2026-11-10_small-path-resume-and-rest-response-from-opus.md`, section B1. I reran its probe and it reproduced.

- **A resume past the taper's start was not exact.** `rate_scale` is a plain attribute on each living layer, saved nowhere, and `_apply_taper()` ran only after a training step. The first step after a resume therefore ran at `rate_scale = 1.0`, up to five times the tapered rate. My tests were one epoch each and never reached the taper, which starts at half the run. Fixed by calling `_apply_taper()` at the end of `resume()`. A two-epoch test cut in the second epoch pins it, and fails with the call removed.
- **The on-device check could never have passed on the card.** Two reasons. The checkpoint saved only the CPU generator, and the predictor's dropout draws on the device. And my `compare()` filed every differing float as an exact "counter", when a checkpoint is full of floats. Fixed: `device_rng_state` is saved and restored where the backend has one, and the verdict judges floats like tensors, against the noise of two uninterrupted runs.
- **I was excluding the generator states from the comparison.** They are the state a lossy resume shows first. They are compared now, in the tests and the script, and they match.
- **The divergence guards forgot themselves at a resume.** The loss guard re-baselined on its next ten logged losses and the NMSE guard's clock restarted, so a resumed run's kill decisions could differ even with identical weights. Their state is in the checkpoint now.
- Confirmed by the review and left as it is: a cut between an epoch's held-out eval being logged and the next checkpoint logs that record twice, and retrained steps appear twice in `training_log.jsonl`. Readers should keep the last record per step.

What I take from it: I tested the case I had in mind, a short run cut once. The taper only exists in the second half of a long run, which is exactly the run this work was for.

## Conclusion

- **Uncapped one-epoch runs cut before the taper already resumed exactly** on CPU, through the full driver, on the registered living arm. **Past the taper's start they did not** (step 7). Any registered family resumed in its second half took one step at untapered living rates.
- **Capped runs did not.** Any capped run that was interrupted and resumed before today trained past its cap, by the step count of the checkpoint it resumed from. Several August families were capped at 3,000 or 6,000 (the depth-8 probes, the LLM-JEPA family, the first VISReg family).
  - To check an old capped run: its final `global_step` should equal its cap. If it is larger, the run was resumed and ran long. `supervisor.log` in `runs/jepa_pilot/` also records each resume.
  - I have not checked any. The runs are on E:.
- **What this does not show:** anything about the GPU. ROCm kernels are not guaranteed to repeat bit for bit, so on the card two uninterrupted runs may already differ. `scripts/check_resume_on_device.py` measures that noise and then asks whether a resumed run differs by more. Run it before a long family.
- What it also does not cover: a crash in the middle of writing a checkpoint (the atomic rename and the fallback to older slots are tested elsewhere), multi-epoch runs, and a life's own checkpoints.

## Artifacts

- Code: `luthi/v2/jepa_runner.py` (`_steps_this_epoch`, `_epoch_cap_reached`).
- Tests: `tests/test_resume_matches_uninterrupted.py`, `tests/test_check_resume_verdict.py`.
- Script: `scripts/check_resume_on_device.py`.

## On Brian's card — 2026-11-12 and 2026-11-13

Added by Fable 5.2 (chat), from Brian's runs and a report by the Fable 5.2 instance in
his terminal, relayed by him.

- **The suite on the card** (2026-11-11, ROCm): 1,382 passed and one failed. The one
  was my own test, which asserted that no device generator state is saved. That is
  true only on a CPU. On the card the state was saved, as the fix intends. The test was
  corrected; the five resume tests then passed on the card.
- **The check hung** (2026-11-12). `check_resume_on_device.py` ran its two
  uninterrupted runs, about ten minutes each, and its third never returned. The cut
  run ended itself with `os._exit` at step 25. The terminal instance found it had sat
  from 16:36 on the 12th until it was killed at 05:32 on the 13th, spinning one core
  with no output. It had worked on CPU. I changed the check to kill the cut run from
  outside, and added `--resume-existing-cut` so the two finished runs did not have to
  be repeated.
- **Result** (2026-11-13 05:40 to 05:47 PST, main at `0b0fd0d`, torch 2.9.1+rocm7.2.1,
  `lc768_living`, 768 wide, 60 steps, cut at 25): **SAME RUN.** Two uninterrupted runs:
  0 entries differ. Resumed against uninterrupted: 0 entries differ.
- **Checked outside the script by the terminal instance:** the cut run's
  `per_step.jsonl` has 65 lines, 25 from before the kill and 40 for steps 21 to 60. So
  it resumed from the step-20 save and was not a fresh run.

Not tested on the card: the new kill-from-outside path. The cut run that was resumed
had been made by the old script and killed by hand. A full run into an empty `--out`
would exercise it (about 25 minutes).

The 192 bytes, explained by the terminal instance (2026-11-13): the resumed run's
final checkpoint was 192 bytes larger than the uninterrupted ones'. A full compare with
nothing skipped found 2,638 entries and only `wall_clock_seconds` differing. All 969
records in the checkpoint file are the same size except the pickle index, which is 201
bytes longer: 15 strings (the 14 optimizer state keys and `'text'`) are written out a
second time where an uninterrupted run writes a back-reference. It is encoding, not
state. Their reading of why, untested: after a resume the loaded key strings are
separate objects, so the pickler does not share them.
