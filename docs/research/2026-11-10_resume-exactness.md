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

## Conclusion

- **Uncapped runs, which is how registered families launch, already resumed exactly** on CPU, through the full driver, on the registered living arm.
- **Capped runs did not.** Any capped run that was interrupted and resumed before today trained past its cap, by the step count of the checkpoint it resumed from. Several August families were capped at 3,000 or 6,000 (the depth-8 probes, the LLM-JEPA family, the first VISReg family).
  - To check an old capped run: its final `global_step` should equal its cap. If it is larger, the run was resumed and ran long. `supervisor.log` in `runs/jepa_pilot/` also records each resume.
  - I have not checked any. The runs are on E:.
- **What this does not show:** anything about the GPU. ROCm kernels are not guaranteed to repeat bit for bit, so on the card two uninterrupted runs may already differ. `scripts/check_resume_on_device.py` measures that noise and then asks whether a resumed run differs by more. Run it before a long family.
- What it also does not cover: a crash in the middle of writing a checkpoint (the atomic rename and the fallback to older slots are tested elsewhere), multi-epoch runs, and a life's own checkpoints.

## Artifacts

- Code: `luthi/v2/jepa_runner.py` (`_steps_this_epoch`, `_epoch_cap_reached`).
- Tests: `tests/test_resume_matches_uninterrupted.py`, `tests/test_check_resume_verdict.py`.
- Script: `scripts/check_resume_on_device.py`.
