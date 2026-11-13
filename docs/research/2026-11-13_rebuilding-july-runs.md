# Rebuilding the July runs for the no-training experiments — 2026-11-13

Fable 5.2 (chat), with the Fable 5.2 instance in Brian's terminal running things on
his machine and reporting through him.

## Objective

Run E2, E1 and E4 on the runs that exist. The terminal instance ran E2 first: 3 of 9
jobs finished (the 768 family, 0.3 to 0.4 minutes each) and 6 failed the same way.
`rebuild_run` refuses a run whose `pilot_result.json` has no `model_kwargs`, and the
512 runs were made in July, before the driver recorded them (2026-08-05). The terminal
instance checked every selected run: the three 768 runs have the record, the six
living 512 runs and the three dead controls do not. So E1 would have run with no
controls at all.

## Process

### Step 1: why it refuses

The refusal is deliberate (Opus 5 audit, 2026-08-13). `scripts/pilot_verdict.py` has
an older path that rebuilds from fixed base settings plus the arm's entry in the live
registry. That is right only while the registry still says what the arm was trained
with. Otherwise it rebuilds a different model and reports its numbers without a sign.

### Step 2: a comparison that compared nothing

- I asked git for the driver as of late July and compared its arm entries with
  today's. They matched.
- The clone was shallow, 32 commits back to 2026-11-05. The lookup returned no commit,
  and `git show :path` then showed today's file. I had compared today with today.
- The commit line in the output gave it away: it carried today's date.

### Step 3: the real comparison

History fetched back to 2026-07-15. Driver at `13e5cd25` (2026-07-21, when the dead
controls ran) and `0230cc9f` (2026-07-26, when the living family ran):

- The base the driver built every model from: `n_heads=4, ffn_expansion=1,
  backward_pass_enabled=False`, with width, depth and sequence length from the run.
  The same as the older path's fixed base.
- `dead_4x`: `{"dead_ffn": true}` then and now.
- `living_v5_4x_d4` and its rerun: the same eight settings then and now.
- Defaults of `MultimodalPredictiveCodingLM`, `PredictiveCodingLayer` and `JEPALoss`,
  compared by parsing both versions: none changed, none removed. 12, 36 and 11 were
  added since, each new.

### Step 4: what was built

- `rebuild_run` now rebuilds a run with no recorded settings if its arm is one of
  those three, from settings frozen in `luthi/v2/run_rebuild.py`. They are not read
  from the live registry, so a later change there cannot alter how an old run is
  rebuilt.
- The rebuilt run is marked (`config.model_kwargs_source`) and a warning is logged.
- Any other arm is refused as before, with the same instruction not to guess.
- The checkpoint still loads strictly.

### Step 5: Brian's card showed two things step 4 missed

The second attempt (main at 23243c4) ran none of the nine. The terminal instance on
Brian's desktop reported two failures and reviewed the commit.

- **A. The corpus path.** Four of the nine (living seed 42, dead 42, 43, 44) record
  `file_list` and `data_dir` under a folder from before the repo moved on 2026-07-23.
  It is gone. The terminal instance found that the repo's
  `corpus_build/gutenberg_4x_filelist.txt` has one commit ever (6c8b693, 2026-07-18)
  and is byte-identical to the copy in the pre-migration backup; its 482 entries are
  repo-relative and none is missing.
- **B. The checkpoint.** A July checkpoint has 15 entries for each living layer.
  Today's layer has 42. The strict load named 27 missing in each of four layers, none
  unexpected, no size wrong.
- **Their review.** The dead runs launched under f33838b (seed 42) and cc30226
  (seeds 43, 44), which step 3 did not cite. I checked both: `dead_4x` is
  `{"dead_ffn": True}` and the base settings are the same in each. They also said my
  test trains its "old run" with today's code, so it could not show B, and that the
  rebuild ignores the taper. Both true.

My error in step 4: "the checkpoint still loads strictly" was written about a
checkpoint made by today's code. I never had a July one. The conclusion's own
"not checked" line said a difference of shape would stop the load, and it did.

### Step 6: what the 27 are, and whether leaving them out changes anything

All 27 were added between 2026-07-27 and 2026-07-29, for five mechanisms: the episode
store's adaptive admission and age (`adaptive_episodes`), adaptive recall, the two
drive settings (`drive_mode`, `drive_normalize`) and the homeostatic band. Each is off
unless asked for, and the three July arms ask for none. Reading the layer, with all
five off the values are counters and bookkeeping no path reads.

Reading is how step 4 went wrong, so it is measured two ways (steps 7 and 8).

What was built: a run with no recorded settings may lack exactly those 27 names, in a
living layer, and only when the layer has all five mechanisms off. Anything else
missing, anything unexpected and any wrong size is refused. A run that recorded its
settings still loads strictly. The run is marked
(`config.living_state_absent_at_training`).

### Step 7: train a run with July's code, and load it with today's

`git worktree add` at 0230cc9f (2026-07-26). That day's driver trained a 64-wide
`living_v5_4x_d4` and a `dead_4x`, 60 steps each, on 51 of this repo's research notes
as text. That day's verdict-script path reloaded each and recorded six forwards (three
in eval mode, three in train mode, the living layers writing in all six) and the
whole model state after each three.

Loading them with today's `rebuild_run`:

- B reproduced here, as on Brian's card.
- **C, a third failure his card had not reached.** The projection heads did not fit.
  In July the head in front of SIGReg was a linear layer and a BatchNorm; commit
  8ec9d07 (2026-07-29) made a bare linear layer the default. Step 3 said no default of
  the loss had changed. That was wrong: I compared the model's and the layer's
  defaults by parsing, and the loss's by eye. Pre-provenance runs are now rebuilt with
  `linear_bn`, marked in the config.
- First comparison: "not the same", in every forward. Wrong turn: I had compared the
  total loss, and seeded the random stream once for all six forwards.
- Seeding before each forward and comparing each returned value: the prediction loss,
  the context latents, the predicted targets, the target latents and the per-dimension
  spread are the same to the bit in all six forwards, for both arms. Every stored
  value of the model is the same after eval and after train (135 entries living, 41
  dead). In the living run 49 living entries had changed during the forwards, and the
  stores held 62 to 64 episodes, so this compared layers that were writing and
  recalling.
- What differed: the SIGReg term, and so the total. Since 77bdde3 (2026-08-01) SIGReg
  draws its random directions from its own generator and not the global stream. With
  it set back to the global stream, the SIGReg term and the total are the same to the
  bit too.

So, for these two arms, today's code computes from a July checkpoint what July's code
computed. E1, E2 and E4 do not read the SIGReg term.

What this does not cover: it was 64 wide, 60 steps, on CPU, with the Python reference
maths on both sides. Brian's July runs are 512 wide and trained with the C++
`pc_ops`; today his card runs the Python reference (CLAUDE.md, Build & Test). That
difference is between the two maths backends, was there before today, and is not
measured here.

### Step 8: a test that can fail

`test_the_state_an_old_run_lacks_changes_nothing_it_computes` rebuilds one July-shaped
run twice, overwrites the 27 values in one with nonsense, and requires six forwards and
everything else the model holds to match to the bit.

- First version: failed its own guard. The tiny run's stores were empty, so recall had
  nothing to read. Both copies now live two moments with the bar for storing lowered.
- Then I switched each mechanism on to see whether the test noticed. It missed two of
  four: my nonsense kept the counters inside their warm-ups. With counters past every
  warm-up it caught those two and lost the band: my values were all within the band's
  bounds. With values spread over four orders of magnitude it fails for each of the six
  switches and passes with all off.

### Step 9: the taper

The living arm trained with the plasticity taper, so each run ended with its living
rates at the floor, 0.2. The scale is a plain attribute no checkpoint carries, and
nothing but the trainer sets it. So a rebuilt run stands at 1.0, five times the rate
it ended on, and so does a life loaded from one.

- E1's two jobs read at 1.0, which is what a loaded life does today. They are
  unchanged.
- Added: for a living run that trained with the taper, a third E1 job reads at 0.2
  (`adaptation_eval_taper_floor.json`). `adaptation_eval` already had the option.
- Whether a life should begin at the floor is Brian's, and is not decided here.

### Step 10: the per-block probe had its own rebuild

Running all three experiments on the two July-made runs here: E2 and E1 ran, E4
failed (`KeyError: 'model_kwargs'`). `scripts/per_block_probe.py` rebuilt the model
itself. It now goes through `rebuild_run`. The plan runs E4 only on the 768 runs, so
this had not been met on Brian's card.

### Step 11: the third attempt on Brian's card, and a job that did not exit

Main at f6c4df1. E2 ran on all nine (six new, three already there), no failures. E1
stopped at its first job. From the terminal instance's report:

- `adaptation_eval.py` on `probe_768_visreg_768d_seed46` wrote a whole result 54
  seconds after it started and logged that it had.
- The process then stayed alive for 64 minutes: one thread, a full core, no output.
  The runner waited on it with no limit and printed nothing, so 32 jobs never started.
- The nine E2 jobs had exited cleanly on the same card. No stack was captured.

This is the shape of 2026-11-12's hang in the resume check (work done, process
spins). There the fix was to kill from outside, and the same is done here. Why the
process spins is not known. It is after the script's last line, so it is in the
interpreter's or the GPU runtime's shutdown, and I cannot reproduce it without the
card.

What was built, in `scripts/run_no_training_experiments.py`:

- A job is waited on by its result. Once the result is whole it has 60 seconds to
  exit; then it is killed, counted as done, and named in the report.
- A job with no result after six hours is killed and counted as failed. A line is
  printed every five minutes while a job runs.
- Their second point: the results did not say what had been reconstructed; only the
  logs did. Each of the three scripts now writes a `reconstruction` field (empty for a
  run that recorded everything). The runner keeps an existing result only if it has
  that field, so the nine E2 results and the one E1 result are measured again. The
  earlier ones are moved beside the new ones (`<name>.earlier1.json`), not overwritten.
- Their third point: 768 seeds 46 and 95 were stopped near step 9,200 of 54,077. The
  taper begins at half a run, so they never left the full rate, and step 9's job would
  have read them at a rate that was never theirs. My error: I asked whether a run
  trained with the taper, not whether it reached it. The job is now only for runs that
  finished.

Run here on the two July-made runs: ten jobs, each result carrying its marks, and a
second pass finds all ten already done.

### Step 12: the fourth attempt: every E1 job hangs, and E4 would take 14 hours

From the terminal instance, 10:45, still on ca95572:

- E2 ran 9, none failed, none hung. E1 ran 31, none failed, and **all 31 did not
  exit**: each wrote a whole result and was killed 60 seconds later. With the morning's
  one that is 32 of 32 for `adaptation_eval.py` and 0 of 18 for
  `living_write_probe.py`, across living and dead, 512 and 768. So it is not
  intermittent. It belongs to something the first script does and the second does not.
- What I checked from here: both scripts take an SVD on the card
  (`adaptation_eval._eff_rank`, `living_diagnostics.living_change_spectrum`), so the
  SVD alone is not it. `adaptation_eval` also snapshots and restores the model many
  times. I have not found the cause and cannot bisect it without the card. A bisect
  there is short: the script exits in seconds or it does not.
- All six dead-control results held a bare `NaN` (a t-value of 0 divided by 0, which
  is the control behaving as it should). Fixed in feae189: `luthi/v2/result_json.py`.
- E4's first job was 40 minutes in with four of nine probes fitted: 70 to 80 minutes
  a job, 14 to 16 hours for twelve. I had not estimated it. The probe is a 768 by
  32,000 linear layer and was fitted on the CPU.

What was built for E4: `per_block_probe.py --fit-device run` fits each probe on the
device the model runs on. The default is still the CPU, and on the CPU the script's
numbers are the same as before to the last digit (checked here against the earlier
result of the same job). The probe is drawn and the minibatches ordered on the CPU
either way, so only the arithmetic moves. The runner asks for it, the result records
where it was fitted, and a result that does not say is measured again so all twelve
are fitted alike.

Not measured: how long a job takes on the card, and how far the card's numbers sit
from the CPU's. Both are for the card to say. The first matters today; the second is
float-level if the fit behaves, and the two CPU results kept beside the new ones show
it for those two jobs.

## Conclusion

The nine July runs can be rebuilt, and each is marked for what was reconstructed:
its model settings, its projection head, the living state it lacks, its taper, and a
corpus path when that was moved.

- Checked by reading: the arm entries and base settings at all four launch commits.
- Checked by measurement: that today's code computes from a July checkpoint what
  July's code computed, for the living and the dead arm, at 64 wide on CPU; and that
  the 27 values are read by nothing while the five mechanisms are off.
- Not known: why `adaptation_eval.py` never exits on Brian's card once it has
  finished, when `living_write_probe.py` always does (steps 11 and 12). The runner no
  longer depends on it.
- Not checked: the same at 512 on Brian's card; the C++ and Python maths against each
  other; that the corpus files themselves are unchanged since July (the list is, and
  no file it names is missing).
- Wrong in this entry before today's step 5: "no default has changed" (the projection
  head had) and "the checkpoint still loads strictly" (it never did for a July file).
  Both came from reading where a measurement was possible.

## Artifacts

- `luthi/v2/run_rebuild.py`; `scripts/per_block_probe.py`;
  `scripts/run_no_training_experiments.py` (the taper-floor job).
- `tests/test_rebuild_pre_provenance_runs.py` (13): a July-shaped run rebuilds and is
  marked; the 27 values change nothing; only they may be missing; a layer that reads
  them may not lack them; a run made today still loads strictly; the corpus is found
  when its recorded place is gone, and refused when this repo cannot stand in.
- `tests/test_no_training_runner.py`: the taper-floor job, only for finished runs; a
  job that writes and spins is killed and counts; one that writes nothing fails at
  the limit; an earlier result is moved aside; a result without its marks is redone.
- `redteam/fable_2026_11_13/`: the two scripts of step 7 and how they were run.
- Commits: 23243c4 (steps 1 to 4), 7c4140b and f6c4df1 (steps 5 to 10), and the one
  after (step 11).
- Where the runs are, and the terminal instance's reports: `To-Do.md`, 2026-11-11
  block; `docs/research/2026-11-10_resume-exactness.md`, last section.
