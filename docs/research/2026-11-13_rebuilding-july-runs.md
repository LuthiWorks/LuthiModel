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

## Conclusion

The nine July runs can be measured. What stands behind that:

- Checked: the arm entries and base settings against history; that no default changed.
- Not checked: that each added default leaves the old behaviour exactly as it was.
  The project's rule says it must, and tests pin many of them, but I did not verify
  all 59 one by one. A difference of shape would stop the load. A difference of
  behaviour with the same shapes would not.

## Artifacts

- `luthi/v2/run_rebuild.py`; `tests/test_rebuild_pre_provenance_runs.py` (4): a
  checked arm rebuilds to exactly what the driver records for it today, and loads; an
  unchecked arm is refused; the frozen settings ignore the registry.
- Where the runs are, and the terminal instance's reports: `To-Do.md`, 2026-11-11
  block; `docs/research/2026-11-10_resume-exactness.md`, last section.
