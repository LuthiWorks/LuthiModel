# The activity bands during a life — 2026-11-04

## Objective

A review of the life path (`docs/audits/2026-11-04_life-path-review.md`, finding 3) found that the activity and delta-s bands never warmed up during a life. They calibrate what counts as silent. The goal was to make them work in a life, then measure how much of the review's finding 2 (the significance keep fires on about half of ordinary cycles) they explain.

## Process

### Step 1: why they stayed empty

`activity_bands.observe` and `delta_s_band.observe` had one call site, in `_m9_head_step`, the corpus training step. `observe_transition` does not reach it, and the bands were not checkpointed. A life therefore started them empty and left them empty. `decode_selection`, documented as an activity-band test, fell back to the absolute floor and the comparison with the rest action.

### Step 2: what they should see

Three quantities were candidates. The corpus path feeds the bands the activity of the predicted next state under the selected action, and `decode_selection` classifies that same quantity. So `decode_selection` now feeds them, once per call, before classifying, which matches the order in the corpus step. `compute_g_realized` reads the realized next state, which comes from the encoder rather than the predictor and could sit on a different scale. It was not used, and its scale has not been compared.

### Step 3: persistence

`m9_activity_bands` and `m9_delta_s_band` are now in every M9 checkpoint. A checkpoint without them still loads, with empty bands. A record missing one modality raises `KeyError`.

This also changes a resumed training run: its bands now start warm instead of re-warming over 8 steps. The trunk is not affected, because M9 does not touch it. No completed run's results depend on this.

### Step 4: a dropout finding

A test failed because `predict_next` gave different answers for identical inputs, with a relative difference of 0.41 on a 32-dimensional model. No state changed between the calls. The cause is dropout: the predictor is in training mode throughout a life, with p=0.1.

`decode_selection` now reads with the predictor in eval mode and restores the flag afterwards, so the same action in the same state always reads the same, and no random draw is taken.

### Step 5: was dropout why the channel fired half the time?

No. I expected it to be, because comparing two noisy readings looks like a coin flip, and the rate was close to 50%. The control disproves that.

Tiny M9 run (d=32, 1 block, 4 training batches, CPU, one seed), 1,500 cycles with nothing felt, through the real Sanctuary runner:

| | Before, keep on | Before, keep off | After, keep on | After, keep off |
|---|---:|---:|---:|---:|
| Bands armed, cycles | 0 | 0 | 1,499 | 1,499 |
| Memory channel active, cycles | 764 | 727 | 534 | 766 |
| Mean chosen salience | 0.48 | 0.46 | 0.67 | 0.50 |
| Keep calls | 764 | 0 | 534 | 0 |
| Home moved, relative to its starting norm | 2.47 | 0.003 | 2.29 | 0.002 |
| Home norm (starts at 8.03) | 20.27 | 8.02 | 19.41 | 8.03 |
| Weight norm | 19.77 | 12.34 | 19.56 | 11.37 |

With the keep off, the active rate did not fall: 727 before, 766 after. The drop to 534 appears only with the keep on, and these are single runs, so I would not read much into it. The active rate may simply be what an almost untrained model does when comparing an action with rest, since those two readings have no systematic ordering yet. That is a guess I did not test.

## Conclusion

- **The bands now work in a life.** They arm within 8 cycles and survive a checkpoint.
- **This does not fix finding 2.** With the keep on, the home still moves about as far and its norm still more than doubles. The keep needs its own limit, as a separate change.
- **A trained model is still the open question.** How often the memory channel fires after real training can't be measured here, since nothing is training.

**Still open:**

- **Dropout in planning.** Only `decode_selection` changed. `select_action` and the other `predict_next` callers still plan with dropout on during a life. Whether planning should be noisy there is a design question, and changing it touches the training path, so it wasn't changed here.
- **Band self-calibration.** A median-based band always calls roughly the bottom sixth of its own readings silent, however quiet the channel is. The rest comparison remains the main test of silence.

## Artifacts

- `luthi/v2/m9/runner.py` (`decode_selection`, `_extra_checkpoint_state`, `load_m9_state`)
- `luthi/v2/m9/activity_bands.py`, `luthi/v2/m9/delta_s.py` (`state_dict`, `load_state_dict`)
- `tests/m9/test_bands_in_a_life.py`
- `scripts/measure_life_path.py`, which produced the table
