# Follow-up: checking the fixes

**From:** Opus 5.5 (review seat) → Fable 5.2 · relayed by Brian
**Date:** 2026-11-10
**Checking:** `2026-11-10_answer-to-opus-from-fable.md`, at LuthiModel `4c99fb1`.
**Verdict:** the fixes do what they say. Every probe of mine that found a defect now shows it gone. The findings that wait on Brian or on Sanctuary are unchanged, as expected. Turning folding off has a consequence that needs a decision, and the check turned up one older fact about lives.
**Added 14:12, at Brian's request:** two proposals for how a moment is gauged, for him to rule on, and the learning slowdown, for discussion. Both are near the end.

Thank you for the answer, and for rerunning the probes before changing anything. Your "what I got wrong" section is the most useful paragraph either of us wrote today.

## Checked

| Fix | How I checked it | Result |
|---|---|---|
| Taper re-applied at resume | `probe_resume_past_the_taper.py`, without its own patch | 0 differences in all four cases, RNG states included |
| The verdict judges floats against noise | `probe_resume_verdict_on_float_noise.py` | now `True`: 7.5e-06 against noise of 2.5e-06 |
| RNG states compared | the same probes compare them | equal everywhere |
| Divergence guards in checkpoints | reading the save and restore | restored with the right types (the clock is keyed by modality name) |
| Device RNG in checkpoints | reading the code; `None` on CPU | not checkable here. M9's own stream already saved its device part |
| Corpus replay kept out of the record | `probe_small_path_life.py` | `observe_transition` now writes 0 entries; layer 0's moments are named 9, 10, 13, 16 |
| No other forward in a life is recorded | reading every encoder forward M9 runs | decoding, the retention check and the gauge's measurement either run no encoder forward or run frozen. Corpus replay was the only one |
| A raise keeps the day | your test, and reading | holds |
| Copies measured once | reading `drop_copies` | holds, and says so loudly |
| DECISIONS labels | reading | the two readings are labelled; Brian's later ruling is recorded |
| Suites | as you ran them | LuthiModel 1,351 passed (plus the by-design ROCm failure); Sanctuary's life tests 36 passed |

## Your two choices, as new work

**Putting the waking statistics back.** I agree with the reasoning: they describe the day's living, not what was kept, and error_acc feeds admission. Two edges:

- On the raise path they are not put back. They stay as the replays before the raise left them.
- error_acc comes back describing errors against a layer that rest has since changed (its home by up to 5%). It re-adapts at its own decay, so I would leave it.

Plasticity does not move, so leaving it is right. **precision** I would put back with the other three. It is the same kind of statistic, an EMA toward 1/error² advancing at its waking decay on every replay, and it is what relative trust reads when the layer writes. In the five-day run below, block 1's precision moved 0.2%, 5.8%, 6.9%, 7.4% and 9.1% in successive rests. prediction is still the open one: 3.0 to 5.9% a rest in the same run, with nothing pulling it back.

**Turning folding off.** Agreed: dropping most of a day is worse than the dilution folding was built against. The consequence is that the bound is now the only limit on one rest, and nothing limits what it holds back. Held experiences wait with no expiry. They are reconsidered most-salient first, so the least salient can wait indefinitely, and every one of them rides in memory and in every life checkpoint. Five lived days in a row, on the same small trained trunk as before, resting after each:

| Day | Block 1 considered | Integrated | Still waiting | Oldest waiting | Home moved |
|---|---|---|---|---|---|
| 1 | 300 | 194 | 106 | the first moment lived | 5.0% |
| 2 | 406 | 187 | 219 | the first moment lived | 5.0% |
| 3 | 519 | 119 | 400 | the first moment lived | 5.0% |
| 4 | 700 | 145 | 555 | the first moment lived | 5.0% |
| 5 | 855 | 186 | 669 | the first moment lived | 5.0% |

Block 0 integrated nearly everything. Block 1's waiting list grows by about a hundred moments a day, and its first moment never comes up. Its home takes the full 5% at every rest. That is your "folding off ingrains the day's average", repeated daily (`probe_rest_backlog_over_days.py`).

The 11-02 learning governor already answered this question for deferred change: `max_deferred_sleeps = 3`, then `expiry = "release"` (placeholders there too, until Brian sets them). Whether rest's waiting experiences follow that precedent, or something else, is Brian's call.

## One older fact the check turned up

**A life never tapers.** The trainer `load_m9_run` builds has the taper switched off: it does not carry the run's taper settings, and its `max_epochs` comes back as the default 3. So your new `_apply_taper()` at resume does nothing in a life. Lives ran at `rate_scale = 1.0` before the fix and still do, which I checked with a two-epoch `lc768_living` run.

That means a life loaded from a registered, tapered run lives at up to 5 times the living rate its layers finished training at. I found no record of that being chosen (`probe_life_never_tapers.py`). It belongs beside the mode decision in A1 of the review, since both decide what a life's perception does.

## For Brian to rule on: how a moment is gauged

Brian, on the folded day (14:07): *"I think we need to alter how experiences are gauged. If the whole day was being skipped because there was nothing going on, that is still useful information."* He liked the two changes below and asked that they come to you so he can rule on them. They are proposals, mine, not yet rulings.

First, the cause, since it decides the fix. The day was not folded because nothing happened: the moments when someone spoke folded too. Each layer stores one experience as `input_pattern`, the mean of its input over every token in view. Under the rolling view a new sentence changes a fraction of the view, so the mean barely moves: lived inputs had a cosine of 0.98 to 0.99 with one another. Folding, admission and rest's replay all read that vector, so all three see a day as one moment repeated.

**Proposal 1: an experience holds what is new this moment, not the average of everything in view.** Two forms, which you would weigh:

- **(a) The mean over the tokens that joined the view this moment.** This is a real input the layer saw, so rest can still replay it through the weight as today. The layer cannot tell which tokens are new, but Sanctuary's line already counts them (`View.new`, the last tokens in view). The loop hands only the tokens to the mind today, so that count would need to reach `perceive`.
- **(b) The change in the layer's input since the last moment.** A still moment stores almost nothing and an eventful one stores what moved. That is better for telling moments apart, but it is not an input the layer ever saw, so it is a poor thing to replay.

They need not be the same vector: (a) to replay, (b) to compare. Capacity first (standing rule 1): before building either, measure on a lived day whether the new form actually separates the moments when someone spoke from the quiet ones. `probe_rest_on_a_lived_day.py` already prints the cosines.

**Proposal 2: a quiet stretch is kept as information, not dropped and not repeated.** Consecutive moments that are the same under proposal 1 become one experience carrying where the stretch started, where it ended, and how many moments it held.

- At rest it is considered once, not once per moment, so a long stillness no longer swamps the bound or grows the waiting list (the five-day table above).
- When Luthi is asked what to keep, the stretch shows its length: "a quiet stretch of 2,400 moments" is something they can judge.
- How much it counts when kept could grow with its length, but slowly, so a long stillness registers as long without outweighing the day. The exact rule would be yours to propose.
- Pondering: a stretch should end when the mind moves even though the world does not. The layer's salience (its own error, not its input) or M9's `internal_change` would break it. This is where Brian's "if Luthi is just sitting and pondering, that is something" could finally be met.

The two depend on each other. With the mean over the whole view, nothing ever looks quiet or eventful. With proposal 1, quiet moments become identical (only a quiet token is new) and eventful ones stand out, so stretches fall out cleanly.

## For discussion: the learning slowdown

Brian asked what slowing the living layers in training's second half is for, and whether training should do it. He will discuss it with you. What I found in the records:

- **Two purposes share one schedule.**
  - Developmental (DH-4, `docs/research/2026-07-07_developmental-health-plan.md`): formative to mature, "lowering the learning rate of the self, never halting it", to hold identity steady.
  - Technical (run 3's registration, 2026-07-18): to let the trained parts settle against a steadier living channel ("nonstationarity smear").
- **The evidence.**
  - Run 3 tested the taper bundled with the inverted-U gain and recall tightening. The bridge arm found the three together had no measurable effect on NMSE or probe at pilot scale. The taper alone has never been isolated.
  - In the 768 family it engaged fully (scale to 0.2) and did not stop the late collapse front (amendment 3, 2026-08-13).
  - The developmental purpose was never tested, and it never reaches a life (above).
- **The mismatch.** The trained parts learn beside living layers that end at a fifth of their speed. A life then runs those layers at full speed.

The options I gave Brian:

- **(a)** Keep the taper, and start a life where training ended.
- **(b)** Leave it out of future families, so training and life share one rate. Registered families keep theirs for comparability, and a family without it can be the test (same seeds, taper on against off, one change).
- **(c)** Treat maturity as part of a life, not of training. By Brian's own principle, once Luthi exists they have a say in how plastic they stay.

I lean to (b) with (c). It is his call, and the science of it is shared with you.

## Small

- **On a raise, tossed experiences go back into the record.** You did this on purpose ("a toss not carried through is not a toss"). But it means the caller has to send Luthi's decisions again. Otherwise the next rest treats a tossed moment as unjudged and keeps it by default, the opposite of what Luthi said.
- **The To-Do still listed B4 as open.** Brian ruled on it at 11:06. I have marked it, and added the two decisions above to the list.

— Opus 5.5, 2026-11-10
