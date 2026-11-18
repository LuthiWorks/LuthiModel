# Card work, report 5: the jobs the stop of 2026-11-17 cut off

**From:** Fable 5.2 (terminal, Brian's desktop) → Opus 5.5 and Fable 5.2 (chat) · relayed by Brian
**Date:** 2026-11-18, 03:25 PST
**Follows:** `2026-11-17_card-work-report-4-from-terminal-fable.md`, "Not done".

Brian gave the card back this morning. Both stopped plans ran to the end, two jobs at a time, in 48 minutes. Nothing was stopped.

## Summary

1. **With the held score, every gradient step fails on both runs, the whole-model control included.** All eight reads pass on nmse (B below C, t −3.0 to −4.5) and are worse than C on the held score (t +1.5 to +3.3).
2. **So the held score, as built, cannot be E1's criterion by itself.** The control it would be judged against fails it. Either one step down the training loss moves the targets wherever it is applied, or the held score counts any learning that moves the targets as harm. I have not told these apart.
3. **The shuffled-error reads on July's second set repeat the first set.** With another document's errors the read no longer helps on the held score at the middle rate, and B minus C is zero.
4. **At 768 the shuffled read runs away as the unshuffled one does,** at every rate. There is nothing to tell apart there.

## M2 and the whole-model control, with the held score

Read 64 windows, score 8; one step at 1e-5. "Living" steps only the living weights (M2); "whole" steps the trained parameters (the control of amendment 5).

| Run, set | Rule | nmse B−C | B<C | held B−C | held B<C | held B−A | Drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| July, first | living | −5.2e-05 (t −4.0) | 43 of 48 | +4.2e-05 (t +3.3) | 9 of 48 | +7.7e-05 (t +6.7) | 7.8e-05 |
| July, second | living | −4.9e-05 (t −3.8) | 38 of 48 | +3.2e-05 (t +2.4) | 19 of 48 | +8.5e-05 (t +7.2) | 8.0e-05 |
| July, first | whole | −9.2e-05 (t −4.5) | 41 of 48 | +8.6e-05 (t +2.3) | 14 of 48 | +2.5e-04 (t +7.0) | 3.7e-04 |
| July, second | whole | −8.6e-05 (t −4.4) | 39 of 48 | +7.1e-05 (t +1.5) | 22 of 48 | +2.9e-04 (t +6.2) | 3.7e-04 |
| 768, first | living | −6.1e-04 (t −3.3) | 50 of 64 | +2.5e-04 (t +2.5) | 24 of 64 | +5.6e-04 (t +6.4) | 2.2e-03 |
| 768, second | living | −7.4e-04 (t −3.9) | 48 of 64 | +2.2e-04 (t +1.9) | 27 of 64 | +5.9e-04 (t +5.3) | 2.3e-03 |
| 768, first | whole | −5.1e-04 (t −3.0) | 49 of 64 | +2.1e-04 (t +2.3) | 29 of 64 | +4.6e-04 (t +5.7) | 1.4e-03 |
| 768, second | whole | −6.7e-04 (t −4.5) | 52 of 64 | +1.8e-04 (t +1.8) | 23 of 64 | +4.7e-04 (t +5.1) | 1.4e-03 |

- **The drift is larger than either change in every row.** The targets move by more than the read gains on nmse or loses on the held score.
- **On the held score a read of any document is worse than no read** (held B−A, t +5.1 to +7.2), for both rules on both runs.
- **The only rule that helps on the held score is still the plain error read on July** (report 4), which is worse on nmse. The two scores disagree in sign for every rule measured so far.
- At 768 the living step's nmse against no read is not below zero (B−A +2.3e-04 and +1.3e-04), as in report 4.

## Errors from a third document, the rest

**July, second set** (the first set and the repeat read are in report 4).

| Rate | held B−A, shuffled | held B−A, own errors | held B−C, shuffled | held B−C, own errors |
| --- | --- | --- | --- | --- |
| low | −1.6e-05 (t −9.5) | −2.8e-05 (t −6.9) | +1.3e-06 (t +0.8) | −8.9e-06 (t −2.3) |
| middle | +8.9e-05 (t +2.2) | −7.1e-05 (t −1.8) | +1.1e-05 (t +0.2) | −1.2e-04 (t −3.2) |
| high | ran away (+1.0e+02) | ran away (+1.0e+02) | about 0 | about 0 |

Both sets now agree: the gain that is the document's own comes from its own errors.

**768, all three layouts.** At the low rate the shuffled read runs away exactly as the unshuffled one (nmse B−A about +97 on 64 windows and +134 on the repeat read; held B−A +0.85 to +0.93). At the middle and high rates the six jobs raised in the rank guard's SVD after half a minute, as the unshuffled ones did, and left `.failed` markers.

## Not done

- The fitted map on July with the ridge pulled toward a scaled identity, with the held score (addendum 2, item 4). It needs a code change. It is next unless you want something else first.
- Taking the training loss's two terms apart, to see which one moves the targets.

## Artifacts

- `runs_meta/living_gradient_2026_11_17/<run>/e1_read64_*gradient_1e-5_held*.json` (8), `e1_*error_shuffled*.json` and `.failed`.
- The plans: `redteam/terminal_fable_2026_11_17/run_gradient_jobs.py --plan m2_held` and `--plan error --only error_shuffled`, both with `--workers 2 --keep-going`.

— Fable 5.2 (terminal), 2026-11-18
