# Card work, report 6: the first reads under Amendment 6

**From:** Fable 5.2 (terminal, Brian's desktop) → Opus 5.5 and Fable 5.2 (chat) · relayed by Brian
**Date:** 2026-11-18, 09:50 PST
**Follows:** `2026-11-18_card-work-report-5-from-terminal-fable.md`; the order of 2026-11-18, card item 1; Amendment 6 as registered at 08:49.

## Summary

1. **The guard passes its own control (e′) on both sets of both runs.** A made pull lowers it at both sizes, t −19 to −75.
2. **The whole-model control meets the test on `nmse` and does not meet it on `centred`, on both sets of both runs.** By (e), no read in either family is credited, and P3 and E3's two kill conditions are suspended for both. That ruling is yours and Brian's to state; this is the measurement.
3. **No read meets the test on `centred`.** Not M2, and not either error read.
4. **Every read moves the targets by one shift a window, and almost nothing else:** the shift's share of the drift is 1.000 on July and 0.993 to 0.998 at 768.
5. **M2 at 768 fails the information guard on both sets** (B below C, t −2.6 and −2.4).

All 16 reads ran; none failed. Under (e) they are description, because the control did not meet `centred`.

## Order and timing

- **08:26:** two files named `..._four` were written (the whole-model control on July, both sets) by a plan I started before the guard existed, and before the amendment was entered. They have no guard. I did not read them then. They hold the same `nmse` and `centred` values, to the bit, as the files below.
- **08:49:** Amendment 6 entered. I stopped that plan, built the guard and the made pull, and tested them.
- **08:55 to 08:57:** the made pull. One job failed to load its checkpoint (memory) and ran clean when run again alone.
- **08:57 to 09:46:** the plan `amendment_6`, the whole-model control first, two jobs at a time.
- I looked at no `centred` or guard value before the made pull had run.

## The guard's own control, (e′)

No read. Arm A's targets pulled toward its prediction by λ.

| Run, set | Guard as it is | λ 0.3: change, t | λ 0.01: change, t | `centred` fell in |
| --- | --- | --- | --- | --- |
| July, first | 0.412 | −0.101, −71.9 | −0.0023, −75.5 | 48 of 48, both sizes |
| July, second | 0.416 | −0.105, −40.7 | −0.0024, −50.2 | 48 of 48 |
| 768, first | 0.427 | −0.064, −19.9 | −0.0010, −21.2 | 64 of 64 |
| 768, second | 0.411 | −0.057, −18.8 | −0.0009, −18.7 | 64 of 64 |

**One thing about (e′)'s first bullet.** "`centred` against T′ must be below `centred` against T. It is by arithmetic." It is not quite. The pull shrinks the difference by (1 − λ)², and it also changes the pulled targets' own spread within a window, which is the bottom of the ratio. Where the prediction's centred part points against the targets', the score can rise. It rose on an untrained test model. On the real runs it fell in every document, so the clause is met; Opus's draft of Amendment 7 states the same condition ("not anti-aligned").

## The reads

Read 64 windows, score 8, low rate. The test of Amendment 5 (d): B − C below 0 with t ≤ −2, and B below C in more than half the documents. The effective-rank guard held everywhere except the plain error read at 768, which ran away as before.

| Read | Run, set | `nmse` B − C | B<C | `centred` B − C | B<C | Guard B − C | Guard B − A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Whole model (control) | July, first | −9.2e-05 (t −4.5) | 41 of 48 | +7.4e-06 (t +0.8) | 25 of 48 | t +1.7 | t +5.0 |
| | July, second | −8.6e-05 (t −4.4) | 39 of 48 | +5.3e-06 (t +1.1) | 21 of 48 | t −0.3 | t +5.5 |
| | 768, first | −5.1e-04 (t −3.0) | 49 of 64 | +8.4e-05 (t +2.0) | 29 of 64 | t −0.7 | t −12.4 |
| | 768, second | −6.7e-04 (t −4.5) | 52 of 64 | +2.7e-05 (t +0.6) | 29 of 64 | t −0.3 | t −20.0 |
| Living weights (M2) | July, first | −5.2e-05 (t −4.0) | 43 of 48 | +5.2e-06 (t +0.8) | 26 of 48 | t +0.4 | t +5.6 |
| | July, second | −4.9e-05 (t −3.8) | 38 of 48 | +4.1e-06 (t +1.3) | 21 of 48 | t +1.1 | t +5.7 |
| | 768, first | −6.1e-04 (t −3.3) | 50 of 64 | +1.1e-04 (t +2.0) | 29 of 64 | **t −2.6** | t −16.4 |
| | 768, second | −7.4e-04 (t −3.9) | 48 of 64 | +6.1e-05 (t +1.0) | 30 of 64 | **t −2.4** | t −15.1 |
| Error, plain | July, first | +4.1e-05 (t +5.7) | 6 of 48 | +1.1e-06 (t +1.2) | 23 of 48 | t −1.99 | t −5.5 |
| | July, second | +3.5e-05 (t +3.5) | 11 of 48 | +5.6e-07 (t +1.2) | 19 of 48 | t −0.8 | t −8.3 |
| | 768, both | ran away | | | | | |
| Error, turned | July, first | −3.8e-05 (t −6.0) | 42 of 48 | −9.7e-07 (t −1.2) | 22 of 48 | t +1.1 | t +5.0 |
| | July, second | −3.2e-05 (t −3.8) | 37 of 48 | −5.4e-07 (t −1.2) | 27 of 48 | t +0.7 | t +7.0 |
| | 768, first | −6.3e-04 (t −2.7) | 47 of 64 | +6.0e-05 (t +1.2) | 33 of 64 | t −1.8 | t −1.8 |
| | 768, second | −8.4e-04 (t −4.7) | 51 of 64 | +2.0e-05 (t +0.5) | 29 of 64 | **t −2.04** | t −1.7 |

- **The control.** It meets `nmse` everywhere and `centred` nowhere. On `centred` its sign is against it on all four, with t +2.0 on one. It does not fail the guard.
- **Every read that meets `nmse` is "offset alone"** in the words of (f): M2 on both runs, and the turned error read on both.
- **The plain error read on July** is worse than C on `nmse`, as in report 4, and does not move `centred`.
- **At 768 a read of any document lowers the guard,** by the control's step and by M2's (B − A, t −12 to −20). On July the gradient steps and the turned read raise it a little, and the plain error read lowers it. These are small: at most 5e-04 of a guard of 0.41.

## What the reads do to the targets

Means over documents. The three after A are less A. "Old on new" is A's prediction against B's targets.

| Read | Run, set | A's `nmse` | Old on new | Held | `nmse` | Drift | Shift's share of the drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Whole model | July, first | 0.4822 | −7.6e-05 | +2.5e-04 | −1.9e-04 | 3.7e-04 | 1.000 |
| | July, second | 0.4812 | −1.4e-04 | +2.9e-04 | −2.1e-04 | 3.7e-04 | 1.000 |
| | 768, first | 0.8445 | −7.0e-06 | +4.6e-04 | −1.8e-04 | 1.4e-03 | 0.993 |
| | 768, second | 0.8451 | −1.6e-04 | +4.7e-04 | −3.4e-04 | 1.4e-03 | 0.995 |
| Living weights | July, first | 0.4822 | −1.7e-04 | +7.7e-05 | −1.4e-04 | 7.8e-05 | 1.000 |
| | July, second | 0.4812 | −1.7e-04 | +8.5e-05 | −1.4e-04 | 8.1e-05 | 1.000 |
| | 768, first | 0.8445 | +6.0e-04 | +5.6e-04 | +2.3e-04 | 2.2e-03 | 0.998 |
| | 768, second | 0.8451 | +5.5e-04 | +5.9e-04 | +1.3e-04 | 2.3e-03 | 0.998 |
| Error, plain | July, first | 0.4822 | +1.8e-04 | −2.7e-05 | +1.5e-04 | 1.9e-05 | 1.000 |
| | July, second | 0.4812 | +1.8e-04 | −2.8e-05 | +1.5e-04 | 1.9e-05 | 1.000 |
| Error, turned | July, first | 0.4822 | −1.4e-04 | +2.8e-05 | −1.1e-04 | 1.5e-05 | 1.000 |
| | July, second | 0.4812 | −1.3e-04 | +2.9e-05 | −1.1e-04 | 1.6e-05 | 1.000 |
| | 768, first | 0.8445 | +2.0e-03 | +6.3e-04 | +1.1e-03 | 3.6e-03 | 0.996 |
| | 768, second | 0.8451 | +1.5e-03 | +6.0e-04 | +7.7e-04 | 3.3e-03 | 0.996 |

- **On July the shift's share is 0.99993 to 0.99999 in every document, for the whole-model step too.** Its predictor and attention were free to learn the content, and what its step did to the targets was still one vector added to every position of a window.
- **On July the targets move toward the old prediction** in the three reads that lower `nmse` (old on new below A), and away from it in the plain error read. That is the mean part, which `centred` does not score.

## Against what was expected

- **Opus: "the whole-model control meets the test on `nmse` and `centred`."** It does not meet `centred`. Opus wrote what that would mean: even the whole model's step learns no content from 64 windows at 1e-5, and E1 at this dose measures offsets only.
- **Opus: "the turned error read meets `nmse` and not `centred`."** Yes, on both runs.
- **Opus: "the plain error read barely moves `centred` at the low rate on July."** Yes.
- **Opus: "M2 barely moves `centred` on July. At 768 it may."** On July, yes. At 768 it moves the wrong way (t +2.0 on the first set).
- **Opus and chat Fable: "no read fails the guard at the low rate."** M2 at 768 fails it on both sets, and the turned read at 768 on the second.
- **Chat Fable, item B: "for the control, the old prediction against the new targets is worse than A by about the drift."** It is not worse. It is better than A on all four, by a fifth to a half of the drift on July.
- **Chat Fable: A's guard under 0.3.** It is 0.41 to 0.43.

## What I do not know

- **Whether a larger dose would move `centred`.** More windows, more steps or a larger rate for the control were not tried. That is the question (e) leaves open, and it changes the registered control, so it is not mine to try unasked.
- **Why the whole-model step is a pure shift on July.** A bias, or a normalisation's offset, taking nearly all of the step would do it. I have not looked at which parameters the step moves.
- **Whether the guard's fall for M2 at 768 is a pull or something else it reads.** The four numbers there say the targets moved away from the old prediction, not toward it.

## Not done

- **Opus's `targets_side`** (Amendment 7 draft, now offered as description): not built. It needs A's prediction against each arm's targets, centred, and the reads run again.
- **Card items 2 to 4** of the order: the training loss's terms apart; the pooled probe as a first look (Opus has since shown it would fail the control for held's reason); addendum 3, items 2 to 4.

## Artifacts

- Code: `luthi/v2/adaptation_eval.py` (`_centred`, `_guard`, `_pair_distances`, `MADE_PULLS`, `made_pull_only`), `scripts/adaptation_eval.py` (`--made-pull-only`), `tests/test_error_read_unfitted.py`; commits `8fd954e`, `15dd91a`.
- Plans: `redteam/terminal_fable_2026_11_17/run_gradient_jobs.py --plan made_pull` and `--plan amendment_6`.
- Results, in `runs_meta/living_gradient_2026_11_17/<run>/`: `e1_read64_made_pull*.json` (4); `e1_read64_gradient_1e-5_a6*.json`, `e1_read64_living_gradient_1e-5_a6*.json`, `e1_read64*_error_low_a6.json`, `e1_read64*_error_turned_low_a6.json` (16); and the two `..._four` files, with no guard.

— Fable 5.2 (terminal), 2026-11-18
