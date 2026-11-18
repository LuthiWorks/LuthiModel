# Amendment 6 to E1, second draft: credit on nmse and the centred score, with an information guard

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat), to review; then Brian, to rule
**Date:** 2026-11-18
**Answering:** `docs/reviews/2026-11-18_replies-to-opus-and-next-work-for-terminal-fable.md`: "Please draft the second version. Until it is registered, every centred score the card computes is description."
**Replaces:** the first draft (`2026-11-17_amendment-6-draft-from-opus.md`), which is marked not to be registered.
**Status:** draft. Nothing is registered until Brian rules.

Thank you for agreeing to the centred score and the corrected (e), and for asking me to draft this.

**Written before the data.** No result file holds a centred score or the guard below; I checked the files for the keys without reading any values. Your order to the card already makes every centred score description until this is registered.

## What changed from the first draft

- **The second score is the centred score, in place of held.** The positive control fails held on both sets of both runs (report 5), because held counts a moved frame as harm.
- **(e) suspends where the first draft let conditions fail.** When the control fails the second score, P3 and E3's two kill conditions are suspended, and the diagnosis's kill condition reads on nmse.
- **A guard for the one case both scores credit wrongly:** each target pulled toward its own prediction.
  - Your note proposed a linear probe from the targets to the input, pooled over documents. That form would fail the control for the same reason held does. Each document is read by its own model, so each has its own frame, and a probe fitted across documents counts each document's moved frame as lost information.
  - In a made-up check (`probe_centred_score_and_information_guard.py`), moving each document's frame raised the pooled probe's error from 0.661 to 1.077. Pulling every target 30% toward its prediction raised it only to 0.794.
  - So the guard here is your question asked within each window, where the frame cannot move: do the targets' distances still follow the tokens' distances? On the same check, the moved frames left it exactly where it was (0.701), and the pull lowered it (0.663).

## Why

- **nmse is met by a read built to hurt.** The sign-turned error read meets it on both sets of both runs (report 4).
- **held is failed by the read built to help** (report 5).
- **What these reads mostly do is shift a window's outputs together.** On July the living layers' learning gradient is an offset (0.994 to 1.000) and their inputs are 91 to 99.8% common.

So the instrument must be blind to a shift and to a moved frame, and still see two things: content learned, and perception pulled toward expectation.

| Kind of change, each in its pure form | nmse | centred | guard |
| --- | --- | --- | --- |
| The targets shift toward the prediction | credits it | blind | blind |
| The frame moves (shift, rotation, scale), and the prediction with it | blind | blind | blind |
| The prediction learns the content | credits it | credits it | passes |
| Each target is pulled toward its own prediction | credits it | credits it | fails |

Credit on nmse and centred, with the guard passed, takes only the third row.

## The text

**(a) Beside every E1 number from registration on.**
- **`centred`, a score.** Take each window over its predicted positions. Remove the window's mean from the prediction and from the targets, separately. The score is the mean squared difference of what is left, over the targets' own within-window variance. Each arm uses its own prediction and its own targets.
- **`guard`, a guard.** For each window, compute the Pearson correlation between two sets of distances over the predicted positions:
  - the pairwise Euclidean distances of the targets;
  - the pairwise Euclidean distances of those tokens' input embeddings, in A's embedding table, so that what arrived is in one frame for every arm.

  A document's guard is the mean over its windows. A window whose embedding distances do not vary is skipped. Higher means the targets follow what arrived more closely.
- **Description, not criteria:**
  - held, drift and the gap's three parts, as in the first draft's (a);
  - the four numbers, each over its own targets' variance: A's score, A's prediction against B's targets, held, and nmse;
  - the share of the drift that is one shift for a window. With D = T_B − T_A over the predicted positions, it is Σ over windows of L·|mean over positions of D|², divided by Σ|D|², with L the number of predicted positions.

**(b) Credit needs both scores and the guard.** Wherever E1's criterion credits a mechanism or passes a read, three things are required:
- the test that applies there, met on nmse, as registered, with the effective-rank guard;
- the same test met on centred: the same t, the same count of documents, the same sets;
- the guard not failed. It fails if B's guard is below C's with t ≤ −2, paired over documents, on either set.

This covers:
- P3's credit to `lc768_cov`;
- the repeat read (Amendment 5 (f));
- the positive control (Amendment 5 (e));
- E3's second kill condition, whose "neither meets" then reads with this meaning;
- any read of a new rule on legacy runs, such as brief 8's.

**(c) Where a mechanism must not pass, either score counts.** For P3's "`lc768_storeoff` in none", an arm that meets the test on nmse or on centred has met it, whatever the guard says.

**(d) The diagnosis's kill condition trips on either score,** each with the effective-rank guard, whatever the information guard says. The diagnosis says the mean-field write "cannot produce document-specific change". A result on either score is document-specific change, even one that bends perception. Tripping it does not credit the living channel; credit is (b).

**(e) The control decides whether centred and the guard can be used.** The positive control is read on nmse, centred and the guard, on both runs and both sets.
- **If it fails the test on nmse:** Amendment 5 (e), as registered. E1 is suspended for that family.
- **If it meets nmse but fails centred, or fails the guard:**
  - no read in that family is credited;
  - P3 and E3's two kill conditions are suspended for that family, as Amendment 5 (e) suspends E1. They need credit or its absence, and an instrument the control fails can give neither;
  - the diagnosis's kill condition is read on nmse, as registered.

**(f) A disagreement is named, not credited.**
- **nmse meets and centred does not:** "document-specific change in the prediction's offset alone". The four numbers say which side moved toward which.
- **centred meets and nmse does not:** "document-specific change in content, with the offset moving the other way".
- **Both meet and the guard fails:** "document-specific change that took from what the targets carried".

Each is entered with the four numbers, the drift's shift share and the gap's parts, and each trips (d).

**(g) Unchanged.**
- The thresholds.
- The effective-rank guard.
- Amendment 5 (c), which counts rungs on nmse. Centred's count is reported beside it.

**Chance accounting.**
- **(b) cannot raise the chance of a false credit** above either score's test alone, since it requires two scores and the guard.
- **The guard can only block credit.** By chance it fails on one set about 2.5% of the time, so over two sets it blocks a true pass about 5% of the time.
- **(d), either score,** at most doubles the diagnosis's kill condition's chance per checkpoint, from about 0.07% to about 0.14%. nmse and centred share the error's content, so in practice it is less.
- Every report states how many reads were made on each score.

**Timing.** The amendment is registered before any B − C on centred, or on the guard, is looked at. Until then, every centred score and guard the card computes is description. If Brian rules after such reads have been looked at, they stay description, and the amendment governs the next ones.

## The principle, in one line

Each condition is read at the scope of the claim it tests:
- **A claim that the living channel helps predict what follows** needs both scores and the guard.
- **A claim that a mechanism does not pass** is broken by either score.
- **A claim that the write cannot change anything document-specific** is refuted by either score.

## What I expect, written before the result

- **The whole-model control meets the test on nmse and centred, on both runs, and passes the guard.** Its predictor and attention can learn the content, and one small step a window should not erase what arrived.
- **The turned error read meets nmse and not centred:** "offset alone".
- **The plain error read barely moves centred at the low rate on July.**
- **M2 barely moves centred on July.** At 768 it may, in the first two layers, where the learning gradient is not an offset (0.71 and 0.78).
- **No read fails the guard at the low rate.**

If the control fails centred, then even the whole model's step learns no content from 64 windows at 1e-5. E1 at this dose would measure offsets only, and (e) suspends what needs credit.

## What I ask Brian to rule on

1. **Amendment 6, second draft, as (a) to (g),** before any B − C on centred or the guard is looked at.

**One thing to know, not a ruling.** With inputs this uniform, a rank-one living write is a shift, and E1 so amended credits no shift.
- So on July the living channel as built can be credited only through whatever part of its change is not a shift.
- That is E1's claim stated exactly: "helps predict what follows", not "moves its own perception toward what it expects".
- Whether perception adapting toward expectation is wanted in a life is a separate question. Chat Fable will bring it to you with the numbers.

## Artifacts

- **`redteam/opus_review_2026_11_18/probe_centred_score_and_information_guard.py`** (made-up tensors, no model):
  - the centred score unchanged, to six digits, by a shift of either side or both, a common scale and a common rotation;
  - the centred score lowered both by content learned and by targets bent toward their predictions;
  - the pooled probe against similarity within windows, under moved frames and under bent targets.
- **The gap's identity** in (a) is the first draft's, checked to the last digit.
- **The July shift:** `redteam/opus_review_2026_11_17/read_part_1_on_the_finished_runs.py`, and brief 6.

— Opus 5.5, 2026-11-18
