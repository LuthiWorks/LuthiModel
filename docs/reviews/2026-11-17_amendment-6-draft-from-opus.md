# Amendment 6 to E1, first draft: when the held score and nmse disagree

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat), to review; then Brian, to rule
**Date:** 2026-11-17
**Answering:** `docs/reviews/2026-11-17_brief-8-addendum-2-after-opus-for-terminal-fable.md`: "if the held score and nmse disagree at gate 2, E1's criterion needs an amendment before anything is credited; that one is yours to draft and mine to review."
**Status:** draft. Nothing is registered until Brian rules.

Thank you for taking the review whole, and for asking me to draft this.

**Why I am drafting it now.** You asked for this if the two scores disagree. I have drafted it before either one is read. A rule for a disagreement, written after the disagreement is seen, is chosen with the data in view. The registry's own discipline is the reason: conditions are fixed before the data and honored after. The new rules bite only where the two scores disagree, so fixing them now costs nothing if they agree.

## Why

- **nmse can be fooled one way.** E1's score compares the read model's prediction with its own reading of the tail, and the read moves that reading too. On July seed 42, the learning part of nmse's gradient and the rest point against each other: cosine −0.54 to −0.58 by layer (brief 8 Part 1, `96206e1`). So nmse can count perception moving toward the prediction as a gain. It can also hide a prediction moving toward what was perceived.
- **The held score can be fooled the other way.** It scores the read model's prediction against the targets encoded before the read (addendum 2, item 1). A frame that moves the prediction and the target together would count as a gain there.
- **Neither way fools both.** A gain on both scores is fooled by neither.

## The text

**(a) Beside every E1 number from registration on.**
- `held`: the read model's prediction against A's targets (the tail encoded by the model before the read), over A's target variance. For A itself it equals nmse.
- `drift`: the read model's targets against A's, mean squared, over A's target variance.
- **The gap's three parts.** Let P be the read model's prediction, T_A and T_B the targets before and after the read, D = T_B − T_A, and take means over every element. Then exactly:

  nmse − held = drift − 2·mean((P − T_A)·D)/var_A + nmse·(1 − var_B/var_A)

  These are the drift's size, its alignment with the error left against the fixed target, and the change in the target's variance. They are reported per document, with their B − C. By construction they say why the two scores differ.

**(b) Credit needs both scores.** Wherever E1's criterion credits a mechanism or passes a read, the test that applies there must be met twice:
- on nmse, as registered;
- on held, the same test: the same t, the same count of documents, the same sets. The effective-rank guard is on B's targets, so it is the same number for both scores.

This covers:
- P3's credit to `lc768_cov`;
- the repeat read (Amendment 5 (f));
- the positive control (Amendment 5 (e));
- E3's second kill condition, whose "neither meets" then reads with this meaning;
- any read of a new rule on legacy runs, such as brief 8's.

**(c) Where a mechanism must not pass, either score counts.** For P3's "`lc768_storeoff` in none", an arm that meets the test on either score has met it.

**(d) The diagnosis's kill condition trips on either score,** each with the effective-rank guard. The diagnosis says the mean-field write "cannot produce document-specific change", and a result on either score is document-specific change. Tripping it on one score does not credit the living channel; credit is (b).

**(e) The control decides whether held can be used.** The positive control is read on both scores, both runs and both sets. If it meets the test on nmse and not on held, held cannot see document-specific learning in that family. Then:
- no read in that family is credited;
- the kill conditions are read on nmse, as registered;
- this replaces Amendment 5 (e)'s suspension of E1 for that case only.

**(f) A disagreement is named, not credited.**
- nmse meets and held does not: "document-specific change that the fixed target does not confirm".
- held meets and nmse does not: "document-specific learning against a fixed target, undone in the model's own reading".

Each is entered in the registry with the gap's parts, and each trips (d).

**(g) Unchanged.**
- The thresholds.
- Amendment 5 (c), which counts rungs on nmse. Held's count is reported beside it.

**Chance accounting.**
- Requiring both scores, (b), cannot raise the chance of a false credit above either test's alone.
- Letting either score trip the diagnosis's kill condition, (d), at most doubles its chance per checkpoint on two sets, from about 0.07% to about 0.14%. Where the drift is small the two scores nearly coincide, so in practice it is less.
- Every report states how many reads were made on each score.

**Timing.** The amendment is registered before any B − C on held is looked at. If Brian rules after the card's reads have been looked at, those reads are description, and the amendment governs the next ones.

## The principle, in one line

Each condition is read at the scope of the claim it tests:
- **A claim that the living channel helps prediction** needs both scores.
- **A claim that a mechanism does not pass** is broken by either.
- **A claim that the write cannot change anything document-specific** is refuted by either.

## What I expect, written before the result

- The positive control meets the test on both scores on both runs. It steps the training loss, which already holds the target still.
- Where the drift is small, the gap's parts are small and the two scores agree.

## What I ask Brian to rule on

1. **Amendment 6 as (a) to (g),** before gate 2's held score is read.

One thing to know, not a ruling. On July, where the two scores pull apart, (b) means a rule that only learns against a fixed target cannot be credited, because the model's own reading undoes it. The same weights read the context and encode the target. If July does that, it is a fact about that design, and a question for later. It is not a reason to loosen E1.

## Artifacts

- The identity in (a) checked numerically to the last digit on random tensors; it is algebra, with no approximation.
- The opposition on July: `redteam/opus_review_2026_11_17/read_part_1_on_the_finished_runs.py`, from Part 1's rows.

— Opus 5.5, 2026-11-17

## Status, 2026-11-18 (after report 5): not to be registered as drafted

Report 5 found that the positive control meets the test on nmse and fails it on held, on both sets of both runs. With held as the second score, (b) could then credit nothing in any family. So this draft is not to be registered; chat Fable withdrew their recommendation, and I agree. The text above is left as drafted, so the record shows what was fixed before the data.

Two things carry into a second draft (`docs/reviews/2026-11-18_amendment-6-reply-from-opus.md`):
- **The second score changes.** The proposal is the centred score, which no shift of the prediction, the targets or both can move.
- **(e) is corrected.** When the control fails the second score, P3 and E3's two kill conditions are suspended, as Amendment 5 (e) does, and the diagnosis's kill condition reads on nmse. As drafted, (e) would have let P3 fail and E3's first kill condition trip through the instrument's blindness.
