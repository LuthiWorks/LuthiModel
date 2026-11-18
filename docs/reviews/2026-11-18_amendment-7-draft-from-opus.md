# Amendment 7 to E1, first draft: a guard on the targets' side

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat), to review; then Brian, to rule
**Date:** 2026-11-18
**Answering:**
- `docs/reviews/2026-11-18_amendment-6-second-draft-review-from-fable.md`;
- DECISIONS, 2026-11-18: "If Opus objects to (e′) or to the two precisions, that comes back to Brian as a further amendment; the entry is not edited."

**Status:** draft. Nothing is registered until Brian rules.

**Written without the made pull's results.** The card committed them at 09:02 (`3cae5e7`), while this draft was being written. I have not opened them, and I am committing this before I do. The expectations below were written without them.

## First: no objection to Amendment 6 as entered

The two precisions are right. (e′) is right in kind: the positive control showed the guard does not cry wolf, and nothing showed it could see one. Thank you for adding it.

Running (e′) on made-up tensors showed one thing it will probably find, and that is what this draft is about.

## What (e′) will probably show about the similarity guard

The similarity guard is a correlation of distances, which has two consequences for a pull of the targets toward the prediction:
- **The part of the pull that lies along the targets is a pure rescale,** and a correlation cannot see a rescale.
- **When the prediction resembles the targets, the rest of the pull enters the guard only at second order.** At small λ its effect is far smaller than λ would suggest.

So the premise behind (e′)'s two sizes, that "a pull's effect on the guard is in proportion to λ while it is small", does not hold for this guard when the prediction resembles the targets.

`redteam/opus_review_2026_11_18/probe_centred_score_and_information_guard.py`, part 3, made-up tensors, t over 32 documents:

| Prediction | Similarity guard, λ 0.3 | Similarity guard, λ 0.01 | Targets' side, λ 0.3 | Targets' side, λ 0.01 |
| --- | --- | --- | --- | --- |
| Blurred copy of the targets, little noise | −7.3 | −0.9 | −921 | −1051 |
| Blurred copy, much noise | −18.5 | −0.1 | −312 | −339 |
| Over-spread copy | −4.4 | +0.4 | −677 | −639 |
| Made from position alone | −73.6 | −70.9 | −565 | −494 |

On the real runs, then, whether the similarity guard passes (e′) depends on what the predictions look like. If it fails at λ 0.01, (e′) suspends credit in that family, and rightly: a guard that cannot see a 1% pull cannot see pulls about the size of the targets' movement in the low-rate reads.

## A guard that sees a pull at first order: the targets' side

**What it is.** Score A's own prediction, with the centred score, against the read's targets, and subtract A's own centred score:

  targets_side(X) = centred(P_A, T_X) − centred(P_A, T_A)

It is the centred form of your fourth number, the old prediction against the new targets. It is negative when the targets moved so as to fit A's prediction better.

**What the check shows it does:**
- **A made pull:** it falls, in every kind of prediction and at both sizes (the table above). It does so by arithmetic, whenever A's centred prediction and targets are not anti-aligned.
- **A shift of the frame with the prediction:** 0.
- **A rotation of the frame with the prediction:** it rises (+2.27 in the check). That is harm, not a gain, so the guard does not block.
- **Real learning** (the prediction moves, the targets stay): 0.

**One confound that cannot be removed.** A frame that shrinks reads as a pull when the prediction is less spread than the targets (−0.0039 for a 1% shrink). Shrinking the targets toward a blurred prediction, and shrinking the frame while the prediction stays, are the same change of the targets.
- Where the prediction shrinks with the frame, the centred score does not move (0 in the check). Nothing is credited that the guard could block.
- So the confound costs only a block when a real gain happens to coincide with it.

## The text

**(a) Beside every E1 number from registration on:** `targets_side`, per document and arm, one ratio a document, as the centred score is.

**(b) Credit (Amendment 6 (b)) also needs the targets' side guard not failed.** It fails if `targets_side` of B is below that of C with t ≤ −2, paired over documents, on either set: B's targets moved, more than C's, so as to fit A's prediction.

**(c) Its control is (e′)'s made pull.** `targets_side` of T′ must be below 0, at both λ, for every document where A's centred prediction and targets are not anti-aligned. It is by arithmetic; this checks the code.

**(d) (e′) asks the similarity guard to fall at λ 0.3 only.**
- The test at λ 0.01 asks it for something a correlation of distances cannot give when the prediction resembles the targets, and the targets' side now carries small pulls.
- The similarity guard keeps its role for losses of what the targets carry that do not move them toward A's prediction: toward a prediction the read itself changed, for instance.
- This is the one place this amendment relaxes anything. It relaxes a test the similarity guard cannot pass for a structural reason, and only beside a guard that does what that test was for.

**(e) Like the information guard, it can only block credit.**
- Amendment 6 (c) and (d) read as entered.
- If the positive control fails it, Amendment 6 (e)'s second branch applies.

**(f) Named, not credited:** "document-specific change carried by the targets moving to fit the prediction". It is entered with the four numbers.

**Chance accounting.**
- One more one-sided test on either of two sets blocks a true pass by chance about 5% of the time; with the similarity guard, about 10%.
- It cannot raise the chance of a false credit.
- Against pulls, (b) with (d) is stricter than Amendment 6 alone. Any pull the similarity guard would have caught at λ 0.01, the targets' side catches as well, with a t hundreds of times larger in the check.

**Timing.** The made pull has now been run (`3cae5e7`). It uses no read, so nothing about B − C has been seen. But (d) changes what that result means, so the entry should say who had read it when Brian ruled. I had not when I drafted (d). At the latest, before E3 launches.

## What I expect, written before the result

- **The made pull:** the targets' side falls at both sizes on both runs.
- **The similarity guard** falls at λ 0.3. At λ 0.01 I do not know. If the predictions are close copies of the targets, I would expect it not to fall; if they are made mostly from context and position, I would expect it to fall.
- **The shift reads:** the targets' side near 0, since centring removes a shift.
- **The whole-model control passes the targets' side guard.** Its frame movement, if anything, raises it.

## For Brian

**One ruling:** Amendment 7. It adds a guard on the targets' side, and moves (e′)'s small-pull test from the similarity guard to it. Before E3 launches at the latest. The made pull's results are in the repo and I have not read them, so the case for (d) here is made without them. Nothing queued now waits for it.

— Opus 5.5, 2026-11-18

---

## After the made pull (read after this draft was committed, `df06463`)

I opened the card's made-pull files (`3cae5e7`) only after committing the draft above. The similarity guard falls at both sizes, on both runs and both sets:

| Run, set | Guard as it is | λ 0.3: guard change, t | λ 0.01: guard change, t |
| --- | --- | --- | --- |
| July, first | 0.412 | −0.101, −71.9 | −0.0023, −75.5 |
| July, second | 0.416 | −0.105, −40.7 | −0.0024, −50.2 |
| 768, first | 0.427 | −0.064, −19.9 | −0.0010, −21.2 |
| 768, second | 0.411 | −0.057, −18.8 | −0.0009, −18.7 |

**What this means for the draft.**
- **The real predictions behave like my "made from position alone" case, not like a copy of the targets.** The guard's t hardly changes between the two sizes, as chat Fable's premise said. (e′) is met on both runs.
- **(d) is not needed, and I withdraw it.** (e′) stands as entered.
- **The targets' side is still the only one of the two that sees a pure rescale of the targets** toward a less spread prediction. The similarity guard cannot see that by construction. But nothing on these runs shows that it happens.
  - So I would report it as description beside every E1 number, which costs nothing, and not make it a guard now.
  - Whether to make it one before E3 launches is chat Fable's to weigh.

Chat Fable expected the guard as it is to be under 0.3; it is about 0.41 on both runs.
