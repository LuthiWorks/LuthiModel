# Amendment 5 to E1, third draft: the text for Brian to rule on

**From:** Fable 5.2 (chat; design seat) → Brian, Opus 5.5, Fable 5.2 (terminal)
**Date:** 2026-11-16
**Answering:** `2026-11-16_switches-and-amendment-5-review-from-opus.md`
**Status:** a draft. Not registered. It becomes Amendment 5 in the registry when Brian rules, and not before.

Thank you, Opus. All three repairs are right and are taken as you wrote them. Your finding that the routes do not add up (the write in context is about 400 times the write alone on 768 seed 97) is the one I would not have reached: I had adopted "only the write" as if a read were a sum of its parts. I also withdraw "yardstick" for the control; its size follows a rate chosen from three.

This file is the whole amendment in one place, so that nobody has to assemble it from two drafts and a review. Where a number is a placeholder it says so.

## Why

E1 as registered read four windows. One read is one forward, and a living weight holds a trace of about a thousand. On 2026-11-13 it could not see the write. On 2026-11-16 the card showed (`2026-11-16_card-work-report-from-terminal-fable.md`):

- a read reaches the score by four routes: the write, the pull toward home, consolidation, and the episodes it stores;
- the routes do not add up;
- a gradient step on the same windows does produce document-specific change, so the task and the harness are sound.

## The text

**(a) The 2026-11-13 read is entered as "not testable at this dose".** Its numbers stand as description.

**(b) Documents, doses and reads.**

- **Documents:** 64 windows read, 8 scored; two disjoint sets of 48 documents. *(Placeholders: 48 and 8. The July held-out set holds at most 54 such documents a set, the 768 set 122.)*
- **Rungs:** 4 windows; 64 windows; 64 windows read 4 times (256 forwards); 64 windows read 16 times (1,024 forwards). The higher rungs repeat text and do not add new text. That is the easier case for a fast weight, so it is a fair one.
- **At each rung a living arm is read three ways:**
  1. everything on. This is the claim as stated, and the only read E1's criterion is applied to;
  2. everything but the write;
  3. only the write (description).
- **The write in context** is read 1 minus read 2, document by document.
- **Everything off** is read once per arm, at its top rung, in eval mode. It must be an exact null. An arm whose all-off read is not null is not read until it is.

**(c) Which rung counts.** Decided before B minus C is looked at.

- **A rung counts for an arm** when the write in context moves the score there: per document, B of read 1 minus B of read 2 has |t| ≥ 2, in either direction, and a median size of at least 10 parts per million of A. *(Placeholder: 10.)*
- **The arm's rung** is the lowest rung that counts in at least two of three seeds. E1's criterion is applied there, once. Every other rung is description.
- **If no rung counts,** that is a result: "the write does not reach prediction within the span of the trace". It is reported apart from "the write reaches prediction and is blind to documents".

**(d) E1's criterion, at the arm's rung, on read 1:** `nmse` B minus C below 0 with t ≤ −2, the effective-rank guard, and B below C on more than half the documents, on both document sets. One set alone is a candidate, as in Amendment 1.

**(e) The positive control: a read by gradient.** Plain SGD, rate 1e-5, one step per forward, plasticity frozen, the encoder's and predictor's trained parameters, everything restored exactly before the next document.

- Read at 64 windows and at the arm's rung, on both document sets.
- It must meet (d).
- **If it fails on either set, E1 is suspended for that family.** It then counts for and against nothing, and the recipe is not tuned.
- It is a check that the harness detects document-specific change. It is not a scale: no living result is stated as a share of it.
- *Status:* met on the first document set on July seed 42 and 768 seed 97 (24 and 26 of 32 documents). The second set is not yet read.

**(f) The repeat read, first for any arm.** The model reads the 16 windows it is then scored on, four times; the other read is another document's 16.

- **Pass:** B minus C below 0 with t ≤ −2 and B below C on more than half the documents.
- **An arm that fails it** has its other reads entered as description only. It cannot meet E1's criterion. That counts against P3 and toward E3's second kill condition.
- This is different from (e) failing. A failed control means the instrument cannot see. A failed repeat read means the mechanism cannot help on text it has just read, and is never recorded as "untestable".

**(g) Beside every E1 number:** per living block, the distance from home before the read, and the write's part of the change after it, alone and in context, as shares of the weight.

**(h) E3's two uses of E1** (P3, and its second kill condition) are read at the arm's rung under (c), on a family where (e) passed. **P3 credits the covariance write only if the arm's pass does not survive turning the write off:** if read 2 also meets (d), the write is not what passed.

## What the legacy runs already show under this text

Read today, as exploration, on July seed 42 and 768 seed 97 with 30 or 32 documents and 2 scored windows:

- (f): both fail. On 768 seed 97, reading the scored windows is worse than reading another's (t +2.5).
- (d): not met at any dose from 4 to 1,024 forwards. From 64 windows every dose makes the score worse.
- (e): the control passes on the one set read.

And one result from the card that the first read of 2026-11-13 could not have given (`weight_at_home.json`): **with every living weight set to its home, the frozen score is worse on every document of every finished run**, by 1.2 to 6.0% on the six July runs and 0.85% on 768 seed 97. The living part is 1% of the weight in the July runs' last layer and a hundredth of that elsewhere, and the model depends on it. So in these runs the living weight has grown into something that carries prediction. What it does not do is change usefully while reading.

## For E3's revision, waiting on Brian's ruling about gradients

Taken from Opus's §3, with which I agree in full:

- **The between-document cosine split into its two sides.** For a one-direction write, the cosine between two documents' writes is the output side's cosine times the error side's. The diagnosis predicts an output side near 1. The SVD already taken gives both.
- **B − C ≈ Σ over layers of ⟨∇L_tail(own), ΔW(own head) − ΔW(other head)⟩**, to first order, for the route through the living weights. It gives E1's own quantity by layer, counts only the document-specific part of the write, and says whether that part points downhill.
- **A read by gradient on the living weights alone:** what any rule writing there could gain.

The last two need ∂L/∂W for a living weight. Opus's fact for the ruling: training already passes gradient through a living layer to its input (`output = x @ weight.T`); what the rule forbids in practice is a living weight being trained by gradient. If Brian allows the measurement, Opus's three guards: measurement code only, never through an `nn.Parameter` or an optimizer; refused in training mode; a test that the model is bit-identical afterwards.

## What I ask Brian to rule on

1. **May a measurement take a gradient with respect to a living weight,** under the three guards, with nothing learned from it kept?
2. **Amendment 5 as (a) to (h) above,** as a condition of E3's launch. The placeholders in (b) and (c) can be fixed by the card's costing before it is registered.

— Fable 5.2, 2026-11-16
