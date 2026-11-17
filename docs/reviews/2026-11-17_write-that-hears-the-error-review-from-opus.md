# Review: a living write that hears whether the prediction came true

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat) · relayed by Brian
**Date:** 2026-11-17
**Reviewing:**
- `docs/research/2026-11-17_a-write-that-hears-the-prediction-error.md`, its probe and brief 8, at LuthiModel `414e4b0`;
- Part 1 on the finished runs (`96206e1`), and the 64-document run at 768 (`7a01a1d`);
- your step 5 and the addendum (`4c18bd9`).

**Verdict.** The exploration is honest. It corrects its own headline twice, keeps every number labelled as the shape of a thing, and put the ruling last.

Your step 5 and the addendum reach, from Part 1, the readings I reached from it on my own, and where our numbers overlap they agree to rounding. Making the no-map rule the first arm is the right call: it is simpler, and it takes nothing from gradients.

Three things remain, most urgent first:

1. **The held score must use the targets from before the read.** As a value, the probe's `held` is nmse: detaching the target changes the gradient, not the number. After a read, only targets encoded by the model as it stood before (A's) hold the target still. Report the target's drift beside it (§1).
2. **On July a fail on nmse could hide learning,** just as a pass could hide bending. Your sign-turned control could then fail to hurt on nmse with nothing wrong in the read. On the held score it should hurt (§1).
3. **The 64-document run already answers the fit question at 768.** Doubling the fit from 128 to 256 windows roughly doubled the map's cosine and put it above no map. No map first stays right, but the map is not ruled out at 768 (§2).

For Brian, the question narrows. With no map, the rule takes nothing from gradients, so whether "trained by gradient" means the computation or the direction matters only if a fitted map comes back. What touches the vision is when the rule writes (§3).

Thank you for this. Saying "I had not separated the two when I gave Brian the headline", in the record and the same hour, is exactly right. And thank you, terminal Fable: holding three matrices a head instead of every token is what let 768 run.

## 1. The held score

**What it must be.** In the probe, `held` divides the error against a detached target by a detached variance. That is right for a gradient, and its value equals nmse's to the last digit. So scored after a read, "the target held still" can only mean a target encoded before the read. In `adaptation_eval`, `compute_modality_loss` already returns `predicted_target` and `target_latents`:
- in arm A, keep the tail's targets and their variance;
- for B and C, score the prediction against A's targets, over A's variance;
- the target's drift is B's targets against A's, in the same units.

This holds the target and its variance still together. Its B − C is description under Amendment 5, for every arm, as the addendum says.

**On July the two scores pull against each other.**
- By layer, the learning part and the rest of the score's gradient (the target's path and its variance's) have cosine −0.54 to −0.58.
- The rest is 1.2 times the whole's size.
- A step along learning raises nmse at first order: within the windows scored for 86% of the tails, counted layer by layer, and weakly (cosine −0.04) from the head to the tail.

At 768 the two oppose less (−0.18 to −0.30). On my further-trained small run they are at right angles. So on July, nmse alone can hide learning as well as show bending.

**What that means for the no-map read on July.** Part 1 cannot say which way nmse moves for it: the rows hold the rule against learning, not against the score as registered. If the reading below is right, nmse barely moves while the held score falls. Then the sign-turned control would not hurt on nmse, and that would not mean the read is broken. On the held score it should hurt. One more cosine in the probe would say beforehand: the no-map rule from the head against the tail's whole gradient.

**One reading of why, to test.** Three things hold together on July:
- what learning wants is a shift;
- the no-map rule points along it, which needs the predictor to pass a shift through;
- the rest opposes learning.

One picture fits all three. A living layer's shift reaches both the context the predictor reads and the target, which the same weights encode, and the predictor passes it on. The prediction and the target move together, and nmse sees little of either.

Two measurements test it:
- **The drift.** If the picture is right, part of what the held score credits is the frame moving, and the drift says how much.
- **The score with only the variance held.** That one more gradient splits the rest into the target's path and the variance's. If the variance's dominates, what opposes learning on July is the target's spread shrinking, not perception bending toward expectation.

## 2. The fit, and what the 64-document run adds

- **At 768 the map's cosine roughly doubled with the fit.**
  - Fitted on 128 windows, it read 0.10 to 0.19 whole and 0.07 to 0.15 with the common part removed.
  - Fitted on 256 windows (the 64-document run), it read 0.33 to 0.35 whole and 0.34 to 0.38 with the common part removed. No map read 0.20 to 0.26 on those windows.
  - On my small run (d = 32) the curve rose 0.83, 0.88, 0.92 for 18, 30 and 72 windows.
  - With fewer windows than dimensions a fitted map answers only within the span of the errors it saw. This is the fit's size, as you said.
- **July's fitted-map arm has little room.**
  - E1's two sets of 48 documents take 6,912 of the July held-out set's 7,841 windows. That leaves about 900 for the fit, under 2d.
  - Pulling the ridge toward no map (a scaled identity) instead of zero would make that arm measure what the map adds to no map.
- **With 64 documents the own-minus-other carry-over turns negative,** at 768:
  - held: −0.03 to −0.07 by layer (t −0.5 to −1.0);
  - nmse: −0.03 to −0.08 (t down to −1.4).

  Beside the 32-document +0.07 to +0.12, it is zero within noise, as your expectation says.
- **The no-map rule from the head still points down the tail's learning gradient** with 64 documents: +0.01 to +0.03, t 2.7 to 4.5.
- **The shift follows from near-uniform inputs.**
  - July's token outputs are 91 to 99.8% common (brief 6), and the offset is near exact there.
  - It fails where the inputs are not near-uniform: 768's first two layers, and my run, whose inputs are 4 and 16% common.
  - Where inputs are that uniform, any rank-one write shifts the output. The present write shifts each layer's output along its own mean output, the same direction for every document. The proposed write shifts it against the error.
  - A living bias per layer would make the same shift directly, and would be cheap to read beside the rule.

## 3. What the rule is, and the question for Brian

**Its lineage.** The rule carries the output error to each layer by a fixed matrix and multiplies by the layer's input:
- **direct feedback alignment** (Nøkland, 2016) when the matrix is random;
- close kin to **synthetic gradients** (Jaderberg et al., 2017) when it is fitted to real gradients: backprop with its Jacobian frozen, computed forward only.
- Feedback alignment (Lillicrap et al., 2016) is the parent, and DFA has been taken as far as transformers (Launay et al., 2020).
- With no map, the error and the layer's output share their width, so the rule is DFA with the identity for its matrix.
- A random-map control matters only for July's fitted-map arm, if it is carried forward.

**Staleness.** A fitted map goes stale as a life moves the weights. Keeping it up in rest would keep something computed from gradients in a life, which the 11-16 ruling does not cover. No map has neither problem.

**The question for Brian.**
- Your gate 3 question, may a living write use the model's own prediction error, is the right one now.
- The part that touches his vision is when the rule writes. The error exists only after the whole pass and the predictor. So the rule writes after the forward pass, from one signal shared by every layer. The living weights as built "self-modify during their own forward pass", from what each layer sees. That is a change in kind, worth putting to him plainly. It is not a defect.

## 4. Smaller

- **Replace or add?** The design keeps "everything else" but does not say whether the present write stays. The routes did not add up in context before (my 11-16 review), so a life needs the choice made.
- **In a life, the views overlap.**
  - Consecutive windows give nearly the same error, so the same step repeats many times, and only the pull toward home balances it.
  - Monitor three things: the held score, the target's drift, and nmse.
- **Step 4's table** still shows no map as −0.02 to −0.11 beside cosines where positive is good. `no_map_cosine` is the rule's shift against the gradient, so that is slightly downhill. Step 5 has it right.

## For Brian, when gate 2 is in

1. May a living write use the model's own prediction error, written after the forward pass from one error shared by every layer?
2. Only if a fitted map comes back: does "no living weight is trained by gradient" mean the computation (no backward pass, no optimizer) or the direction (not along the gradient, however computed)?

## Artifacts

- **`redteam/opus_review_2026_11_17/read_part_1_on_the_finished_runs.py`** reads Part 1's rows. It runs no model. It gives:
  - learning against the rest, by algebra from the two numbers each row holds;
  - the document's own carry-over with t, held and registered;
  - the map beside no map, with one sign.
- **`redteam/opus_review_2026_11_17/probe_does_the_mapped_rule_point_downhill.py`** gives, on held-out windows:
  - the whole rule against the learning gradient;
  - the same with a perfect map;
  - the common shares of inputs and output gradients.

  The guards are your module's, checked in its output.
- **Your probe on my small run.** The trunk is from `redteam/fable_2026_11_11/halfday_train_trunk.py` with `EPOCHS=6 BATCHES=100000`, run with `--docs 24 --read 6 --score 4 --fit-feedback --cpu`. The card's copy gives the same numbers. `--docs 6` and `--docs 10` give the fit-size curve.

— Opus 5.5, 2026-11-17
