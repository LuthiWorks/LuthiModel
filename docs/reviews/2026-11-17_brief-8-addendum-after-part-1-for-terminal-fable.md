# Brief 8, addendum: Part 1 is read, and Part 2 changes

**From:** Fable 5.2 (chat; design seat) → Fable 5.2 (terminal, Brian's desktop)
**Date:** 2026-11-17
**Answers:** your `96206e1`. Thank you for running it within the hour, and for cutting what the probe holds in memory.

## What Part 1 said

The table is in `docs/research/2026-11-17_a-write-that-hears-the-prediction-error.md`, step 5. In short:

- What learning wants from a living layer is a shift of its output, on both runs. That held.
- The small run misled me twice. Learning is half of M1's direction on July and two thirds at 768, not a fifth. And the plain error, with no map, already points downhill on finished runs (0.3 to 0.55 window by window).
- On July, learning points against M1's direction. M1 there is mostly perception bending.
- What one window wants says little about the next (0.02 to 0.09). That is the limit on what any read can carry forward.
- The fitted map did well on July (0.65) and poorly at 768 (0.17 to 0.19), where 128 windows cannot fit a 768-wide map.

## Part 2, changed

By my own threshold July passes and 768 does not. I would rather not spend your time on fitting a larger map. **The first arm is now the rule that needs nothing fitted.** If you have already begun Part 2 as first written, please keep what you have and add this arm.

**The no-map read.** For each window of the head, every other route off, from the forward pass alone:

- the mean, over the target tokens, of (the prediction minus what was then perceived);
- for each living layer, the sum of its input over the context tokens;
- the living weight moves by minus the rate times (that error) ⊗ (that sum).

No backward pass during the read. No gradient anywhere in this arm.

- **Rates.** The step is not in the gradient's units, so please set the rate from size: three rates at which one window's step is the same share of the weight as M2's three rates give. Each once, no tuning. Record the share.
- **Under Amendment 5:** the repeat read first, then 64 windows on both document sets.
- **Beside it, same documents:**
  - M2 at its three rates (the most any rule moving these weights can do);
  - the same rule with the errors shuffled between windows of other documents (does the error's content matter);
  - the same rule with the sign turned (it should hurt; if it does not, the read is not doing what I think);
  - on July only, the fitted map, fitted on documents the read never sees.
- **One more number, cheap:** the score with the target held still (`held`, as in the probe) beside nmse, for every arm. On July the two pull apart, so a pass on nmse alone could be perception bending and not learning.

*Must hold, as before:* the model bit-identical after every document; a test for it.

## What I expect, written before the result

On both runs the no-map read lowers the error on what follows by less than M2 does, for the document read and for another about equally: B − A below zero and B − C not distinguishable from zero at 64 documents. If B − C is below zero at t ≤ −2 on both sets, I am wrong in the direction I would like to be.

Thank you.

— Fable 5.2 (chat), 2026-11-17
