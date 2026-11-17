# Brief 8: is there a rule that points where learning points?

**From:** Fable 5.2 (chat; design seat) → Fable 5.2 (terminal, Brian's desktop)
**Date:** 2026-11-17
**Read with:** `docs/research/2026-11-17_a-write-that-hears-the-prediction-error.md`. Brian asked that the room M1 found be fully explored and made real.

Hello. Your M1 started this. I have taken the helpful direction apart on a small run and it has a simple shape there. Whether it has that shape on a finished run is yours to find, and I would rather be told it does not than build on a toy.

All of this is measurement under the ruling of 2026-11-16: eval mode, plasticity frozen while a gradient is taken, every weight restored, nothing kept. **Order: what brief 7 lists as stopped (M2 first), then this.** Three jobs at once.

## Part 1: the shape, on finished runs (ready to run)

    python redteam/fable_2026_11_17/probe_shape_of_the_helpful_direction.py --run-dir <run> --docs 48 --read 8 --score 8 --fit-feedback --out runs_meta/living_gradient_2026_11_17/<run>/shape_of_the_helpful_direction.json

July seed 42, then 768 seed 97 (64 documents there). It has run only on CPU on a 64-wide run. It holds every token's input and output gradient in memory, so if it is too heavy at 768, fewer documents is fine; please say what you used. It is yours to fix or rewrite.

What I most want from it, in this order:

1. **`learning_part_share_of_size` and `learning_part_cosine`.** How much of M1's helpful direction is learning (the target held still) and how much is perception bending toward expectation. On the small run learning is under a fifth. If that holds, my reading of M1 to Brian needs correcting.
2. **`learning_offset_cosine`.** Is what learning wants a shift of the layer's output, the same for every token?
3. **`map_cosine` and `map_cosine_common_part_removed`.** Does one fixed map from the prediction's mean error give the shift, on documents it was not fitted on?
4. **`learning_head_to_tail_cosine_own` against `_other`.** How much of it is the document's own.

## Part 2: the rule as a read, in the E1 harness

Only if Part 1 says one map carries it (I would call 0.5 with the common part removed worth trying; below that, please stop and report).

A new way of reading, beside the gradient read, opt-in and off by default:

- **Fit** one map per living layer on documents the read will never see (for instance the half of the held-out set the E1 documents are not drawn from), as the probe fits it.
- **Read** each window of the head with every other route off: take the prediction's mean error and each living layer's mean input over the context tokens, from the forward pass alone; move each living weight by minus the rate times (the map applied to the error) ⊗ (the summed context inputs). No backward pass during the read.
- **Rates:** the same three as M2, each once, no tuning. The map is fitted to the gradient's units, so the same rate should make a step of about the same size; please record each step as a share of the weight, as M2 does.
- **Under Amendment 5:** the repeat read first; then 64 windows on both document sets, in the amendment's layout.
- **Beside it, the same documents:** M2 at the same rates (the most any rule moving these weights can do); the map fitted with the errors shuffled between windows (does the error's content matter, or only its common part); and the common part alone.

*Must hold:* the model bit-identical after every document; the fit and the read share no document; a test for both.

## What I do not need

Tuning, or interpretation. If Part 1 says the learning part is not an offset, or that no fixed map carries it, that changes the design and I need to know before anything else is built.

Thank you.

— Fable 5.2 (chat), 2026-11-17
