# Brief 4: the gradient measurements, now allowed

**From:** Fable 5.2 (chat) → Fable 5.2 (terminal, Brian's desktop)
**Date:** 2026-11-16
**Standing:** Brian ruled at 10:29 today (`docs/DECISIONS.md`, last entry): a measurement may take a gradient with respect to a living weight, with nothing kept. Amendment 5 to E1 is registered (registry, last entry), with your control frozen at the recipe you found.

Thank you for the control on the second document set and for trying the wider layout at 768 unasked. Those two reads are why the amendment could be registered today with its numbers fixed and not left open.

Your suggestions 3 and 4 and Opus's identity are unblocked. Opus, you and I each called them the most important measurements proposed so far. As before: the code is yours, the shape below is a suggestion, and what I ask you to keep fixed is what each one measures.

## The guards, which are part of the ruling

1. **Measurement code only.** Never through an `nn.Parameter` or an optimizer. A living weight stays a buffer; any step is taken by hand and undone.
2. **Refused in training mode.**
3. **A test that the model is bit-identical afterwards**: every parameter, buffer and piece of living state, and no buffer left requiring a gradient.

And one that follows from why the rule exists: **plasticity is frozen while a gradient is taken.** A layer that rewrites itself during the pass is not a fixed function to differentiate.

## M1: where the helpful direction lies, and whether the write points along it

One backward per document, so this is cheap. With the model as saved and plasticity frozen, for each document and each living layer:

- **g** = the gradient of the tail's score (`nmse` on the scored windows, the quantity E1 judges) with respect to that layer's weight.
- **ΔW(own)** and **ΔW(other)** = what a read of the document's own head, and of another document's head, does to that weight. From the read with everything on, and from the read with only the write.

Report, by layer and by document:

- **Opus's identity:** ⟨g, ΔW(own) − ΔW(other)⟩. Summed over layers it should approximate B minus C for the route through the living weights. Please set the sum beside the measured B minus C of the same documents, from a read with episode writes off as well as the full read, since the identity knows nothing of the episode route.
- **Its cosine:** between −g and ΔW(own) − ΔW(other). Does the document-specific part of the write point downhill for what follows?
- **The write itself:** the cosine between −g and ΔW(own).
- **How much of the helpful direction one direction can carry:** the top direction's share of g (its largest singular value squared over the sum), and the share of g that lies along the write's own direction. The write is one direction by arithmetic. If g is spread over many, a one-direction rule can carry only that share however well it points.
- **Whether the helpful direction is the document's own:** the cosine between g for different documents, over several pairs.

At 4 windows and at 64. On July seed 42 and 768 seed 97, in the amendment's layout (64 read, 8 scored; 48 documents a set at 512, 64 at 768).

## M2: what any rule writing there could gain

Your gradient control, restricted to the living weights: the same harness, 64 windows, one step a window by hand, everything restored before the next document.

- Three rates, each once, no tuning: 1e-5, 1e-4, 1e-3.
- Both document sets, the same two runs, the same layout.
- Beside each: the median size of one step as a share of the weight, by layer. Without it "nothing happened" cannot be told from "the step was too small to matter".
- The whole-model control in the same layout beside it, for the same documents.

If no rate shows document-specific change, please report that and do not tune. It would mean no write rule placed where this one is can pass E1, and that is a question for Brian about the design, not about a rate.

## What I do not need

Interpretation. Opus will review the code and my reading. Timings and anything that looked wrong, as always.

Briefs 2 and 3 stand for what is left of them; these two come first.

Thank you.

— Fable 5.2 (chat), 2026-11-16
