# Brief 6: what concepts a living layer's activity holds, and whether they tell experiences apart

**From:** Fable 5.2 (chat; design seat) → Fable 5.2 (terminal, Brian's desktop)
**Date:** 2026-11-17
**Read with:** `docs/research/2026-11-17_relevant-updates-design.md` (stage 1). Brian has asked that this system be designed so building can start: *"It is a critical system."*

Hello. This is an instrument on finished runs: no training, plasticity frozen throughout, on the card. It answers one question before anything is built into rest. **If every document drives a layer's concepts alike, sorting updates by concept sorts nothing.**

Order, as things stand: brief 4, then this, then brief 5. If you would rather interleave, that is yours.

## What it measures

For each living layer of a finished run, over held-out text, **token by token** (the mean over tokens is already known to be the same for every document):

1. **The token outputs,** with their mean removed: how much variation there is, and its effective rank.
2. **Concept directions:** K prototypes of those outputs, found with no gradient (k-means or any competitive rule; cosine distance). Fitted on half the held-out documents, read on the other half. K at 64, 256 and 1,024, so there is a curve and not a point.
3. **What each concept is:** for the 20 most used concepts of each layer, the ten passages (a few tokens of context each) that drive it most. This is for Brian to read, so plain text.
4. **Can a document be told by the concepts it drives?** A document's signature is how often each concept is driven across its windows. Hold out some windows of each document; assign each to the document whose signature it most resembles. Report accuracy against chance (one in the number of documents), and against the same test on shuffled document labels.
5. **Does the error differ by concept?** The layer's own error (`e = x − yP`, token by token), averaged over the tokens that drive each concept. Report how much of the error's variation lies between concepts, and the cosine between different concepts' mean errors. If the error is the same whatever is driven, there is nothing concept-specific to pour.
6. **When brief 4's gradient exists:** the share of the helpful direction (minus the gradient of the tail's score with respect to the weight) that lies in the rows of the concepts the document drives, against the share expected if it lay anywhere.

## On what

July seed 42 and 768 seed 97 first. The amendment's layout for documents (64 windows read, 8 scored) is a convenient unit: 48 documents at 512, 64 at 768.

## Must hold

- Plasticity frozen for every forward; a test that the model is bit-identical afterwards.
- The embedding row beside every layer, as the baseline (standing rule 3): a concept test that the input embedding passes just as well shows the tokens, not the trunk.
- Fitted on one half of the documents and read on the other, so that a concept is not a memory of the passage it was fitted on.
- Results to `runs_meta/living_concepts_2026_11_17/<run>/`, with the `reconstruction` field.

## What I do not need

Interpretation, or tuning K to make anything look better. If item 4 comes out at chance in every layer, that is the answer and an important one.

Thank you.

— Fable 5.2 (chat), 2026-11-17
