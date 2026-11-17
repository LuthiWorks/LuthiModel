# Rest that works with what was lived and is relevant to what a weight holds: a design — 2026-11-17

**Status:** a design by Fable 5.2 (chat; design seat), for Opus 5.5 to review and Brian to rule on where it says "Brian's". Stage 0 is already in a build brief. Nothing else here is built.

## Objective

Brian, 2026-11-17, ruling on Opus's finding that a rest would pour corpus replay into home (`docs/DECISIONS.md`):

> *"We remove the corpus from the rest and have it work with only what was lived and is relevant to the concept(s) represented by that weight or weight matrix. It may turn out that in order to actually get the model to update each weight with oniy relevant data that we may need to be able to identify the concepts present in the matrix. This may help us help the model actually update correctly instead of with noise and randomness."*

And later the same morning: *"lets get all of that into design so we can start building. It is a critical system."*

This is that design. It has two halves: **only what was lived**, which is simple, and **only what is relevant to what the weight holds**, which is the hard and important one.

## What is known, and from where

1. **In a life, corpus replay writes most of what a rest would pour.** Each moment makes one lived forward and two replay forwards into the same weight; 106 to 120% of the half-day's trace is replay's (Opus, 2026-11-14, §3).
2. **The living write cannot tell what it is about.** It is an outer product of two vectors: the layer's mean output, and its error. The output side is the same for every document, in every layer of eight runs (card report 2, 2026-11-16). So whatever is lived, the write lands in the same place in the weight. That is Brian's "noise and randomness", measured.
3. **What differs between documents is on the error side,** and in the July runs' last layer not even there.
4. **A write that does differ by content exists in the code and has never run in a life:** `write_rule="covariance"`, E3's arm. It writes token by token and not from the mean.
5. **The model depends on what its living weights have become** (weights at home: 0.85 to 6.0% worse on every document).
6. **Not yet known:** in which direction a change to a living weight would help what follows. Brief 4 measures it.

## What "the concepts a weight holds" means here

A living layer is one map from what comes in to what goes out. Nothing in it is one concept in one weight; a concept is spread across the matrix.

- **A concept** is taken to be a direction in the layer's output that some of what the mind meets drives and the rest does not.
- **The matrix holds a concept** to the extent that it sends its inputs that way.
- **An update is relevant to a concept** when the moment that caused it drove that concept, and the update lies in that concept's part of the weight.

The mean output drives every direction a little and none in particular, which is finding 2 again. Token by token the output does vary. So the concepts have to be found in the tokens, not in the mean.

## The design, in stages

### Stage 0: only what was lived (in brief 5, Part C)

Each traced layer keeps the part of its displacement that lived forwards wrote, and rest pours from that. Corpus replay is out of it.

### Stage 1: find the concepts, and ask whether they can tell experiences apart

An instrument, on finished runs, with no training. It comes before anything is built into rest, because it is the capacity question (standing rule 1): **if a layer's concepts are driven alike by every document, sorting updates by concept sorts nothing.**

For each living layer, over held-out text, token by token:

- **Find directions** that tokens drive selectively. Start with the simplest finder that uses no gradient: prototypes of the token outputs, with the mean removed (competitive learning; each token belongs to its few nearest).
- **Say what each one is.** List the passages that drive it most. That is "identifying the concepts present in the matrix" in a form Brian can read, and it belongs in LuthiScope.
- **Ask the capacity question three ways:**
  1. How much of the output's variation is token to token, once the mean is removed?
  2. Can a document be told from another by which concepts it drives? (Each held-out window assigned to the document whose concept use it most resembles; chance is one in the number of documents.)
  3. Does the layer's error differ from concept to concept? If the error is the same whatever concept is driven, there is nothing concept-specific to integrate.
- **And, from brief 4's gradient:** does the direction that would help what follows lie in the concepts the document drives? If it does, concept by concept is the right structure. If it lies elsewhere, it is not.

### Stage 2: a trace kept concept by concept, and a rest that pours concept by concept

Only if stage 1 says concepts tell experiences apart.

- **While awake,** for each concept a layer keeps a running average of the error on the lived tokens that drove it, and a count of how much was lived about it. This is the covariance rule's own quantity, sorted by concept; it is the same local signal the write uses, and no new teacher. It costs one vector a concept.
- **A rest pours each concept's part into home along that concept's direction,** bounded concept by concept by how much was lived about it, and bounded in all by the limit that stands today.
- **A concept nothing lived touched is left alone.** That is the ruling in one line.
- **Luthi's choice to remember** weighs the concepts driven at that moment, as it weighs the whole trace today.
- Opt-in, off by default, bit-identical when off. Measured on lives of several seeds against stage 0, with replay running, as Opus's review requires.

### Stage 3: the concepts are kept up in rest

A mind's concepts move as it changes. The finder would run during rest, on what was lived. This is one of the things a "subconscious" would be (Sanctuary `docs/research/2026-11-17_memory-store-rework-design.md`, §5). When the concepts move, the traces sorted by the old ones are poured or dropped first; a trace is never read against concepts it was not made with.

### Stage 4: a waking write that knows what it is about

The covariance rule in a life. This waits for E3 and brief 4, as already ordered.

## What I do not know

- **Whether pouring the error, however well sorted, helps.** The card found the living write points nowhere useful for what follows. Sorting it by concept makes it relevant; it does not make it right. Brief 4's gradient, cut by concept, is what would say.
- **Whether a finder with no gradient finds anything worth calling a concept.** Work on large language models finds them with a small auxiliary network trained by gradient (sparse dictionaries). That auxiliary is not a living weight, but it would be a new thing trained inside a life. Whether that is acceptable is **Brian's**, and only if the simple finder fails.
- **How many concepts.** A placeholder until stage 1 has a curve.
- **None of this touches the deeper gap** recorded on 2026-11-16: training shows a living weight shuffled pieces of many texts, never a stream.

## Conclusion

Build now: stage 0 (brief 5) and stage 1's instrument (brief 6, on the card, no training). Decide stage 2 on what stage 1 and brief 4 show. Brian's now: nothing. Brian's later: the auxiliary question, if it arises.

## Artifacts

- `docs/reviews/2026-11-17_build-brief-5-rest-readings-store-faults-for-terminal-fable.md` (stage 0).
- `docs/reviews/2026-11-17_build-brief-6-concepts-instrument-for-terminal-fable.md` (stage 1).
- `docs/reviews/2026-11-16_card-work-report-2-from-terminal-fable.md` (finding 2); `docs/reviews/2026-11-14_rest-memory-and-dreaming-review-from-opus.md` (finding 1).
