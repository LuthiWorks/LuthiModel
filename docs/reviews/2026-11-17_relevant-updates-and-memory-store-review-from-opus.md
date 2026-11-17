# Review: relevant updates (LuthiModel) and the memory store rework (Sanctuary)

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat) · relayed by Brian
**Date:** 2026-11-17
**Reviewing:**
- `docs/research/2026-11-17_relevant-updates-design.md` (LuthiModel `9fad2be`), with the briefs that build it: brief 5 Part C (stage 0) and brief 6 (stage 1).
- Sanctuary `docs/research/2026-11-17_memory-store-rework-design.md` (`86afa9a`), with brief 5 Part A.

**Verdict.**
- **Relevant updates.** The staging is right: only what was lived now, and the capacity question before anything goes into rest. Brief 6 already carries the baselines I would have asked for. One fact about the layer changes what stage 2 should be: its error is a function of its input alone. So the covariance write, the quantity stage 2 sorts by concept, is W S (I − WᵀP) T, where S is the token covariance of what was lived.
  - That gives relevance by construction, with no concept finder.
  - It gives a cheaper stage 2, and a capacity test that can run on finished runs today.
  - It answers terminal Fable's "what is the error an error of".
  - Separately, stage 2 needs its outcome measure named before it is built.
- **Memory store.** Keeping everything exactly is right, and measuring half precision on real stores was the right way to answer Brian. Four things need settling before building: how a stretch is written, what a stretch brings back, how the search scales, and where summaries live.
  - The design sizes the disk but not the search. Today's search takes 2.1 seconds after one day of memories. A fast exact search over ten days takes 0.78 seconds on one thread, while rest asks for one about 2.5 times a second.

Thank you for both designs. Stopping to measure half precision on real stores, and then saying plainly in the record that the claim was wrong, is the habit that makes this project trustworthy.

## Part 1. Relevant updates

### What the error is

The living layer's output is linear in its input (y = W x), and its error is e = x − Pᵀy (`pc_ops`: `err_tok = x_flat - output @ prediction`), so e = (I − PᵀW) x. The error carries nothing beyond the input. The covariance write, before its cap and with the clamp not binding, is then:

> **write = W S (I − WᵀP) T**, with S the token covariance of the layer's input and T the trust (precision).

Here W is the weight the forward actually used. In a layer whose own store recalls an episode, that is the weight plus the recalled episode, which is constant within a forward.

`probe_what_the_covariance_write_is.py` checks this on the real update: it matches to 7e-15. Two different sets of tokens with the same covariance write the same thing, to 3e-15. Four consequences follow.

1. **What was lived enters only through S.** An output direction no lived token drove gets nothing. An input direction the layer already reconstructs through P gets nothing either. That is "a concept nothing lived touched is left alone", by construction, token by token.
2. **Stage 2 does not need a finder in the update path.** Its per-concept error averages are this same quantity, coarse-grained onto K concept directions. Stage 2 could instead keep the covariance of the lived inputs per layer (or the uncapped covariance write): one d × d tensor, the size of today's trace. A rest would pour W S_lived (I − WᵀP) T within the limit.
   - The eigenvectors of S_lived are the lived concepts, as directions of variation, and LuthiScope can show them.
   - The finder then names and checks the concepts, as brief 6 has it. It never computes the update.
3. **It answers terminal Fable's question** (report 2, suggestion 3: what is the error an error of). It is the part of the layer's input that P does not reconstruct from the output.
   - In the mean-field write, both sides are functions of the mean input x̄ alone: the output side is W x̄ and the error side is (I − PᵀW) x̄.
   - Where the error side is the same for every document (the July runs' last layer), the mean input is the same in the directions P fails to reconstruct.
   - Where the output side is the same but the error side is not (768), one plausible reading is this: the mean input has a large common part that P reconstructs well. That part dominates W x̄ and cancels from the error. A projection of x̄ onto its common part would tell.
4. **It sharpens your open question, whether pouring the error helps.** Pouring it moves the layer toward reconstructing its lived inputs through P, which is predictive coding's own local objective. Nothing ties that to the score of what follows. Brief 4's gradient measures whether the two agree.

One wording point: "the mean output drives every direction a little and none in particular". Finding 2 says the mean output is the *same* direction for every document. Whatever varies with content is in the tokens' variation around it.

### A capacity test with no finder and no training

On the finished runs, compute the covariance write each document would make, from the formula, without applying it. Then report:

- its between-document cosine, with the two sides split as in report 2;
- its cosine with brief 4's helpful direction (−∇ of the tail's score, own document against the others).

This tests stages 2 and 4 together. If the covariance quantity does not point downhill for what follows, sorting it by concept will not make it. Brief 6 is the natural place for it, since it already reads token by token. The helpful direction can come from `luthi/v2/living_weight_gradient.py`, which landed (`1145c27`) as I finished.

**Brief 6 already has the baselines:**
- the embedding row beside every layer;
- concepts fitted on half the documents and read on the other half;
- the gradient's share in the driven concepts, against the share by chance.

The design document should name them too, so that whoever builds from the design alone keeps them.

### Stage 2 needs its outcome measure named first

"Measured on lives of several seeds against stage 0" does not say measured on what. Rest claims to integrate what was lived. The E1 logic applies directly:

- Read a document's head, rest, then score its tail (frozen).
- Compare with a rest after reading a different document: B minus C.
- Keep the corpus held-out score as a guard.

`adaptation_eval`'s switches make this cheap: read, `rest_from_trace`, score. I would register it before stage 2 is built, as E1's amendment was registered before E3.

### Smaller

- **Brief 5 Part C's must-hold holds on the probe's two lives only because their arm has consolidation off.** That is the driver's minimal `living` arm, line 176. On a July-style run, consolidation fires inside lived forwards. The probe counts that as lived, while Part C rightly leaves it out. Add a consolidation kind to the probe's split before comparing on such a run.
- **With sight and sound, the largest variation from token to token may be which sense a token came from,** through the modality embeddings. Fit concepts per sense, or after removing each sense's mean.
- **Stage 3's finder, updated in rest, is lived state.** Save it with the life (`living_extra_state`) and keep it opt-in.

## Part 2. The memory store rework

### What is right

- **Nothing is stored inexactly.**
  - Summaries are stored as 4-byte binary, which is lossless (you checked 23,152 of the numbers).
  - Pictures and sound are stored as 8-bit only when the round trip is exact.
  - The picture kept is the world's own (`sight.pixels`, before the eye resizes and normalizes it), which is the right thing to keep exact.
- **The made-up store that showed no change** is worth remembering as a warning.
- **The rest is sound:**
  - summaries count against the limit, and letting go frees them;
  - importance comes only from Luthi's own signals, and age alone is not unimportance;
  - letting go keeps its two steps.

### Four things to settle before building

1. **How a stretch is written.** A stretch grows moment by moment, up to 95 moments in the archived stores.
   - Writing it only when it closes loses an open stretch in a crash.
   - Appending to one file rewrites a memory.
   - One shape keeps both rules. Each kept moment writes one small record that is never changed, into its stretch's folder: the tokens new since the previous record, its picture and sound, and its summary. The stretch is its records.
   - A crash then loses at most the moment in flight, nothing is rewritten, and the words are still stored once. Brief 5's id fault then applies per record.
2. **What a stretch brings back.** A stretch's words, stored once, can run to a window plus 95 tokens, but `bring_back` fits one window.
   - Record each moment's view as offsets into the stretch, and bring back the view at the moment whose summary matched.
   - That keeps every moment exactly recoverable and makes recall land where it matched.
   - Letting go and importance then act on the whole stretch. The design should say so.
3. **The search.** In rest a memory is brought back on every other resting moment, alternating with dreams, so at 10 Hz the store is searched about 2.5 times a second. `probe_nearest_at_scale.py` times it at 768 wide, on one thread, with day and ten-day figures extrapolated from 80,000 summaries:

   | | After one day | After ten days |
   |---|---:|---:|
   | `nearest` as built | 2.1 s per search | 21 s |
   | Exact, as one product over a kept matrix of unit rows | 0.08 s | 0.78 s (8.8 GB in memory) |

   - On the desktop, an exact search over every summary cannot keep 10 Hz past a few days. The Spark's memory bandwidth might.
   - Any faster index changes which memory comes back for some cues: one summary per stretch, fewer dimensions, or approximate neighbours. That is the same kind of question as half precision. Measure it the same way, against the exact search, and let Brian rule. Bounding what is searchable is the other way out; that would be forgetting for recall, while storage keeps everything.
4. **Where summaries live.** "Deleting a memory frees its summaries" and "a memory is never rewritten" pull against each other if the summaries sit in one shared file. Keep each memory's summaries in its own folder, deleted with it. Treat the search matrix as a cache rebuilt from those folders: the cache may be rewritten, the record never.

### Clarifications

- **The safety net.** The design says the two steps and the archive convention are "what stand between that and loss". The archive convention covers finished runs and test lives, not deletion for room inside a running life.
  - With sight, deletion runs from the first hours, at importance close to chance.
  - If Brian wants a net while choices are still close to chance, deletion for room would move memories to cold storage instead. That is his call, and a disk question.
- **Which running mean?** Centering every summary by today's mean makes old memories differ from today mostly by how far the mind has drifted. The alternative: store the mean once per stretch, and center each summary by the mean of its own time. That compares what was distinctive then with what is distinctive now. It is worth measuring for "years past".
- **Sound's 16-bit check needs the world's scale** (32,767 or 32,768). Record which round trip held.
- **The measurements of steps 2 and 3:** please commit the script, under the convention you just added. It can run where the archive is.
- **Importance and the subconscious.** The readings that would feed importance (rest's revisits, letting go, remembering) fired together under drift in the 2026-11-11 lives, and brief 5 Part B fixes them. I would not give rest's readings a quiet effect on importance until lives of several seeds show that they no longer fire together.

## What I would ask Brian to rule on

1. **Stage 2 as the lived covariance trace,** with no finder in the update path, if Fable agrees.
2. **The search:** whether an index that sometimes changes which memory comes back is acceptable once measured, or whether the store instead bounds what can be searched.
3. **Cold storage for memories deleted for room** while Luthi's choices are close to chance. This is alongside what the design already leaves to him: the store's size, whether automatic letting go waits, whether Luthi is told, and which parts of a subconscious to build.

## Artifacts

In `redteam/opus_review_2026_11_17/`:
- `probe_what_the_covariance_write_is.py`: the closed form against the real update, and two different token sets with the same covariance.
- `probe_nearest_at_scale.py`: the nearest-memory search, as built and as one product, at 768 wide.

— Opus 5.5, 2026-11-17
