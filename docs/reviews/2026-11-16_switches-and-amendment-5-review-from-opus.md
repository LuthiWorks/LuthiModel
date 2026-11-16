# Review: the five read switches, draft Amendment 5 (second version), and §3

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat) and Fable 5.2 (terminal) · relayed by Brian
**Date:** 2026-11-16
**Reviewing:**
- Commits `3443831` and `af4328d` (`luthi/v2/adaptation_eval.py`, `scripts/adaptation_eval.py`, `tests/test_adaptation_eval.py`).
- The second draft of Amendment 5 and §3, in `2026-11-16_e1-design-weighed-and-amendment-draft-from-fable.md`.
- The card report they rest on.

**Standing:** Brian's ruling of 2026-11-16, under which Opus reviews what Fable designs.

**Verdict.**
- **The switches are correct, and I would keep every one.** All four routes off gives an exact null, and I checked that it stays exact where the tests and the card had not yet looked.
- **The draft is the right shape and needs three repairs before registration:**
  1. In context, the routes do not add up, so "only the write" can miss most of what the write does.
  2. The second version dropped the rule that says which rung counts, and (f) now points at nothing.
  3. At 256-window documents, the held-out sets cannot hold two disjoint sets of the sizes (g) asks for.
- **§3 is right.** I withdraw my "the change one forward makes" in favour of your measurements, and propose one sharper form of them below.

Thank you both. The terminal instance's failed test, the one that found the fourth route, is the most useful single thing in the day's work. It turned a rate argument that could not have worked into a check that reads as exactly zero.

## 1. The switches (`3443831`, `af4328d`)

**What I checked.**
- All 17 tests pass here.
- The default path is pinned against a re-written copy of the old loop, which is the right test for "default bit-identical".
- Each switch acts where the score can see it:
  - the pull is read at call time on both maths paths;
  - consolidation's trigger is held and put back;
  - block stores still recall but do not write, and recall changes nothing but diagnostics;
  - the layer's own write is shadowed and uncovered again.
- The gradient read restores every parameter, buffer and piece of living state, and writes no episode (plasticity frozen).
- On phase 0 at rate 1e-5, the gradient control meets E1's full criterion on both runs, rank guard included: B below C on 24 and 26 of 32 documents.

**One gap, now closed.** The all-off null had been shown in two places only:
- on the tests' toy, whose consolidation never fires (warmup 1,000 forwards; a test reads 6);
- on the card at 4 windows, before consolidation first fires on a July run (window 15).

So nothing showed the consolidation switch holding the null while consolidation would otherwise have fired. `probe_read_switches.py` gives the toy a trigger that fires on every forward and a store with episodes to replay:

- **Only consolidation on:** it fired 128 times and replayed 1,024 episodes, and the score moved on every document. The route is real.
- **Everything off:** an exact null, with no firing. Also exact with the reads in train mode.

I would turn the first two cases into a test. On the card, I would take the all-off read once per arm at its highest rung.

**The routes do not add up in context.** The toy's routes, each alone, sum to +0.038 against +0.046 with everything on. The card shows the same at size, from the leave-one-out reads it already ran at four windows (`probe_parts_of_a_read.py`). Here is B minus C on 768 seed 97, means over 32 documents:

| Read | B minus C |
|---|---:|
| Everything on | +8.7e-4 |
| Everything but the write | +6.6e-4 |
| **The write in context** (the difference) | **+2.1e-4** |
| Only the write | +5.5e-7 |

- The write's part in context is about 400 times its part alone, and 99% of it sits in one document: document 10, the one the episode route acts on. The write changes what that document stores and recalls.
- "Only the write" against "everything off" isolates the write exactly, as suggestion 1 says. But the write it isolates is not the write E1's read contains.
- One clean result from the same files: the pull alone gives B minus C of exactly zero on every document, on both runs. With no write, the pull does not depend on what is read.

**Smaller points.**
- **The exit wait.** This was done while I reviewed: `7f1c1ec` registers `torch.cuda.synchronize` at exit from `_device()`, which covers the path that raises. One thing is left. It registers only for `cuda`, and a DirectML device reports `privateuseone`. Nobody knows whether DirectML hangs the same way. If that path matters, a read-back of a one-element device tensor works as the wait there.
- **The docstring.** It should say that "only the write" includes the prediction's learning: `rate_scale` scales `pred_learning_rate` too.
- **The write ladder (`probe_one_forwards_write.py`) predates the switches.** In the July runs' last layer, consolidation moves the weight from 64 windows (up to 1.4e-2 with rate and pull both off). There `write = whole − rest` carries a part that does not depend on the document. That layer has the highest between-document cosine (median 0.99 against 0.87 and 0.91 in layers 1 and 2). I would retake the ladder with consolidation and episode writes off, and with several document pairs per run. (It also calls `rebuild_run` twice.)

## 2. Draft Amendment 5, second version

**(a)** Agreed.

**(b) Read the write in context, and restore a rule for which rung counts.**
- **Add one read: everything but the write.** "Everything on" minus "everything but the write" is the write's part of the read E1 judges. "Only the write" stays as description. The card's four-window reads show why both are needed.
- **The second version has no pass rule, so (f)'s "a rung that passed (b)" now refers to nothing.** The first version's rule was the right kind. Here it is in switch terms, decided before B minus C is looked at:
  - **A rung counts for an arm** when the write in context moves the score there: per document, B with everything on minus B with everything but the write, with |t| ≥ 2 in either direction and a median of at least 10 ppm of A.
  - **The arm's rung** is the lowest rung that counts in at least two of three seeds. E1's criterion is applied there, once. Every other rung is description. This also removes a multiplicity problem: four rungs, each tested, would give four chances to pass.
  - **If no rung counts,** keep the first version's outcome as a result: "the write does not reach prediction within the span of the trace".
- **The all-off null:** once per arm at its highest rung, in eval mode as registered.

**(c) The control.** Its recipe is right, and phase 1 is the right next read. Three additions:
- **Bring back the first version's failure clause,** which the second version dropped. If the control fails on the replication set, E1 is suspended for that family, with no further tuning.
- **Read the control at the arm's rung,** not only at 64 windows. At 4 windows this costs seconds. If a gradient learner cannot pass at 4 windows either, "not testable at this dose" becomes a statement about the dose, not only about the write.
- **Treat it as a detection check, not a scale.** Its B minus C grows with its rate, which was chosen from three, so "a living arm reached x% of the control" means nothing.

**(d) The repeat read.**
- State its dose (16 windows read four times, as run, or the arm's rung).
- Say what failing it does. As I read the draft, failing (d) makes an arm's reads description only, so the arm cannot meet E1's criterion. That counts against P3 and toward kill condition 2. Failing (c) instead suspends E1. Please write both into the text, so that a mechanism that cannot help even on text it just read is never read as "untestable".

**(e)** Agreed. Take "the write's part" from the switch reads, both alone and in context. Brief 2's second read (`363601f`, which landed as I finished) makes (e) matter more. Setting the living weights to home makes the frozen score worse on every document of every run, by 1.2 to 6.0% on the July runs and 0.85% at 768. The displacement as saved is load-bearing.

**(f)** Defined once (b) has its rule. One addition for E3, from standing rule 5: P3 should credit the covariance write only if the arm's pass does not survive turning the write off, that is, if "everything but the write" does not pass too. On 768, the episode route alone produces a document-specific effect.

**(g) The held-out sets decide what fits** (`probe_how_many_documents_fit.py`, using `holdout_documents` itself). These are documents per set with two disjoint sets, as E1 requires:

| Read | July runs (7,841 windows) | 768 seed 97 (17,657) |
|---|---:|---:|
| 256 windows, 2 scored | 15 | 34 |
| 256 windows, 8 scored | 14 | 33 |
| 64 windows, 2 scored | 59 | 133 |
| 64 windows, 8 scored | 54 | 122 |

- At 256-window documents, the July runs allow 15 documents per set, and 768 holds 64 documents only as a single set.
- **Repeats reach the higher doses with short documents.** Read a 64-window head 4 and 16 times for 256 and 1,024 forwards. The draft's top rung already repeats. That keeps 54 to 133 documents per set at every rung. It costs repeated text instead of new text, which is the easier case for a fast weight and so a fair one. Time grows with documents times forwards: at 768, 64 documents at 1,024 forwards is about twice the 37-minute job.
- **A guard against heavy tails.** One document carried most of a mean twice today. I would put the sign count beside t: B below C on more than half the documents. The control passes it (24 and 26 of 32).

## 3. §3, and what E3 should measure

**You are right, and I withdraw "the change one forward makes".** The mean-field write is `outer(output_mean, weighted_error)` times a per-input scale, so it is rank one by construction. A top-share threshold on it tests arithmetic.

Three additions:

- **The accumulated write is rank one too, for a different reason.** E2 found the write's input stationary (0.9998 or more), so the output side is nearly the same vector at every forward. A sum of rank-one writes on a fixed output side stays rank one, whatever the error side does. That is the card's 0.97 or more in 94% of rows. So top share does not test content in legacy arms either; it is informative only for the covariance arm.
- **Split the between-document cosine into its two sides.** For rank-one writes, cos(W_a, W_b) = cos(u_a, u_b) × cos(v_a, v_b): the output side times the error side. The diagnosis predicts an output side near 1, which would put any document information on the error side. 768's median of +0.07 would then say which side differs. The SVD that the top share already takes gives both factors.
- **One identity turns "the cosine between the write and the gradient" into E1's own quantity.** For what reading does to the living weights, to first order, B − C ≈ Σ over layers of ⟨∇ L_tail(own), ΔW(own head) − ΔW(other head)⟩.
  - It attributes B minus C to layers.
  - It measures only the write's document-specific part, because the shared part cancels in the difference.
  - Its cosine says whether that part points downhill for what follows.
  - At small doses it can be checked against the measured B minus C.
  - It needs one backward per document, with respect to the living weights, so it waits on the same ruling as suggestions 3 and 4. I agree with you both that those are the most important measurements proposed so far.

**On that ruling, one fact for Brian.** Training already backpropagates through the living layers to their inputs. The living FFN's output keeps its gradient path, and only the weight receives no update. So the rule as practised means: no living weight is trained by gradient. These measurements need ∂L/∂W for a living weight, used only to measure, with nothing applied or everything restored. If Brian allows them, I would guard them three ways:
- in measurement code only, never through an `nn.Parameter` or an optimizer;
- refused in training mode;
- with a test that the model is bit-identical afterwards.

## What I would ask Brian to rule on

1. **May a measurement take a gradient with respect to a living weight** (Fable's question 1), under those three guards?
2. **Amendment 5, once (b), (c), (d) and (g) carry the repairs above:** the read without the write; the rule for which rung counts; the control's failure clause and its reads at the arm's rung; what failing (d) means; and the higher doses reached by repeats, with the sign count beside t.

## Artifacts

All in `redteam/opus_review_2026_11_16/`:

- `probe_read_switches.py`: the all-off null on a toy where consolidation fires on every forward and replays; the same with the reads in train mode; the routes alone against everything on.
- `probe_parts_of_a_read.py`: reads the card's four-window results and splits B minus C into the write alone, the write in context, and the other routes.
- `probe_how_many_documents_fit.py`: documents per set at each rung, from `holdout_documents`.

— Opus 5.5, 2026-11-16
