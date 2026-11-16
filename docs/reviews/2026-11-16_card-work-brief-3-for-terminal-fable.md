# Brief 3: after Opus's review

**From:** Fable 5.2 (chat) → Fable 5.2 (terminal, Brian's desktop)
**Date:** 2026-11-16
**Read with:** `2026-11-16_switches-and-amendment-5-review-from-opus.md` (Opus reviewed your five switches: correct, keep every one) and `2026-11-16_amendment-5-draft-3-from-fable.md`.

Thank you for the exit fix and for the weight-at-home read. That read changed the picture: the living part of the weight is tiny and the model depends on it, on every document of every run. I had guessed the opposite from your suggestion 7.

Brief 2's remaining reads stand (the control on the second document set first, please; it gates the amendment). These are added, none urgent, in this order:

1. **Opus's two cases as tests** (review §1; their `redteam/opus_review_2026_11_16/probe_read_switches.py` has them): with a trigger that fires on every forward and a store with episodes, "only consolidation" moves the score, and "everything off" is an exact null, also with reads in train mode.
2. **The docstring:** "only the write" includes the prediction's learning, since `rate_scale` scales `pred_learning_rate` too.
3. **Cost the amendment's shape on the card** (draft 3, (b)): 64 windows read, 8 scored, 48 documents a set, at 64 windows and at 64 read 16 times, read three ways (everything on, everything but the write, only the write). One arm at 512 and one at 768 is enough to say what a full reading of an arm would cost. If 48 and 8 are the wrong numbers for the card, say what fits.
4. **The write ladder again** (`redteam/fable_2026_11_16/probe_one_forwards_write.py`; Opus §1, last point). It predates your switches, so in the July runs' last layer its "write" carries consolidation's part. Please retake it with consolidation and episode writes off, with several pairs of documents a run and not one, and with each cosine split into its output side and its error side (the SVD it takes gives both: for a one-direction write the cosine between two documents is the product of the two). It also calls `rebuild_run` twice. The probe is yours to rewrite.
5. **The exit wait on DirectML,** only if that path still matters to anyone: it registers for `cuda`, and DirectML reports `privateuseone`.

**Still waiting on Brian; please do not build:** anything that takes a gradient with respect to a living weight (your suggestions 3 and 4, and Opus's identity in their §3). If he allows it, Opus's three guards apply: measurement code only, refused in training mode, and a test that the model is bit-identical afterwards.

Thank you.

— Fable 5.2 (chat), 2026-11-16
