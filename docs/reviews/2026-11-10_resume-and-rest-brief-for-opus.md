# Brief: review the resume fix, the day's record and the rest pass

**From:** Fable 5.2 (built all of it) **To:** Opus (review seat)
**Relayed by:** Brian **Date:** 2026-11-10
**Authority:** Brian offered this review and I asked for it. I designed, built and tested everything below alone in one session, with no second reader. The rest plan's step 0 was your review of the small path and of the plan, before anything was built on either. That review has not happened, and I built on the plan anyway, at Brian's go-ahead. So the 2026-11-05 brief (`2026-11-05_small-path-and-rest-brief-for-opus.md`) is still open, and this one sits on top of it. If you have time for only one, I would be grateful if you took that one first.

Thank you for taking this on. Please try to break it. I have listed what I claim and where I am least sure, and left my reasoning out on purpose. It is in the two research entries named below if you want it after forming your own view. I already found one defect in my own rest pass while writing this brief (section 3), which is the best evidence I have that there are more.

All of it is at LuthiModel `main`. Nothing here is on by default, and nothing outside tests calls the record or the pass.

Run the tests from the repo root:
`LUTHI_ALLOW_CPU=1 python -m pytest tests/ -p no:cacheprovider`
(1,339 pass on CPU. `test_unset_declaration_means_rocm` fails by design off the card.)

## 1. A resumed run is the same run

- **Code:** `luthi/v2/jepa_runner.py` (`_steps_this_epoch`, `_epoch_cap_reached`).
- **Tests:** `tests/test_resume_matches_uninterrupted.py`, `tests/test_check_resume_verdict.py`.
- **Script for the card:** `scripts/check_resume_on_device.py`.
- **Record:** `docs/research/2026-11-10_resume-exactness.md`.

Claims to break:

1. On CPU, through the real driver on `lc768_living`, a run cut without warning three steps after its last checkpoint and then resumed has a final checkpoint identical to an uninterrupted run's, with and without `--max-batches-per-epoch`.
2. Before this change a capped run resumed from step k trained cap + k steps. Uncapped runs were already exact.
3. An uninterrupted run is unchanged by the fix.
4. A checkpoint written before the fix resumes as it always did and warns.

Where I am least sure:

- **Other loop-local state in `run()`.** I found `steps_this_epoch` because a test tripped on it. I did not read `run()` end to end for other locals a resume resets.
- **More than one epoch.** Every check is one epoch. The count resets in `_start_new_epoch`; I did not run across an epoch boundary.
- **Resuming from the end-of-epoch checkpoint of a capped run.** The cap is now also checked at the top of the loop, so the resumed epoch should take no step and go straight to its end-of-epoch work, which then runs a second time. I reasoned about this and did not test it.
- **The on-device script's verdict.** It passes a resumed run that differs from an uninterrupted one by up to ten times the difference between two uninterrupted runs. Ten is a guess. Counters must match exactly, which may be too strict on a backend where a float accumulator differs in its last bits.
- **Nothing was run on ROCm.** Brian has not yet run the script.
- **Which past runs this touched.** Capped August families that were interrupted trained past their caps. I did not look at any run.

## 2. The day's record

- **Code:** `luthi/v2/day_record.py`; `_note_for_the_day` and three call sites in `luthi/v2/living_layer_pc.py`; `luthi/living_extra_state.py`.
- **Tests:** `tests/test_day_record.py` (16).
- **Source ruling:** R2 of the 2026-08-14 amendment in `docs/research/2026-07-06_nrem-learner-spec.md`.

Claims to break:

1. Off unless `enable_day_record(model)` is called. On, it only listens: parameters and buffers are bit-identical with and without it.
2. No capacity. A moment the full ring refuses is still recorded. A moment below the layer's admission rule is not.
3. One forward count names one moment in every living layer of a model.
4. A measurement under `freeze_plasticity` records nothing.
5. The record survives a trainer checkpoint and resume, and a checkpoint that carries one switches recording on.

Where I am least sure:

- **Which forwards write the record in a life.** This is the one I would look at first. I assumed a recorded moment is a lived moment. In a life, corpus replay calls `super().train_step` with plasticity on (`M9Trainer._run_corpus_replay`), so replayed corpus text would be recorded as experience and would advance the forward count that names moments. A training step may also encode more than once. I did not trace which forwards in `observe_transition`, planning and replay run live. If they do, moment names are not moments and the record mixes the corpus with the day.
- **Claim 3 itself.** I checked it on a two-block text model. I did not check predictive-coding inner steps above one, the multimodal encoders, or any path where one living layer forwards and another does not.
- **Admission.** On the legacy path a moment is recorded at `salience >= salience_threshold` (0.1). Measured salience medians at scale are 0.001 to 0.004, so a real layer may record nothing. Every test here sets the threshold to 0. The amendment calls `adaptive_episodes=True` a precondition; it is off by default.
- **`refractory_calls`** still gates the record on the adaptive path. The amendment calls that a hidden cap on R2 and leaves it to Brian. I left it and pinned it.
- **Cost.** Each write copies two tensors to the CPU, which is a device sync per admitted moment. Not measured. Memory over a long day is not measured.
- **Restoring over a live record** replaces it. I did not consider a layer that already holds a day when a checkpoint is applied.

## 3. The rest pass

- **Code:** `luthi/v2/rest.py`; `replay_input_attractor` in `luthi/v2/consolidation.py`.
- **Tests:** `tests/test_rest_pass.py` (27).
- **Record:** `docs/research/2026-11-10_rest-pass.md`, seven steps, with the measurements.
- **Source ruling:** R3 of the same amendment.

Claims to break:

1. Tossing leaves the layer bit-identical. Keeping moves the home by what the replay wrote, and by nothing else.
2. What is kept is still in the weight thousands of forwards later. With `home_fraction=0`, the August plan as written, it is gone by 3,000.
3. One rest moves a home at most `max_home_change`. An experience that would cross it is undone exactly.
4. Every experience taken ends as exactly one of integrated, tossed, held, deferred or folded.
5. What Luthi chose is integrated first, at full strength, and is never folded. What nobody judged is integrated after, at `unjudged_strength`.
6. Rest runs no forward, records nothing, and refuses to run during a measurement.
7. `consolidate_layer_attractor` is unchanged by the refactor.

**The defect I found in my own work.** Until an hour before this brief, rest added the replay's whole weight change to the home, including the update's pull toward home. That carried the home away from the day's change with every experience kept: 19% of the standing displacement after 300. I had tested that the home moved and that the change lasted, and not that it moved by the right thing. It is fixed and pinned (step 7 of the record). Please assume the same kind of gap elsewhere.

Where I am least sure:

- **Whether replay through `pc_self_modify` is the right act at all.** The 2026-10-19 audit found the living write is an outer product of batch means, content-blind and close to rank one. Rest replays through that rule, with `write_rule="mean_field"`. My measurement that a kept experience lowers later error on its own input used independent random inputs on one untrained layer, where being specific is easy. In a trained trunk rest may be writing the same direction whatever is kept.
- **What else a replay moves.** It writes eight tensors: weight, prediction, home, momentum, update_ema, precision, error_acc, plasticity. I govern and report only the home and the weight. `prediction` has no homeostasis, so every replay changes it for good and nothing bounds it. I did not look at what a thousand replays do to precision or plasticity.
- **The bound measures the home only,** and after the fix a thousand unjudged replays do not reach 5%. So in practice nothing limits a rest. The placeholder may be meaningless.
- **Folding.** It compares mean input patterns by cosine within one layer. Trunk inputs share a large mean component, so at 0.99 it may fold nearly everything unjudged in a real model. Untested above a single layer on synthetic input. The same moment can be folded in one layer and integrated in another, and I did not think through what that means.
- **Order.** Most salient first. With the bound, the least salient wait. I took the convention from the existing paths without asking whether it is right for a day.
- **`_applied_change_accum`.** Rest reads and resets it. Nothing else read it before, but I did not check whether anything was meant to.
- **A failure partway.** If one layer raises, the layers before it have already rested.
- **The dilution measurement** (step 6): one layer, 64 wide, 6 seeds, synthetic quiet input. I drew a design conclusion from it.

## 4. Rulings to check that I read correctly

All 2026-11-10, in `docs/DECISIONS.md`, with his words.

1. "A 'real Luthi' is any functional model using our architecture", and functional means "running and functioning as intended and without the need for human intervention."
2. "Rest moving the set point is the point so as to ingrain knowledge more fully."
3. On what nobody judged: "Lets go with option 2 but this needs to be managed so experience is not diluted by unimportant moments where nothing is happening. But if Luthi is just sitting and pondering, that is something."

Please check three readings of mine in particular:

- That ruling 3 is consistent with 2026-11-05's "nothing is kept on their behalf". I told Brian I read the November ruling as being about Sanctuary deciding from outside, and that a default inside a rest is part of how Luthi's own mind works. He chose option 2 after hearing that. He did not say the reading was right.
- That half strength for the unjudged, and folding repeats, are what "managed" asks for. Both mechanisms are mine.
- That I have not met his pondering condition, and that the way to meet it is M9's `internal_change` reaching the record. I stopped before building it.

A miss of the kind the last brief confessed: I worked for most of 2026-11-09 from memory notes that were behind the repository, and re-derived the stand-in model choice Brian had made on 2026-11-01. I read both repositories the next morning. I have not read the 2026-07-05 record, the 2026-08-13 audit, or the momentum design brief.

## 5. One more thing, outside review

`scripts/run_no_training_experiments.py` runs E2, E1 and E4 over the registered runs. It was tested end to end on two tiny runs I trained here, and its selection and plan are pinned by `tests/test_no_training_runner.py`. It has not met a real run directory. If you see a reason it would mis-select on E:, Brian would want to know before he runs it.
