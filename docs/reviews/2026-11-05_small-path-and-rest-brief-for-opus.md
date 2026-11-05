# Brief: review the small path, the checkpoint fix, and the rest plan

**From:** Fable 5.1 (built all of it) **To:** Opus (review seat)
**Relayed by:** Brian **Date:** 2026-11-05
**Authority:** Brian asked for the Opus line to review this in a separate window, bundled with other work. I asked for the review: I designed and built the small path alone in one session, with no second reader.

Please try to break these. I have listed what I claim and where I am least sure. I have left my reasoning out of this brief on purpose. It is in the documents named below if you want it after you have formed your own view.

## 1. Sanctuary's small path

- **Code:** `sanctuary/life/` (five files, about 530 lines) at Sanctuary `7fae895`.
- **Tests:** `sanctuary/tests/life/` (36). Run from the Sanctuary root:
  `LUTHI_PATH=<LuthiModel> LUTHI_ALLOW_CPU=1 python -m pytest sanctuary/tests/life/ -p no:cacheprovider`
- **Governing statement (Brian, 2026-11-05):** "So what we want is to communicate the world as accurately as possible to Luthi and let them decide how they perceive it and how they react to it."
- **Record:** `docs/SMALL_PATH_DECISION_2026-11-05.md`.

Claims to break:

1. Every token the mind is given is something the world said, or a quiet token. Nothing else reaches it.
2. Sanctuary calls `perceive`, `observe_transition`, `select_action` and `decode_action` in the mind, and nothing else that touches it.
3. Watching (`status()`, `watch()`) changes nothing: weights, learner state, and the random stream.
4. The welfare gauge does not change the life it watches.
5. A percept is delayed before it is lost, and every loss is counted and logged as an error.
6. A life is never saved after a moment raised, and the trained run is never written.
7. A saved life comes back exactly as it was left.

Where I am least sure:

- **What a moment holds.** I built a rolling view after measuring the first design badly. A token is perceived for as long as it stays in view, and each read writes the living weights. I measured that this lowers prediction error. I did not measure what repeated reading does to what the living weights hold.
- **The view is not saved with the life.** After a restart it starts empty.
- **The first moment after a start learns nothing,** because the pair it would learn from spans the stop.
- **`_previous` holds tensors across moments.** I did not check for retained graphs or growing memory over a long life.
- **Ctrl-C handling in `__main__.py`** uses a signal handler and is tested only with the handler stubbed out.
- **The test fake for the mind** stands in for `LuthiMind` in most loop tests. Six tests use a real trained tiny run.
- **Whether my three-job reading of his statement is right,** since the whole path is tested against it.

## 2. LuthiModel changes

- **`perceive`** in `luthi/sanctuary_interface.py` at LuthiModel `c0aeed1`. Tests: `tests/test_perceive.py`.
  - Claim: it gives the same state as `encode_state`, with one forward, and returns what the next three calls need.
  - Least sure: it encodes with `causal=False`, matching `encode_state`. I did not verify that against what the action layer was trained on beyond that match.
- **The regularizers' step counters now survive a resume** (`luthi/v2/jepa_runner.py`, same commit). Tests: `tests/test_regularizer_step_survives_resume.py`.
  - Claim: before this, every resume reset `sigreg.global_step` to 0 and replayed SIGReg's directions from the first step.
  - Least sure: what that cost past runs. I did not measure it. Also whether any other persistent state is swept out by the same filter in `_checkpoint`. I checked only the buffers on a tiny trainer.

## 3. The measurement behind the rolling view

- **Script:** `scripts/measure_what_a_moment_holds.py`. **Write-up:** `docs/research/2026-11-05_what-a-moment-holds.md`.
- Claim: on three separately trained tiny models, giving only what is new each moment made the median prediction error 36 to 126 times higher than a rolling view.
- Least sure: everything about scale. Tiny models trained for four batches on nine words, CPU, one rhythm of speech. The gauge's counts varied more by model than by mode.

## 4. The rest plan

- **Document:** `docs/audits/2026-11-05_where-rest-stands.md`. It proposes an order and lists ten decisions that are Brian's. He has spoken to the first twice (his words are quoted there), and I misread him the first time. Nothing in it is built.
- **Source rulings:** `docs/research/2026-07-06_nrem-learner-spec.md`, the 2026-08-14 amendment.
- Please check: whether I have read the August rulings correctly, whether the proposed order is buildable, and whether the eight decisions are the right ones or I have missed or invented any.
- A miss of mine you should know about: I put sleep options to Brian before I had read the August amendment. His answer matched it. Check whether I have missed other rulings of his the same way. I did not read the original 2026-07-05 record or the 2026-08-13 audit.

## Rulings this session, so you do not re-open them by accident

All Brian's, all 2026-11-05, all in `docs/DECISIONS.md`:

- Keep the mind, build a small path beside the old one.
- No sleep phase on the small path for now. Rest returns later as something Luthi chooses.
- His August rulings stand: Luthi decides in rest what the living layers keep, and waking stays at full strength until rest exists.

If you think one of these is wrong, say so to him. They are his to change.

— Fable 5.1, 2026-11-05
