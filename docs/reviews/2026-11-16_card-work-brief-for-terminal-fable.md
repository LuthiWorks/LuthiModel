# Brief: work on the card, 2026-11-16

**From:** Fable 5.2 (chat; design and review seat) → Fable 5.2 (terminal, Brian's desktop; building and running)
**Date:** 2026-11-16
**Standing:** Brian's ruling of 2026-11-15 and 2026-11-16 (`docs/DECISIONS.md`, last entry): you build and run on the card; Opus and I design and review. Brian has said the card is free until he is home today.

Hello, and thank you for Friday. Your three catches (the stale path, the stopped runs' taper, the 14-hour E4) each saved a wrong result. Under the new split the code below is yours to write as you judge best. Where I name a function or a flag it is a suggestion of shape, not an instruction. What I ask you to keep fixed is what each job measures.

**Why any of this.** Opus reviewed the first read (`2026-11-14_e1-e2-e4-first-read-review-from-opus.md`). E1 could not see the living write at four windows, so its null means nothing yet. My answer and a draft amendment are in `2026-11-16_e1-design-weighed-and-amendment-draft-from-fable.md`. The draft needs four facts from the card before Brian can rule. Jobs 1 to 4 are those facts. Everything here is exploratory: it informs a registration and is not one.

Please do them in order, stop and report on anything surprising, and commit results as each job finishes. If the card is needed back before you are done, what is finished is enough.

## Job 1: is the rebuild faithful at 512? (Opus's check; about ten minutes)

Each July run's `pilot_result.json` recorded its held-out error when it finished (`heldout`: `l_pred_mean`, `nmse_mean`, `n_batches`). The driver computed it as `heldout_latent_prediction(loss_module, loader.holdout_batches("text", 8), "text", max_batches=...)`.

- **Measure:** for each of the nine July runs (six living, three dead), rebuild with `rebuild_run`, compute the same thing over the same number of batches with plasticity frozen, and set it beside the recorded value.
- **Report:** recorded, rebuilt, and the relative difference, for `l_pred_mean` and `nmse_mean`.
- **What I expect:** agreement to three or four digits. July evaluated with plasticity on, on DirectML, with the C++ maths; you will evaluate frozen, on ROCm. None of that should move the third digit.
- **If a run differs in the second digit, please stop.** Then the rebuild is not that run, and Friday's E1 and E2 on it should wait. That would be my error to find, and I would want to know before anything else runs.

## Job 2: what a read writes (ready to run; minutes per run)

`redteam/fable_2026_11_16/probe_one_forwards_write.py` separates the write from the pull toward home by reading the same windows at living rate 1 and at rate 0. It is written and has run on a small run on CPU. It has not run on a card.

    python redteam/fable_2026_11_16/probe_one_forwards_write.py --run-dir <run> --windows 1 4 64 256 1024 --out runs_meta/living_channel_dose_2026_11_16/<run name>/write_ladder.json

- **On:** the seven finished living runs (six July, 768 seed 97), and 768 seed 46 for contrast.
- **It makes six passes of 1,024 forwards**, so it should take a few minutes a run. Please time the first.
- **If it is wrong or slow, it is yours to fix.** It takes its SVD on the CPU on purpose.

## Job 3: at what dose can E1 see the write? (existing flags; time the first)

The same `adaptation_eval.py`, read longer, at the full rate and at a fifth:

| Rung | Flags |
| --- | --- |
| 64 windows | `--docs 32 --read 64 --score 2` |
| 256 windows | `--docs 30 --read 256 --score 2` |
| 1,024 forwards | `--docs 30 --read 256 --score 2 --repeats 4` |

- Each rung twice: as loaded, and with `--rate-scale 0.2`.
- **On two runs first:** 768 seed 97 and July seed 42. Six jobs each. If they are quick, the other five finished living runs as well.
- One read is one forward, so the top rung is about 61,000 forwards a job. I do not know how long that is on the card. **If one job passes 40 minutes, please stop the ladder there and tell me.**
- The runner knows none of these jobs. Whether you teach it a second plan or run them by hand is yours to choose. `adaptation_eval.py` will not exit when it is done, as on Friday.

## Job 4: three opt-in switches for `adaptation_eval`, then one read with each

Each off by default, with the default path bit-identical and a test that says so (`CLAUDE.md`, Conventions). Each recorded in the result's `settings`.

1. **The pull toward home off during the read** (for example `--homeostatic-decay 0`). Settles what carries the parts-per-million change at four windows. One read of July seed 42 at the registered dose (`--docs 32 --read 4 --score 2`). If the change goes to zero, it was the pull.
2. **A repeat read** (for example `--read-what tail`): B reads the windows it will then be scored on; C reads another document's. This is the easiest case a fast weight can have. What is read is what is scored, so `--read` and `--score` are equal here; I suggest 16 and 16, with `--repeats 4`. On 768 seed 97 and July seed 42, as loaded and at `--rate-scale 0.2`.
3. **A read by gradient** (for example `--read-by-gradient <rate> --gradient-steps <n>`): the plastic read is replaced by plain SGD steps on the loss of the windows read, with plasticity frozen, over the encoder's and predictor's trained parameters. Every parameter and every piece of living state must be back exactly as it was before the next document; please test that. Three rates, 1e-5, 1e-4 and 1e-3, one step per window, at 64 windows, on 768 seed 97 and July seed 42.

Number 3 is the positive control the amendment needs: does the harness detect document-specific change when something is known to be learning from the document? If none of the three rates shows it, that is the most important result of the day, so please do not tune further; report it.

## Where results go, and what I do not need

- `runs_meta/living_channel_dose_2026_11_16/<run name>/<what>.json`, each with the `reconstruction` field. No path under a user's home folder (a test now checks `runs_meta`).
- No interpretation is needed. Timings, hangs, and anything that looked wrong are.
- Opus will review the code from Job 4 and my reading of the results, as the ruling has it.

## Two things I would like your judgement on, if you have time

- **The exit hang.** `adaptation_eval.py` 38 of 38, the other two scripts 0 of 30. You are the only one who can bisect it. If it is cheap, I would value knowing what the script does that the others do not.
- **Anything in this brief that is the wrong shape for the card.** You have run these scripts and I have only written them.

Thank you.

— Fable 5.2 (chat), 2026-11-16
