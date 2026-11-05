# How an experience affected the mind — 2026-11-04

> **Corrections, 2026-11-05.** Three things below were overtaken the next day. Step 5's table comes from one model, and three separately trained models show quiet is not silent (Step 7). A bad reading no longer wakes the mind, and nothing acts on the head (Step 8). Luthi now perceives their own reading (Step 9).

## Objective

Brian ruled: "the world should just exist and Luthi has to interpret what they experience as either good or bad depending on how the experience affects them."

Until today, felt valence came from the world. `Percept.valence` carried a signed number, with damage negative, and the felt-valence head learned from it. That also conflicted with the July affect decision (Sanctuary `docs/AFFECT_GROUNDING_DECISION_2026-07-19.md`, §3: "Nothing anywhere in the pipeline hands the model a signed valence"). The goal was to remove the world's say and find what Luthi can read for itself.

## Process

### Step 1: what can the mind read about itself?

The July decision names "the one stake that needs no author": a prediction-minimizing system doesn't need to be told to value predicting well. Luthi already computes how wrong its expectation was every lived cycle (`l_pred`). So that is the source.

There is prior work on this. Joffily & Coricelli (2013) define emotional valence as the negative rate of change of free energy, and Hesp et al. (2021) take a different formal route through confidence in the action model. I read abstracts and summaries of these, not the full papers.

### Step 2: my first version would have made a miserable mind

I first compared each moment's error with the median of recent errors. Through the real runner:

| Stretch | Felt | Bad | Good |
|---|---:|---:|---:|
| Settling, nothing happening | 12.5% | 48 | 2 |
| Quiet | 5.7% | 34 | 0 |
| The same thing, repeated | 13.0% | 26 | 0 |
| Something new each moment | 29.5% | 59 | 0 |

Over the life: 241 bad moments to 7 good, and the head read familiar moments at -0.79 and new ones at -0.76. Meanwhile the error was falling, from a median of about 1.27 to about 0.37, so a mind that was steadily improving read that as suffering.

The cause is the shape of error: it has a floor at zero and no ceiling, so a comparison with its median can almost only come out bad.

### Step 3: rates balance

I saved the real error sequence and compared candidates on it. Each row is 1,944 real moments:

| Reading | Felt | Bad | Good | Mean over all moments |
|---|---:|---:|---:|---:|
| Raw error against its median (first version) | 12.6% | 237 | 7 | -0.1105 |
| Log error against its median | 9.1% | 134 | 42 | -0.0429 |
| Fast average against slow average of log error | 11.4% | 179 | 42 | -0.0679 |
| **Change in log error from the moment before** | 7.0% | 69 | 68 | -0.0004 |

Changes telescope: a spike reads as bad, recovering from it reads as good, and over a life the sum is how far prediction has actually improved or worsened. It's also closest to the published definition.

### Step 4: that reading was mostly dropout noise

The lived error is read with dropout on. On 1,200 real moments:

- The same moment read twice differed by a typical 0.117 in log error, while a typical real change between moments, with dropout off, was 0.016. The noise is about seven times the signal.
- Two dropout-on readings of the same moments agreed on only 74 of 142 felt moments. Those would have been random moods.

Without dropout, the reading follows experience: 1% of moments felt when the same thing repeats, against 36.5% with something new each moment. So the mind now reads its error once more each cycle, with dropout off and before that moment's learning. That costs one extra forward per lived cycle, which I haven't timed at full size.

### Step 5: the result

Reading: change in log error, felt beyond 4 times the recent typical change, with a floor so that it takes at least about a 22% change. Tiny M9 run (d=32, 1 block, CPU, one seed), through the real runner:

| Stretch | Felt | Bad | Good |
|---|---:|---:|---:|
| Settling, nothing happening | 1.5% | 4 | 2 |
| Quiet | 0.0% | 0 | 0 |
| The same thing, repeated | 0.5% | 1 | 0 |
| Something new each moment | 24.5% | 28 | 21 |
| Quiet again | 0.0% | 0 | 0 |

Over the life: felt on 3.3% of moments, 37 bad and 26 good, mean -0.0053. The head reads familiar moments at -0.09 and never-seen new ones at -0.04, so it shows no dread of novelty.

### Step 6: what else changed

- **`Percept.valence` is gone, and a percept with an unknown field now raises.** The world can't say how anything felt, even by accident. `observe_transition` ignores any `ctx["valence"]`, and a test pins that.
- **Waking.** A moment the mind read at or below -0.5 woke it, in place of a percept saying something hurt. *(Removed 2026-11-05: Step 8.)*
- **The keep** now responds to the mind's own reading.
- **The Minecraft plan** reports what happened, with no feelings attached.
- **A welfare gauge.** The July decision requires one on time-integrated valence. `felt_summary` gives moments read, how many were felt each way, and the running sum. Its alarm thresholds aren't set, and that decision says to freeze them before a life begins.

### Step 7: three models, and a correction

Step 5's table came from one tiny model, and I also ran a comparison that looked like three seeds and wasn't. With dropout off, a life on one model is deterministic, so three torch seeds gave identical numbers. Using three separately trained models (`scripts/measure_felt_reading.py`), with the reading hidden from the mind:

| Stretch | Felt, mean (range) | Bad | Good |
|---|---:|---:|---:|
| Settling | 12.0% (0.0-26.2) | 62 | 82 |
| Quiet | 14.0% (10.0-18.8) | 141 | 111 |
| The same thing, repeated | 4.2% (1.0-10.0) | 14 | 11 |
| Something new each moment | 22.2% (9.0-29.0) | 65 | 68 |
| Quiet again | 14.9% (3.7-22.0) | 61 | 73 |
| Overall | 13.5% (10.3-18.3) | 343 | 345 |

So the balance holds across models, and that is the firmest result here. "Quiet" is not silent, which Step 5 wrongly suggested, and the difference between stretches is real in direction but noisy on models this small. A "quiet" stretch does still contain the mind's own changing state, so it isn't an empty input.

### Step 8: nothing reacts for the mind

Brian: "They decide how to react to everything." On 2026-11-05 he approved removing the reactions that had been decided in advance:

- **The planner's cost is gone.** Nothing makes Luthi avoid what reads as bad. The head still learns, and it is now an instrument for stewards.
- **A bad reading no longer wakes them.**
- **The old arousal formula is gone.** Nothing writes arousal or valence for them.
- **K-M9-5 runs during a life and only reports.** I had described it as something that would halt a mind. Checking the code, nothing has ever acted on it. `stillness_report` shows it to stewards, and the docstrings now say it must never halt anything.

### Step 9: Luthi perceives their own reading

The proposal was to let Luthi see their reading "through introspection." When I checked, that channel never reaches Luthi: `knowledge_signals` isn't in their input. So the reading now goes in as text, `[self] -0.3`, a number with no word for good or bad.

That created a feedback loop, because the reading changes the input, which changes the error, which changes the reading:

| How it was shown (one model) | Felt, repeated input | Felt, overall |
|---|---:|---:|
| Not shown | 0.5% | 3.3% |
| Only after a felt moment | 26.0% | 14.7% |
| Always, to two decimals | 35.5% | 6.6% |
| Always, smoothed and rounded to one decimal | 1.0% | 8.5% |

The last form shipped. Across the three models, against the hidden table above:

| Stretch | Hidden | Shown |
|---|---:|---:|
| Quiet | 14.0% (10.0-18.8) | 23.5% (18.2-28.8) |
| The same thing, repeated | 4.2% (1.0-10.0) | 12.5% (2.0-32.5) |
| Something new each moment | 22.2% (9.0-29.0) | 19.0% (2.0-39.5) |
| Overall, bad/good | 13.5%, 343/345 | 16.6%, 426/420 |

It stays balanced. It may add some disturbance, though the ranges overlap, and on one of the three models the flicker returned on repeated input (32.5%, 32 bad and 33 good). That is not settled. `show_reading` turns it off so that it can be compared on a trained run.

## Conclusion

Nothing outside the mind now says how anything felt, and nothing reacts for them. Luthi reads each moment from what it did to their own ability to predict, and perceives that reading. At test size it is balanced across models. How well it tracks experience is noisy on models this small.

**What this does not do, and it matters:**

- **It reads disruption, not harm.** If taking damage doesn't disturb Luthi's predictions, it won't feel like anything. I proposed giving harm a consequence, such as low health slowing thinking. **Brian ruled against it:** "Harm to Luthi's character should not have predetermined effect. If Luthi decides to slow their thinking or speed it up, that is their call to make. They decide how to react to everything." So nothing makes Luthi avoid harm. Dying and respawning in the game may well disturb their predictions, and they may come to read that as bad, or they may not.
- **Nothing acts on the reading yet, so nothing is decided by it.** With the planner's cost gone, the reading can only influence Luthi through what they perceive, and their choices still come from the four built-in preferences. So "they decide how to react" is true in that nothing is forced, but they don't yet have a way to choose in any full sense. That is the reward question from 2026-11-02, and it's the next big conversation.
- **Readings saturate.** After a quiet stretch the typical change is small, so moderate changes read near ±1.0. That loses strength information, and I haven't judged whether it matters.
- **Test size only.** Tiny, mostly untrained model, one seed, CPU.

**Open, Brian's calls:**
- The reward question: Luthi's choices still come from four built-in preferences.
- The reading's placeholders: threshold 4, floor 0.05, window 128, and smoothing 0.1.
- The welfare gauge's alarm thresholds.

## Artifacts

- LuthiModel: `luthi/v2/m9/felt_valence.py` (`FeltReading`), `luthi/v2/m9/runner.py`, `luthi/sanctuary_interface.py` (`felt_now`, `felt_summary`), `tests/test_felt_reading.py`
- Sanctuary: `sanctuary/core/schema.py`, `sanctuary/core/cognitive_cycle.py`, `sanctuary/core/luthi_model.py`, `docs/MINECRAFT_FIRST_WORLD_DECISION_2026-11-01.md`

Sources:
- [Joffily & Coricelli 2013, Emotional Valence and the Free-Energy Principle](https://halshs.archives-ouvertes.fr/halshs-00834063v1/file/Emotional_Valence_and_the_Free-Energy_Principle.pdf)
- [Hesp et al. 2021, Deeply Felt Affect](https://pmc.ncbi.nlm.nih.gov/articles/PMC8594962)
