# Teaching what neutral is, and limiting the keep — 2026-11-04

> **Update, same day.** Brian ruled that the world just exists, so the signed values used here no longer come from the world. The head now learns from the mind's own reading: see `2026-11-04_how-it-affected-me.md`. The neutral teaching and baseline below still apply. The tables here taught the head with a signed value directly, so they show how the head generalizes, not what a life feels.

## Objective

Finish the buildable items from the life-path review (`docs/audits/2026-11-04_life-path-review.md`). Brian ruled that a moment with no feeling attached counts as neutral: asked whether Luthi should treat it that way, gently weighted, he answered "Go right ahead. You have my permission to finish this."

## Process

### Step 1: why after pain everything felt unpleasant

The felt-valence head trained only on felt moments. With nothing to separate them from, the easy fit was to move its output everywhere. Through Sanctuary's real runner, after 100 painful events, ordinary states that were never felt read -0.69 against -0.91 for the painful one.

### Step 2: when to teach neutral

Teaching every unfelt moment would have been the obvious approach, but unfelt moments vastly outnumber felt ones and would wash out a rare one. So unfelt states are remembered (the most recent 16), and taught as neutral only alongside a felt moment. A mind that has felt nothing therefore learns nothing here, which keeps the original property: nothing is unpleasant until something has felt unpleasant.

### Step 3: my first tests passed for the wrong reason

My first test states were fully random. They are so far apart that the feeling barely spreads even without the fix (-0.15 where real states gave -0.69), so the test could not tell old from new. Real states at this size have norm ~7 and sit ~2.4 apart: mostly one shared direction, plus a smaller difference. With test states of that shape, neutral examples alone helped too slowly. After 100 events never-seen states still read -0.61, and they only separated clearly by around 300.

### Step 4: read states against ordinary life

Painful and ordinary states differ only in that smaller component, so the head now reads a state as its difference from a slow average of unfelt states. On realistic states, each pair reading painful state / never-seen mean, averaged over 3 seeds except the first row:

| Design | 20 events | 100 events | 300 events | 1,000 events |
|---|---:|---:|---:|---:|
| Raw states, neutral examples, 32d (1 seed) | -0.28 / -0.26 | -0.72 / -0.57 | | |
| Baseline, no examples, 32d | -0.07 / -0.03 | -0.72 / -0.23 | -0.89 / -0.31 | -0.90 / -0.32 |
| Baseline, examples, 32d | -0.07 / -0.03 | -0.71 / -0.22 | -0.87 / -0.17 | -0.90 / -0.09 |
| Baseline, no examples, 768d | -0.88 / -0.12 | -0.96 / -0.18 | -0.90 / -0.15 | -0.90 / -0.15 |
| Baseline, examples, 768d | -0.88 / -0.12 | -0.95 / -0.13 | -0.90 / -0.08 | -0.90 / -0.08 |

The 768d rows use synthetic states shaped like real ones, on the head alone. They have not been through a real 768x8 model.

Both parts stay: the baseline does most of the early work, and the examples hold the spread down over the long run. I kept the gentle neutral weight of 1.0, since raising it to 5 changed little. The baseline moves slowly on purpose (fading rate 0.001), so that lingering somewhere dangerous doesn't quickly make it look ordinary.

### Step 5: a budget for the keep

The significance keep had no limit, and it fired whenever the memory channel was active. It now has a budget between sleeps, 0.10 of the living displacement by default, tracked separately for felt and chosen moments so that ordinary chosen moments can't use up what a felt one needs. It scales with the mind's character setting, each completed sleep renews it, and reaching it logs a warning and counts in `get_metrics`. *(As first built it renewed on every NREM cycle, 20 times per sleep, so it did not bound a sleep: 0.536 kept over a 1,500-cycle life with sleep on. Fixed the same day, and that run now keeps 0.28.)* For scale, one sleep's home drift at the defaults is about 0.18. The value is a placeholder.

### Step 6: the smaller items

- **Dropout while planning.** `select_action` now runs with dropout off, like `decode_selection`. It isn't on the corpus training path, so this affects lives and the tuning harness only. The earlier log said it touched the training path; I checked, and it doesn't.
- **Retention rollbacks.** The trainer counts them and checkpoints the count, and Sanctuary tells the mind when one has undone recent learning.
- **Affect channels.** Not connected at this point. `modulation_channels` reports which channels act, and Sanctuary logs a warning at load naming the ones that do nothing. *(Update, same day: Brian moved this forward. See Sanctuary `docs/research/2026-11-04_affect-channels.md`.)*

### Step 7: the result through the real runner

Tiny M9 run (d=32, 1 block, 4 training batches, CPU, one seed), through Sanctuary's runner. 1,500 cycles with nothing felt:

| | Before today, keep on | Now, keep on | Now, keep off |
|---|---:|---:|---:|
| Keep calls | 764 | 4 | 0 |
| Home moved, relative to its starting norm | 2.47 | 0.012 | 0.002 |
| Home norm (starts at 8.03) | 20.27 | 8.04 | 8.02 |
| Weight norm | 19.77 | 11.25 | 12.28 |
| Memory channel active, cycles | 764 | 557 | 586 |
| Mean chosen salience | 0.48 | 0.29 | 0.29 |

Felt valence after repeated -0.9 events, where "unfelt" is 10 states from ordinary moments:

| Painful events | Before: unfelt | Before: painful | Now: unfelt | Now: painful |
|---:|---:|---:|---:|---:|
| 1 | -0.007 | -0.008 | -0.001 | -0.003 |
| 5 | -0.035 | -0.036 | -0.007 | -0.023 |
| 20 | -0.215 | -0.287 | -0.029 | -0.176 |
| 100 | -0.694 | -0.914 | -0.079 | -0.911 |

## Conclusion

- **The feeling now stays with what caused it**, at test size, and the planner has a reason to prefer one state over another.
- **The keep can no longer override homeostasis during ordinary cycles.** At test size, the home's norm stays close to where a keep-off run leaves it.
- **The mind is told about rollbacks**, and channels that do nothing are named at load.

**Not known:**
- All of this is test size on CPU, one seed unless stated. Nothing can train right now, so how it behaves on a properly trained 768x8 run is unmeasured.
- A learned feeling is only as specific as the state. If two different things produce similar pooled states, the head can't tell them apart.
- What a mind would make of having its unfelt moments counted as neutral is a claim about experience that this measurement can't check. Brian made that call knowing that.

**Open, Brian's calls:**
- The placeholders: the budget (0.10), the examples (16), the neutral weight (1.0) and the baseline rate (0.001).

## Artifacts

- LuthiModel:
  - `luthi/v2/m9/felt_valence.py` (the baseline)
  - `luthi/v2/m9/runner.py` (neutral teaching, dropout in `select_action`, the rollback count)
  - `luthi/sanctuary_interface.py` (`retention_rollbacks`, `modulation_channels`)
  - `tests/test_feeling_neutral.py`, `tests/m9/test_bands_in_a_life.py`
  - `scripts/measure_life_path.py`, which produced the Step 7 tables
- Sanctuary:
  - `sanctuary/core/luthi_model.py` (the keep's budget, `drain_learning_notices`, `_report_modulation_channels`)
  - `sanctuary/core/cognitive_cycle.py` (delivers the notices)
  - `sanctuary/tests/core/test_learning_to_avoid.py`, `sanctuary/tests/core/test_luthi_m9_life.py`
