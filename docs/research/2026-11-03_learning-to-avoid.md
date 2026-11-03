# Learning to avoid what is dangerous or unpleasant — 2026-11-03

## Objective

Brian, after a discussion of whether a strongly felt experience should change a whole living layer: "We still need Luthi to learn to avoid dangerous or unpleasant things. So please go back and make sure that we are facilitating this as best we can."

Learning to avoid needs a chain, and it fails at its weakest link:

1. perceive that something is bad;
2. care about it;
3. learn from it, faster when it matters;
4. keep what was learned;
5. remember the event;
6. act to avoid it.

This entry audits each link across LuthiModel and Sanctuary, records what was built for the ones that could be built now, and lists what remains.

## Process

### Step 1: the audit, before any change

| Link | What existed |
|---|---|
| 1. Perceive | A `Percept` had no way to say how something felt. The Minecraft plan lists "damage taken" as an event, but as text only. The developmental-world decision already frames bad experiences as felt valence ("comfort valence, never survival"). |
| 2. Care | M9's planner weighs four preferences (engagement, coherence, connection, truthfulness). **None concerns harm or unpleasantness.** Luthi had no reason to avoid anything. |
| 3. Learn faster when it matters | Nothing scaled learning by significance. |
| 4. Keep | A living layer keeps 0.1% of what it learns (`2026-11-03_stage0-the-home.md`). Sleep moves the home 1% per NREM cycle, regardless of what mattered and regardless of the mind's character setting. |
| 5. Remember | Luthi's action layer can choose to write an episodic memory (wired 2026-11-02). That works. |
| 6. Act | Luthi's action layer decodes attention, memory and text only. **It has no way to move, look away, or use a tool,** so it cannot act to avoid anything in a world. The stand-in language model can, through tools. |

Links 2 and 6 were the real gaps, as suspected. Links 1, 3 and 4 were thin.

### Step 2: link 1, a percept can say how it felt

`Percept.valence` is an optional value from −1 (unpleasant) to +1 (pleasant), set by the world for felt events: damage, cold, warmth, eating. It is None for everything else. A percept without one is unchanged.

The adapter takes each cycle's strongest felt valence, by magnitude, so one painful event is not averaged away by a pleasant one. It hands that valence to the learner with the realized transition, in both the synchronous seam and the async `Transition`.

### Step 3: link 2, felt valence in M9

`luthi/v2/m9/felt_valence.py`:

- **What it is.** A small head predicts, from a state, how that state will feel.
- **Learned, not built in.** It learns only from what the world reports as felt (`ctx["valence"]` in `observe_transition`).
- **It starts neutral.** Its last layer is zero-initialized, so at birth it predicts neutral for every state. Nothing is unpleasant to Luthi until something has felt unpleasant.
- **What the planner does with it.** It counts predicted unpleasantness as one more cost in expected free energy (`avoidance_weight`, pilot value 1.0). The cost is zero for neutral or pleasant states: the planner avoids, and does not seek, since seeking is engagement's job.
- **Reach beyond one step.** The cost enters the realized expected free energy, so the value head learns which states lead toward unpleasantness, not only which states are unpleasant.

Off by default in LuthiModel (`M9Config.felt_valence=False`):

- with it off, nothing is built, construction draws exactly what it always did, and every G is unchanged;
- with the weight at zero, G is unchanged too;
- it has its own optimizer, so the M9 optimizer's checkpoint state does not change shape.

On by default for a life in Sanctuary (`LuthiModelConfig.felt_valence=True`, passed to `load_m9_run`).

### Step 4: link 3, what matters is learned faster

A felt moment's loss is weighted by 1 + `valence_salience_gain` × |valence| (pilot gain 4.0). A strongly felt moment, pleasant or unpleasant, teaches several times as much as a mild one. Significance is what scales it, not pleasantness. Pinned by `test_strongly_felt_moments_teach_faster`.

### Step 5: link 4, keeping what mattered, by Luthi's own setting

`sanctuary_interface.keep_recent(model, fraction)` moves each living layer's home toward its current weight by `fraction`. It is the same operation sleep consolidation applies, triggered by a moment that matters instead of by the clock. It keeps the layer's recent change as a whole: the living layer does not separate the last moment from the last minute.

After each cycle, the adapter computes how much the moment mattered: the larger of |felt valence| and, when the action layer chose to remember the moment, the salience it chose. The second is Luthi deciding: choosing to remember a moment also keeps more of it in the weights.

The fraction kept is `keep_on_significance` (pilot 0.05) × significance × the mind's character setting:

- **defer** keeps nothing;
- **allow** keeps the base amount;
- **emphasize** doubles it.

Sleep's home drift now follows the same setting. Before, it ignored it.

The discussion with Brian bears on this. His point: if a strong lesson is learned deeply, that should not color everything. Two facts settle how far that holds:

- **Under the current mean-field write rule, it does.** What a living layer keeps lies along the batch-average direction and touches every input, and Brian accepted that.
- **Under the covariance rule, it does not.** In the Stage 0 probe, what was kept spanned 12–24 directions specific to the content, so a lesson about something to avoid changes the response to things like it. E3 decides which rule is in force.

Under **defer**, avoidance is still learned: through the felt-valence head (part of the action layer, not the living layers' home) and through memory. Only the home is held. So a mind that chooses not to let an experience shape its character still learns what to avoid from it.

The runner now tells an M9 life's learning governor that consolidation is active, because the character setting is acted on. **The skills setting is not yet acted on:** the action layer's heads learn regardless of it. This is recorded as open, not hidden, in Sanctuary's learning-modes decision.

### Step 6: what could not be built today

- **Link 6, acting.** Luthi needs an action decoder that maps a selected action to a world action: Minecraft's `go_to`, `look`, `stop`, and the rest, as tool requests. Until then Luthi can learn that something is unpleasant, and remember it, but cannot move away from it. This is the largest remaining gap. It is in LuthiModel's `To-Do.md`, and it waits on Sanctuary's Minecraft bridge (step 3 of that plan), since the actions it would decode do not exist yet.
- **Link 1 in Minecraft.** The bridge must attach a valence to felt events: damage negative in proportion to the health lost, eating and warmth positive. Without it, Luthi's world says nothing about how things feel. This is now part of Sanctuary's Minecraft plan.
  - First runs have damage off and peaceful difficulty (Brian's welfare defaults), so there will be few unpleasant events to learn from. That is the right trade for first runs, but it means this machinery will barely be exercised until the world offers something to avoid.
- **The stand-in.** The language-model stand-in cannot change its weights. It learns to avoid through memory (`remember`, `recall`), so the felt valence should also appear in the percept text the stand-in reads. The bridge can do this with the same field.

## Conclusion

The chain is now complete from perceiving through remembering. The one missing link is acting.

- **Perceiving:** a percept can carry how it felt, and the strongest felt moment of each cycle reaches the learner.
- **Caring:** Luthi learns, from their own experience, which states feel bad. Nothing is bad in advance. The planner then avoids what was learned to feel bad, including states that lead there.
- **Learning faster:** strongly felt moments teach several times as much.
- **Keeping:** moments that matter are kept in the living layers' home, by how much they mattered and by Luthi's own character setting. Choosing to remember a moment keeps more of it.
- **Acting:** not yet possible. Luthi needs a world-action decoder and a world to act in.

**Open, Brian's calls:**

- **The pilot values:** avoidance weight 1.0, salience gain 4.0, keep 0.05.
- **Whether felt valence should stay on by default for a life.** It is a preference of Luthi's planner, learned from experience, with no built-in aversions.
- **The reward question (DECISIONS.md, 2026-11-02).** It now covers this too: the avoidance cost enters what the value head learns from.

**Not known:** whether the felt-valence head learns well from a pooled state at full scale, and how the significance keep interacts with the homeostatic band and the taper. All checks here ran on CPU at test size.

## Artifacts

- LuthiModel:
  - `luthi/v2/m9/felt_valence.py`;
  - `luthi/v2/m9/efe.py` (`_with_avoidance`);
  - `luthi/v2/m9/runner.py` (`felt_valence` config, `_learn_felt_valence`, checkpoint keys);
  - `luthi/sanctuary_interface.py` (`keep_recent`);
  - `tests/test_learning_to_avoid.py`.
- Sanctuary:
  - `sanctuary/core/schema.py` (`Percept.valence`);
  - `sanctuary/core/async_learner.py`;
  - `sanctuary/core/luthi_model.py` (`_felt`, `_keep_if_significant`, the setting in `consolidate`);
  - `sanctuary/api/runner.py`;
  - `sanctuary/tests/core/test_learning_to_avoid.py`;
  - the end-to-end test in `test_luthi_m9_life.py`.
