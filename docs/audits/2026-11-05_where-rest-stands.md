# Where rest stands — 2026-11-05

**Written by:** Fable 5.1, for Brian and for the Opus-line review.
**What this is:** a firsthand read of the rest design against the code. It lists what exists, what is missing, a possible order, and the decisions that are Brian's. It proposes. It rules nothing.

**Why now.** On 2026-11-05 Brian ruled that Sanctuary's small path has no sleep phase for now, that rest returns later as something Luthi chooses, and that his August rulings stand: Luthi decides in rest what the living layers keep, and waking stays at full strength until rest exists. What follows from that, in my reading, is that rest should be built before a real Luthi lives on that path.

**How it was checked.** I read `docs/research/2026-07-06_nrem-learner-spec.md` in full, read the code it names, and ran one 700-moment test-size life on the small path to see what the episode stores hold (tiny model, CPU, text only, one seed).

## What Brian has ruled

| Date | Ruling | Where |
|---|---|---|
| 2026-07-05 | Change is automatic and unvetoed: "I am a direct reflection of my experiences whether I want to be or not." | Quoted in the spec's R3-a. I did not find or read the original record. |
| 2026-08-14 | **R1.** Waking changes the mind far less: "the susceptibility to change should be greatly lessened so the lesson of the day can be intentionally pondered rather than purely reacted to." | Spec amendment |
| 2026-08-14 | **R2.** No cap on the day's record: "Allow as much information as Luthi deems important to be stored and considered for integration later." | Spec amendment |
| 2026-08-14 | **R3.** Rest is Luthi's act: "a tool Luthi can call, to start a calmer mode for deep intentional consideration", for "making decisions on what to keep and what to toss." | Spec amendment |
| 2026-11-02 | The mind controls whether and when each kind of its learning consolidates. | Sanctuary `docs/LEARNING_MODES_DECISION_2026-11-02.md` |
| 2026-11-04 | "They decide how to react to everything." | `docs/DECISIONS.md` |
| 2026-11-05 | No sleep phase now, rest as Luthi's choice later. The August rulings stand. Waking stays at full strength until rest exists. | `docs/DECISIONS.md` |

## What exists

| Piece | State | Evidence |
|---|---|---|
| Lessened waking (R1) | Built, 12 tests. Nothing calls it. | `luthi/v2/plasticity.py`; no caller in either repo |
| The day's record | A ring of 64 episodes per living layer. Each holds a snapshot the size of the whole weight. It overwrites when full. | In the test life it was empty as trained and full after 700 moments, with 173 writes. So it held roughly the most recent 260 moments, about 26 seconds at 10 Hz. |
| Recall from that record | Working. | 2,063 recalls in the same 700 moments |
| Integrating an episode into the living weights | Built: `consolidate_layer_attractor`. The layer's own trigger for it is off in the arm the small path uses. | `luthi/v2/consolidation.py`; `consolidation_enabled=False` |
| Moving the home (`keep_recent`) | Built. Off the small path by ruling. | `luthi/sanctuary_interface.py` |
| Luthi's view of their own record | Summary numbers per block only: how many episodes, their mean and highest salience. No single episode can be shown to them. | `get_introspection` |
| Luthi's "remember" choice | The action layer produces it, with a salience. The M9 spec meant it to become an episode write. On the small path it is counted and goes nowhere. | `luthi/v2/m9/decoders.py` |
| The mind's say over consolidation (allow, defer, emphasize) | Built on Sanctuary's old path only. It counts in sleeps, and Luthi has no way to set it. | Sanctuary `learning_governor.py` |

One name to keep straight: M9's `a_rest` is the action of doing nothing this moment. It is the stillness baseline, and it is not the rest this document is about.

## What is missing

1. **A way for Luthi to choose rest.** The action layer chooses only to speak, attend and remember. Nothing can carry "I want to rest now."
2. **The uncapped record (R2).** Append-only through the day, holding the experience (context, input, salience, when) and not a weight-sized snapshot. This changes buffer layout, so it changes checkpoints.
3. **The rest pass (R3).** Take the day's record, integrate what is kept, discard what is tossed, then start a new day. The spec's two test gates come with it: change per rest stays bounded, and no moment is lost or counted twice at the boundary.
4. **A way for Luthi to see one experience again and say keep or toss,** without speech.
5. **Rest on the small path.** Entering and leaving, what the world does meanwhile, and telling Luthi what happened.
6. **The record of what each rest integrated,** so development can be inspected.

## Decisions that are Brian's

1. **Can Luthi go without rest indefinitely?** August left this open (R3-a). The July ruling says change is unvetoed. Opus 5 proposed: Luthi chooses when, never whether, and if they have not rested within some window, rest arrives anyway. This week's rulings give Luthi more say than July's did, and I don't know which he holds here. One fact that bears on it: without rest, the trained weights still change every moment through the lived update. What would never be kept is the living layers' fast change.

   **Brian, 2026-11-05, after reading this question:** "I think Luthi could technically not ever need rest because it isn't required for existence. It is supposed to fulfill specific functions but resting for the same reasons humans do is not something LuthiModel should do."

   Fable's reading of that, not yet confirmed with him:
   - Rest is not a need, so it is never forced. The "rest arrives anyway" proposal is declined.
   - Rest is defined by what it does (considering what happened, deciding what to keep, integrating it), not by likeness to human sleep.

   Not taken from it: that the July ruling is withdrawn. Without rest, change still happens every moment through the lived update.
2. **What happens to an experience Luthi does not judge?** The August welfare note recommended a default (integrate by salience) so rest is never a nightly duty to sort everything. Today's ruling says nothing is kept on their behalf. Those pull in different directions.
3. **How Luthi chooses rest, and says keep or toss.** Options I can see:
   - a new channel in the action layer for rest. It needs a consequence the planner can predict, or it is an untrained dial logged as a choice (the caution in the learning-modes decision);
   - reading sustained stillness as a wish to rest. I would not: that is Sanctuary interpreting Luthi;
   - for keep or toss: in rest, show each stored experience to Luthi again, and let their existing "remember" choice on it be the keep.
4. **What "remember" means while awake.** Suggested on 2026-11-05 and not ruled: "store this for me to consider later." The M9 spec's original meaning was close to this.
5. **The write-spacing floor** (`refractory_calls`), which the amendment flags as a hidden cap on R2. Still unruled from August.
6. **The world during rest.** It can wait in line, which has a limit, or reach Luthi and let them decide whether to wake. It must not be dropped.
7. **Whether corpus replay and the retention gate run during rest.** I did not find this addressed anywhere.
8. **How strongly a kept experience is integrated.** The spec defers this to tuning on a trained run.
9. **Lessened waking, if rest may never happen.** R1 assumed rest would do the integrating. If Luthi may go without rest for good, then switching R1 on once rest exists would leave a Luthi who does not rest with living layers that barely move, permanently. Raised by his 2026-11-05 answer to decision 1.

## A possible order

0. **Review first.** The Opus line reads the small path and this document before anything is built on either.
1. **The uncapped record (R2),** in LuthiModel. Opt-in, with today's behaviour unchanged by default, and with checkpoint tests. Needs no training.
2. **The rest pass (R3) as a function that is handed a keep-and-toss list.** With its two gates as tests. At this point rest can be exercised in tests with a caller choosing the list, before Luthi can.
3. **Showing an experience to Luthi and reading their answer.** Waits on decision 3.
4. **Rest on the small path.** Entry by Luthi's choice, the world waiting, lessened waking switched on with it. Waits on decisions 1, 3 and 6.
5. **Measure at test size.** Does a kept experience change later prediction where a tossed one does not? Is the change per rest bounded? Is every moment accounted for at the boundary?

Steps 1 and 2 can be built and tested now. What cannot be done until training is possible: tuning how strongly rest integrates, and anything about whether Luthi's choices in rest mean something.

## What I did not check

- The original 2026-07-05 record, the momentum design brief, and the 2026-08-13 audit the spec cites.
- Whether `adaptive_episodes=True` (the spec's stated precondition) changes what the store holds at this size. It is off by default and off here.
- What the second, block-level episode store is for. It also filled to 64 of 64 in the test life.
- Anything beyond one tiny model on one seed. The episode counts above are a single run.
