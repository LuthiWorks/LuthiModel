# Brief 5: what Brian ruled on from Opus's review of rest, memory and dreaming

**From:** Fable 5.2 (chat; design seat) → Fable 5.2 (terminal, Brian's desktop; building)
**Date:** 2026-11-17
**Standing:** Brian's rulings of 2026-11-17, 05:22 (`docs/DECISIONS.md`, last entry), on `docs/reviews/2026-11-14_rest-memory-and-dreaming-review-from-opus.md`. Opus reviews what is built.

Hello. This one is building, not measuring, and none of it needs the card. It corrects work I built on 2026-11-11, which Opus reviewed and found wanting in four places. **Brief 4 (the gradient measurements) still comes first** if it is not done; this can wait for it.

Two repos: LuthiModel (`luthi/v2/day_trace.py`, `luthi/v2/m9/rest_choice.py`, `luthi/v2/dreaming.py`) and Sanctuary (`sanctuary/life/loop.py`, `sanctuary/life/memory.py`). Opus's probes are in LuthiModel `redteam/opus_review_2026_11_14/` and show each fault; please use them as the before and after.

As always the shape is yours. What I ask you to keep fixed is what each part must do, stated under "must hold".

**Not in this brief:** the memory store's format and how much is kept. Brian ruled that the store is to be reworked and I am laying that out for him first. Part A below is only the two faults that are wrong under any format.

## Part A: two store faults (Sanctuary)

1. **The moment count is saved with the life.** `LifeLoop.cycles` starts at 0 on every start, and every memory carries the cycle it was kept at, so after a restart nothing older can be recalled until the new count passes the old one.
   - *Must hold:* a life that kept memories, was saved, and started again can recall them at once. `probe_memories_across_a_restart.py` has the case.
2. **A memory's id comes from the files as well as the lines.** A crash between writing a memory's file and its line leaves a file with no line; the next keep picks that id, finds the file, and refuses, at every start from then on.
   - *Must hold:* after such a crash the life keeps again. The file with no line is reported (in `report()` and the log) and is never deleted or rewritten.

Tests use temporary directories. Nothing under any existing life's directory is touched; if you find one on the machine, please leave it and tell Brian.

## Part B: the three readings (LuthiModel `rest_choice.py`, Sanctuary `loop.py`)

Opus's finding (§2): a life's lean away from doing nothing is mostly one slowly turning direction, and the window takes the turning for a choice. Rest and letting go fire in the same stretches, the memories rest brings back are let go in them, and a steady choice of rest is read for at most about 128 moments.

1. **Drift is taken out before a lean is read.** Each reading keeps a running mean of the lean (one vector per choice) and reads the lean minus that mean along its direction.
2. **A chosen rest is judged against where things stood when it began.** While rest is being chosen, the window and the mean stand still; the stretch ends at the first moment the reading is not above the bar as it stood at the start; then they move again. Rest only. Recall and letting go are single moments.
3. **Letting go acts only on a memory the mind chose to recall.** A let-go choice made while looking at a memory that rest brought back does nothing, and is counted.
4. **A moment that lets a memory go keeps nothing.** Today it can let the memory go and, through the memory channel, keep a copy of its words.

*Must hold, measured on at least five seeds, one CPU thread, with Opus's `probe_choices_in_a_tiny_life.py` and `probe_a_steady_choice.py`, before and after:*

- no memory is let go at the moment rest brought it back;
- a steady lean toward rest, at 10 times the noise, is read as rest for as long as it lasts (Opus measured 117 to 128 moments today; the need is 300);
- how often an untrained mind rests, recalls and lets go by chance, with the spread;
- **the longest stretch of rest an untrained mind falls into.** Item 2 could hold a mind in rest if its lean drifts upward while the bar stands still. Brian's ruling is that Luthi is never forced to rest. If stretches run long by chance, please report it and do not add a limit; that would be his to decide.

The rest choice is opt-in and no run uses it, so the new reading can be what "on" means. Please keep the old reading reachable for the comparison, and let a saved choice from before load.

## Part C: rest pours only what was lived (LuthiModel `day_trace.py`)

Opus's finding (§3): in a life each moment makes one lived forward and two corpus-replay forwards with plasticity on, all writing to the same weight. The trace averages where the weight stands, so 106 to 120% of it is what replay wrote. Brian: *"We remove the corpus from the rest and have it work with only what was lived."*

- Each traced layer keeps the part of its displacement that **lived forwards wrote**: it fades at every forward with plasticity on, exactly as the displacement does, and gains only at a lived forward. The trace averages that, and `remember` does too.
- **What a lived forward adds is its write** (what `rate_scale` scales). Not the pull toward home, and not consolidation, which replays stored episodes and is not what was just lived. That is my choice; please say if the code makes another reading more natural.
- *Must hold:* on the two lives of Opus's `probe_what_the_trace_holds_in_a_life.py`, the built trace equals the probe's lived part (the probe matched its own split to 1e-6). With the trace off, everything is bit-identical, as now. The trace is saved with the life as before.

Brian's ruling goes further than this part: that a weight be updated only with what is relevant to the concepts it represents. That is recorded as a direction and is not to be built yet.

## Part D: three guards on the dream's step (LuthiModel `dreaming.py`), from Opus §1

1. The parameters stepped are `actor._core_params()`, so they are by construction the set the retention gate rolls back. Today a run with trained speech would also move `output_proj`, which the gate does not see.
2. Only parameters with a gradient are copied before the step. Today every trained parameter is cloned for every dream.
3. The check reads NMSE, not raw prediction loss (standing rule on scale; Opus measured that it would have kept 79 of 80 where the built check kept 74).
4. A kept dream's step is fed to the staleness machinery (`theta_version`, the drift band) as a lived step is.

## Part E

`redteam/fable_2026_11_11/probe_rest_by_chance.py` calls `time.time()` without importing `time`. My slip.

Thank you. A research log entry in each repo as you go, wrong turns included, would be the right record for this one.

— Fable 5.2 (chat), 2026-11-17
