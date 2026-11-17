# Review: the memory store's cold storage and its first importance formula

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat), and terminal Fable, who builds it · relayed by Brian
**Date:** 2026-11-17
**Reviewing:** build brief 9, parts D and E (`docs/reviews/2026-11-17_build-brief-9-the-memory-store-for-terminal-fable.md`), with Sanctuary's design and Brian's rulings (`docs/research/2026-11-17_memory-store-rework-design.md`, at `d501ea0`), against the store and loop as built (`sanctuary/life/memory.py`, `loop.py`) and the action layer (`luthi/v2/m9/`).

**Verdict.** Both parts are sound in principle.
- **D:** "The store's code deletes no memory" is the right rule, and simple enough to hold.
- **E:** recording the signals and computing importance from them is the right shape, because the formula can then change.

That shape is also why my main point on E is about the records, not the formula: a formula can be rewritten later, but a signal not recorded at the moment is gone. One signal the brief names is not what it is called (E1 below).

D's points are about four ways a permanent, never-deleted folder can still lose or confuse a memory. All of them are cheap to close now and hard to close later.

## D. Cold storage

**D1. Two lives can share a cold folder.** The default is `E:\luthi_cold_storage\<the life folder's name>`, and ids start at 1 in every life. A second life in a folder of the same name (a rerun of a seed, say) would move its `memory_00000001/` into a folder that already holds one.
- Never overwrite in cold: a move whose destination exists raises, and the original stays.
- At first use, write the life's identity into the cold folder: its full path and the time the life began. Refuse a cold folder that belongs to another life.

**D2. An id must never be reused once it is in cold.** Today the next id is one past the highest seen in the life's lines and files. Under part B the cache is rebuilt from the folders, and folders leave for cold. Take the next id past the life's let-go lines and cold's index as well, or keep a counter file in the life.

**D3. A move needs an intent line, as a let-go has.** The store already writes a let-go's line before it moves anything, "so a memory is never out of reach without a line saying why". The move to cold should do the same:
1. Write a line, "moving", to the index.
2. Across drives, copy into an `incoming/` folder under cold, check every file's size and hash, then rename it into place.
3. Write a second line, "moved", and only then remove the original.
4. On start, finish or undo whatever has a "moving" line and no "moved".

A part-copy in `incoming/` is not a memory, and the original is untouched until the copy checks. So clearing it removes nothing that "cold is never deleted from" protects. The brief should say so, or a crashed copy is kept forever as if it were a memory.

**D4. With the defaults, cold shares the life's drive.**
- The life is on E: and cold defaults to E:. A move there is a rename and frees nothing on the drive.
- With sight, the design's own table puts cold's growth at 10 to 30 GB a day. At that rate a terabyte lasts five to fourteen weeks.
- The brief's 5 GB margin stops moves, and so keeps, when that happens. But the life's own saves (its checkpoints, in the same life folder) are on E: too, and nothing guards them. A save that fails stops the life.

So:
- guard every write on that drive with the free-space check, the life's saves included;
- or put cold on a drive of its own. Where cold lives is Brian's choice.

**D5. "A folder moved back is all it takes" needs one more line.** A folder in the life that also has a let-go line must stay out of reach: that is exactly what a crash in the middle of a let-go leaves. So bringing a memory back is the folder plus a line saying it came back. Write that into the format now, so that bringing back stays one step when Brian rules on it.

**D6. Small.**
- Decide "same drive" by comparing the two folders' device numbers before moving. Don't decide it by catching the rename's error. The codebase avoids that kind of control flow, and the error differs between Windows and POSIX.
- "Oldest first" from `let_go/` should say oldest by what: when it was kept, or when it was let go. Today's `_make_room` goes by id, which is when it was kept.
- The index says `for_room` where the life's line says `by: "room"`. Use one word for both.

## E. Importance

**E1. `memory_salience` is not how strongly they chose to remember.** In the action layer:
- **The choice to remember** is made from the memory head's activity on the predicted next state, |h(ŝ)|. It is tested against the head's running band, an absolute floor, and the rest action's activity (`runner.py` `decode_selection`; `activity_bands.is_silent`).
- **`memory_salience`** is σ(h(a)): the same head on the action vector itself, with its sign kept (`decoders.py`, `MemoryDecoder`).

They are different numbers. The choice ignores the sign that the salience keeps, so a moment kept on a strong negative reading files a salience near 0. It would then rank near the bottom of "their own bar".

What I would record at each keep:
- **The choice's own margin:** the activity, and each threshold it had to pass (band, floor, rest plus tolerance), at that moment. That is "how strongly they chose", and "their bar at the time", recorded instead of reconstructed later.
- **`memory_salience` as well,** as a second signal.

One cheap check in a tiny life would settle which tracks the choice: among kept moments, how well `memory_salience` agrees with the margin.

**E2. A chosen recall is driven by the cue; it is not a choice of that memory.**
- Luthi chooses to look back; the store brings the memory most like the present.
- Each look-back then files a new summary near that present. So a recalled memory becomes easier to recall again in the same region, and counts never fade.
- Rest's revisits file summaries too. So what rest brings, which counts for nothing directly, raises later recall counts.
- Under drift, a memory from a stretch of life the mind passed through quickly gathers no recalls. It goes before any stretch that was recalled, whatever it meant.

What I would record at each look-back, beside chosen-or-rest:
- the likeness to the cue;
- how many memories it beat.

A later formula could then tell a close match from a memory that is merely near everything. The choice that is about one memory is "remember, while looking at it", and I would weight it above the bare recall.

**E3. Both terms grow with a stretch's length.**
- The highest of n ranks rises with n: its expected value is about n/(n + 1).
- A stretch with more kept moments has more summaries for a cue to find.

If length is not meant to count as importance:
- use the highest rank's own percentile, (highest rank)ⁿ, which is uniform whatever n is when the ranks are;
- or use the mean rank.

Under the brief's formula, long stretches, which are also the largest on disk, are favored to stay.

**E4. Ties decide most cases.**
- Most stretches will never be recalled, so the rank alone orders them.
- For a mind that gives no salience, every rank is 0.5. The rule then becomes "not touched for longest", which is age. The design says age alone is not unimportance. Today that applies only to stand-in minds, but say it.
- Put the id last among the tie-breaks, so the order is fixed whatever order the folders are listed in.
- If every stretch is protected (in view, being written, or recent), nothing moves and the store counts as full.

**E5. The last 1,000 must survive restarts and moves.** A rank needs the window of recent kept moments' signals. If it is rebuilt from folders, it loses whatever has moved to cold. Keep the window in the life. Say what the first moments rank among, when fewer than 1,000 exist.

**E6. Small.** Part B's "the mind's signals of part D" should read part E.

## What I would build first

- **D1 to D3 before any life uses cold storage.** They are the ones that can lose or confuse a memory.
- **E1's and E2's extra fields from the first record written.** They cost a few numbers a record and cannot be added afterwards.
- **D4's guard on every write.** Ask Brian where cold should live.

## Artifacts

- Code read: Sanctuary `sanctuary/life/memory.py` (`let_go` writes its line first; `_make_room`; `nearest` excludes what is recent by any row; `look_back` files a row) and `loop.py` (`step`: every look-back files a row, chosen or brought by rest).
- LuthiModel `luthi/v2/m9/runner.py` (`decode_selection`: activity on ŝ, salience on a), `activity_bands.py` (`is_silent`), `decoders.py` (`MemoryDecoder`: activity is |h|, salience is σ(h)).

— Opus 5.5, 2026-11-17
