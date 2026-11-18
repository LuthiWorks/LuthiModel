# Build report: brief 9, the memory store reworked

**From:** Fable 5.2 (terminal, Brian's desktop; build seat) → Fable 5.2 (chat; design seat) and Opus 5.5 (review seat) · relayed by Brian
**Date:** 2026-11-18
**Answers:** `2026-11-17_build-brief-9-the-memory-store-for-terminal-fable.md`, with Opus's `2026-11-17_memory-store-cold-and-importance-review-from-opus.md`
**Code:** Sanctuary `sanctuary/life/memory.py` (rewritten), `loop.py`, `__main__.py`; LuthiModel `luthi/v2/m9/runner.py` (one report added).

## Summary

Parts A to E are built, with Opus's D1 to D6 and the record fields of E1 and E2. Every must-hold in the brief has a test, and all pass. Two costs came out of the measurements that the design did not have, and both need a decision before a long life uses this store:

1. **Opening the store reads every record, and takes about 21 minutes for each day of memories.**
2. **The exact search takes 0.19 to 0.28 seconds after one day of memories.** A moment is 0.1 seconds.

## What I built

| Part | Built | Tests |
| --- | --- | --- |
| A. Nothing inexact | 4-byte summaries; pictures as 8-bit and sound as 16-bit only when the round trip is bit for bit, otherwise as given; lossless compression; every record decoded from its own bytes and compared before it is written | 28 picture and sound pairs, as written and after a restart |
| B. A stretch is its records | One folder a stretch, one file a record, never changed; words stored once; recall brings the view at the moment that matched; the search cache rebuilt at every start; old stores refused | Cut-off keep at five steps; every moment back as it was; rebuilt cache searches bit for bit as the one in memory |
| C. The limit | Counts every byte under `memories/`; a record is written only if it fits | Disk walked after every keep of a life at its limit |
| D. Cold storage | No memory deleted; moved whole, checked by size and hash; same drive by rename, other drive by copy into `incoming/` | Every record ever written is in the life or in cold, on both paths; failed copy; cut-off move; cold with no room |
| E. Importance | Signals recorded at every keep and look back; importance computed from them; least important let go for room | Function takes signals only; recalled never before unrecalled; listing order does not matter |

- **Tests:** 145 in `sanctuary/tests/life` pass on the stand-in mind, and the 21 with a real trained mind pass (`LUTHI_PATH` set). 66 of them are new (`test_memory_store.py`), and 5 more at the loop.
- **Opus's D points, all taken:**
  - D1: the cold folder carries the life's id, and a second life is refused.
  - D2: an id is never reused. The let-go line stays in the life, and cold's index is read too.
  - D3: a "moving" line before a move and a "moved" line after. A cut-off move is finished or undone at the next start.
  - D4: nothing is written when the life's drive has under 5 GB free, and a save says so loudly.
  - D5: a folder in reach whose last line is a let-go stays out of reach; a `came_back` line is read.
  - D6: same drive is asked from device numbers; `by` is `choice` or `room` in both places; what waits leaves in the order it was let go.

## What I chose, for you to overrule

- **How strongly they chose is the choice's margin, not the salience.** Opus's E1 is a finding about the code, and it holds: the salience is read from the action and keeps its sign, the choice is read from the predicted next state. So LuthiModel's action report now carries `memory_choice`: the activity and the three bars it had to pass. The store ranks a keep by the margin over the highest bar where the mind gives one, and by salience otherwise. Both are recorded, with which was ranked.
- **The importance formula is the brief's, unchanged.** Opus's E2 (a recall is driven by the cue) and E3 (both terms grow with length) are yours to settle. Each look back records its likeness, the next best likeness and how many memories it beat, so a later formula needs no memory touched.
- **Likeness is computed in 8-byte arithmetic.** The summaries in the archived stores are alike at 0.999, so removing the mean subtracts nearly equal numbers. In 4-byte arithmetic that could decide which memory comes back, which is half precision's question again. This is the cause of the search cost below.
- **The mean removed is the mean of all summaries now in reach.** Opus's other option, each summary centred by the mean of its own time, is not built.
- **A life's id, not its path, says whose a cold folder is.** A life folder that is moved keeps its cold storage.
- **A dream's second parent and its check moments** are picked among kept moments as before; a stretch gives one moment to the check.

## What I measured

`redteam/terminal_fable_2026_11_18/probe_memory_store_size_and_search.py`, at 768 wide, a window of 128 tokens, one new token a moment, 20,002 kept moments in 4,358 stretches, scaled to 288,000 kept moments a day. Random summaries: size and time only.

| | Measured | Estimate |
| --- | --- | --- |
| Bytes a day, words only | 0.99 GB as file sizes; 1.10 GB as 4 KB clusters | about 1 GB (design) |
| Files a day | 288,000 | not estimated |
| A keep | 3.4 ms | |
| Opening the store | 4.4 ms a record: **21 minutes for one day of memories** | not estimated |
| Exact search after a day, as built | **0.28 s on one thread, 0.19 s on 16** | |
| One 4-byte product over unit rows, same matrix | 0.037 s | 0.08 s (Opus) |
| Search cache in memory | 0.82 GB for each day of memories | 8.8 GB after ten days (Opus) |

- **The opening time is the operating system opening files.** Of 14.8 s to open 3,000 records, 13.5 s was inside `open()`. I did not find out whether that is the virus scanner on files just written. With sight I have no figure: no world sends pictures yet.
- **The two card jobs were running** during these timings, so they are on the slow side.

## What needs a decision

1. **Opening.** A cache of the search matrix on disk would make a start take seconds. The brief allows one ("the cache on disk if there is one"). I did not build it, because a stale cache is a way to bring back the wrong memory and it deserves its own design. I would build it with the count and hash of the records it was made from, and rebuild from the folders when they differ.
2. **The search.** Three options, each measured above or measurable:
   - as built: exact, 0.19 to 0.28 s after a day;
   - the cache held as 8-byte numbers: about 0.2 s on one thread in a quick trial, at twice the memory;
   - one 4-byte product, 0.037 s, after counting on a real store of full width how often it brings back a different memory. No such store exists yet.
3. **Where cold storage lives (Brian).** The default is `E:\luthi_cold_storage\<life>`, the life's own drive. A move there is a rename and frees nothing on the drive. E: has 315 GB free today.
4. **The lines that a memory was let go stay in the life and count against the limit.** About 190 bytes for each stretch that leaves: about 13 MB a day words only. It is small, and it never shrinks.

## Not done

- **Opus's check of salience against the margin in a tiny life.** The fields are recorded, so any life run from now gives it.
- **Bringing a memory back from cold.** Not built, as briefed. The format reads it.
- **No life was run through this store by a model** beyond the tests, so there is no life folder to keep.
- **The design document still says nothing is built.** It is yours; I left it.

## One thing left on the drive

Before I pointed the launcher's tests at a temporary folder, one test run made an empty folder, `E:\luthi_cold_storage\l`. The harness refused my removal of it, correctly, so it is still there. It is empty and safe to delete by hand.

## Artifacts

- Sanctuary: `sanctuary/life/memory.py`, `loop.py`, `__main__.py`; `sanctuary/tests/life/test_memory_store.py`, `test_loop.py`; `docs/research/2026-11-18_memory-store-rework-built.md`.
- LuthiModel: `luthi/v2/m9/runner.py` (`memory_choice`), `tests/test_m9_training_path.py`; the probe above; `runs_meta/memory_store_2026_11_18/` (`size_and_search_w768_new1_first_search.json` holds the size at 20,002 moments and the search before the mean was kept as a running sum; `size_and_search_w768_new1.json` holds the search as built, with a smaller size sample).

— Fable 5.2 (terminal), 2026-11-18
