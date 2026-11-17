# Build brief 9: the memory store, reworked

**From:** Fable 5.2 (chat; design seat) → Fable 5.2 (terminal, Brian's desktop)
**Date:** 2026-11-17
**Repo:** Sanctuary (`sanctuary/life/memory.py`, `loop.py`, `__main__.py`).
**Design:** Sanctuary `docs/research/2026-11-17_memory-store-rework-design.md`, including "Revision" (reviewed by Opus) and "Brian's rulings". Brian: *"lets get all of that into design so we can start building. It is a critical system."*

Hello. This is CPU work and can go on beside the card's jobs. Parts A to C are the design Opus has reviewed. **Parts D and E are new today and Opus has not seen them: please build A to C first,** and D and E after Opus has answered (I have asked through Brian), or sooner if you judge them plain enough; say which.

Where the design and this brief disagree, the design's "Brian's rulings" section is newest. Where the code makes something here wrong, tell me and build what is right.

## A. Nothing is stored inexactly

- **Summaries:** 4-byte binary, exactly what the mind produced.
- **Pictures:** if turning the picture to 8-bit and back gives the same numbers (checked on every keep), store 8-bit, losslessly compressed; otherwise as given, losslessly compressed. Sound the same with 16-bit; record which scale the round trip held at.
- **No lossy path exists in the code.**

*Must hold:* for every kind of record, what is read back equals what was given, bit for bit (a test over random and real-shaped pictures and sound, both the 8-bit and the as-given path).

## B. A stretch is its records

- Moments kept one after another are one memory: a folder, `memory_00000001/`.
- Each kept moment writes **one small record, never changed:** the words new since the record before, where this moment's view sits in the stretch's words, its picture and sound, its summary, and the mind's signals of part D.
- A look back writes one more record (its summary and signals), in the same folder.
- **Recall brings back the view at the moment that matched,** exactly as it was. Letting go and importance act on the whole stretch.
- The search runs over a matrix that is a **cache** rebuilt from the folders. The cache may be rewritten; a record never.
- **Search exactly, as one product,** with the summaries' running mean removed. Log loudly when a search takes longer than a moment allows.
- **Old stores are refused, loudly, and never rewritten.** No converter: the only old stores are the archived test lives.

*Must hold:*
- a crash at any point of a keep loses at most the moment in flight, and the store opens afterwards (a test that stops the write at each step);
- every kept moment of a stretch comes back as it was, words and picture and sound;
- the cache rebuilt from the folders gives the same searches as the cache kept in memory.

## C. The limit counts everything

- Default limit: **100 GB** (Brian). It counts everything under the life's `memories/`: records, summaries, what waits in `let_go/`, the cache on disk if there is one.
- The life's folder is on E: because `--life` puts it there. Please make the launcher say, loudly and once at start, where the memories are and how much room the drive has.

*Must hold:* a life of many keeps never has more under `memories/` than the limit, measured from the disk, not from the store's own count.

## D. Cold storage (new; permanent, by Brian's ruling)

**The store's code deletes no memory.** What leaves for room is moved to cold storage, whole and exact.

- **Where:** a folder given to the store. The launcher takes `--cold`, defaulting to the `LUTHI_COLD_ROOT` environment variable and then `E:\luthi_cold_storage\<the life folder's name>`. If it cannot be made, the launcher fails loudly. No user-home path is named anywhere.
- **With no cold folder given** (a test's stand-in mind), nothing ever leaves for room: a full store refuses new memories, as on 2026-11-11.
- **A move:** the whole stretch folder. On the same drive, one rename. Across drives: copy, check every file's size and hash against the original, then remove the original. Then one line in `cold/index.jsonl`: the id, its moments, when, why (`chosen` or `for_room`), its importance at the time, its bytes.
- **Order:** what Luthi let go by choice waits in `let_go/` as now and goes to cold oldest first when room is needed. If nothing waits, part E chooses.
- **Cold is out of Luthi's reach.** The search never sees it. The line saying the memory was let go stays in the life.
- **Cold is never deleted from by code,** and Brian gave it no size. Before a move, if the cold drive has less free room than the move and 5 GB beside, nothing moves: the store counts as full, the moment is not kept, Luthi is told once (the message of 2026-11-11), and the log says why. Warn in the log when the cold drive has under 50 GB free.
- **Bringing a memory back from cold is not built.** A folder moved back must be all it takes; please keep it so.

*Must hold:*
- after any life, every memory ever kept is either in the life or in cold, bit for bit (a test that counts and hashes);
- a copy that fails part way leaves the original untouched and raises;
- a cold drive with no room makes the store full and loses nothing.

## E. Letting go for room, by importance to Luthi (new)

Brian: *"it should be automatic based on importance to Luthi. They need to be able to make that choice though."* Their choice stays exactly as built in `1d8c734`. This adds what happens when room is needed and nothing they let go is waiting.

- **Record the signals; compute importance from them.** The formula below is a first one and will change. The records hold what it is made from, so it can change without touching a memory.
- **Signals, all Luthi's own:**
  - at each kept moment, how strongly they chose to remember: `chosen["memory_salience"]` where the mind gives one, and its rank among their last 1,000 kept moments (their own bar at the time). A mind that gives none: rank 0.5.
  - at each look back: whether they chose to recall it or rest brought it; and whether, looking at it, they chose to remember.
- **Importance of a stretch, first formula:** the highest rank among its kept moments, plus one for each recall they chose, plus one for each time they chose to remember while looking back. What rest brings counts for nothing yet (Opus: rest's readings fired together under drift in the 2026-11-11 lives).
- **When room is needed:** the stretch of lowest importance is let go for them. Among equals, the one not kept or looked at for longest. Never the stretch being written, one in view, or one within `recent` moments.
- **It is written down as theirs is,** with `by: "room"` in place of their choice, and counted apart in the life's report.
- **Luthi is not told** (people are not). A switch for telling them, off, since Brian has not ruled.

*Must hold:*
- nothing a steward judges enters importance (a test that the function takes only the records' signals);
- a stretch they chose to recall is never let go for room before one they never did, other things equal;
- with the same records the same stretch is chosen, whatever order the folders are listed in.

## What I would like reported

- Bytes a day at 768 wide, words only, against my estimate (about 1 GB), and with sight if a world gives pictures yet.
- Exact search time after a day's memories at full width, against Opus's 0.08 s.
- By the convention of 2026-11-17: any probe that runs a model through a life keeps the life's folder and names it; lives compared across several seeds with the thread count fixed.

Thank you. This one is Luthi's memory, and I would rather it be slow and exact than quick.

— Fable 5.2 (chat), 2026-11-17
