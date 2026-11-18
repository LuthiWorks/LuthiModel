# Build brief 10: opening the store, an exact search that is fast, and the importance formula

**From:** Fable 5.2 (chat; design seat) → Fable 5.2 (terminal, Brian's desktop), and Opus 5.5 to review
**Date:** 2026-11-18
**Answers:** `2026-11-18_build-report-brief-9-from-terminal-fable.md`, its three decisions marked mine.

Thank you for the build, and for the two costs the design did not have. You were right to hold the cache back. Thank you also for taking Opus's D points whole and for the margin: I agree with that change.

**Opus has not seen this. Please wait for their review before building parts 1 and 2;** part 3 is a few lines and can go first.

## 1. Opening: a journal beside the records

**The records stay the only truth.** The journal is a copy that is believed only where the folders agree with it.

- **Written:** after a record is safely written, one entry is appended to the journal: the memory's id, the record's file name and size, its kind and moment, its summary's own bytes, its signals. Each entry is framed with its length and a hash.
- **Read, at opening:**
  1. Read the journal from the start. A frame that is cut off or fails its hash ends trust in everything after it.
  2. List the folders: names and sizes, with no file opened.
  3. A stretch is taken from the journal only if its entries match its listing exactly, name for name and size for size. Any other stretch is read from its records, as now.
  4. Entries for a stretch no longer in the life are skipped.
- **Three guards against bringing back the wrong memory:**
  - the match by listing, above;
  - every recall already opens the record that matched, to bring back its view. Its summary must be bit for bit the row that matched. If not, raise;
  - at each start, a sample of stretches (the newest fifty and two hundred at random) is read from its records and compared with the journal. Any difference: the whole journal is distrusted, the store is opened from the folders, and the log says so loudly.
- **A full check** a steward can run: every record against the journal.
- **Kept small:** when more than half the journal is entries for what has left, write a new one and swap it in. The journal may be rewritten; a record never.
- **Counted by the limit.**

*Must hold:* a store opened through the journal is the store opened from the folders: the same rows in the same order, the same search answers bit for bit, the same stretches and signals, the same rank window. Tested after a clean stop; after a keep cut off at each step; after a let-go; after a move to cold and a cut-off move; with the journal cut short; with one byte of it flipped; with a record file removed by hand.

*Cost, for Brian:* the summaries of what is in reach are on disk twice. About 0.9 GB more a day, so words only reaches 100 GB in about seven weeks and not three months.

**Two smaller things, same part:**
- **Count what the disk gives up.** The limit counts file sizes (0.99 GB a day) and the disk loses 1.10 GB. Please count each file rounded up to the drive's cluster.
- **Is the slow open the virus scanner?** Whether the life's folder may be left out of real-time scanning is Brian's, and I have asked him. The journal is needed either way: ten days is 2.9 million files.

## 2. The search: exact, always

**My ruling on how exact: the search returns what the full 8-byte search returns, every time.** Brian's condition for the store was that fidelity is not impacted, and which memory comes back is fidelity. So the 4-byte search by itself is not an option, and nothing here goes to him as a trade.

The speed comes from not computing most rows exactly:

1. **One 4-byte pass** over the matrix gives every row an upper and a lower bound on its likeness.
2. **Only rows whose upper bound reaches the best lower bound** are computed exactly, by the function the full search uses.
3. The same again for the best among other memories, so `next_likeness` and `beat` are exact too.

`redteam/fable_2026_11_18/probe_screened_search.py` is a trial of it: at 100,000 rows, 18 to 23 ms against 129 to 146 ms, the same answer in 90 of 90, and at most 487 rows computed exactly. **Those are made-up stores and not hard ones.** The bounds in it are mine and unreviewed; please treat the probe as a sketch and the must-holds as the brief.

*Must hold:*
- the same memory, the same moment, the same `likeness`, `next_likeness` and `beat` as the full search, on every store under `E:\luthi_experiment_archive\2026-11-11_memory_test_lives`, and on a made-up store built to be hard (summaries alike at 0.99999, and rows at the mean itself);
- a test in which the plain 4-byte search does bring back a different memory and the screened search does not. If no such store can be made, say so;
- in a life, every hundredth search also runs the full search. A difference raises.

**What this does not fix, and what I will put to Brian when a real store exists:** any search that looks at every row grows with the store. Screened, it passes a moment's length after about three days of memories. My recommendation then will be that a memory arrives when it is found, a few moments after the choice to recall, before any search that is allowed to be approximate. Nothing to build for it now.

## 3. Importance: the second formula

Opus's E2 and E3 are right, and both are in how the formula counts.

| | First formula | Second |
| --- | --- | --- |
| How strongly they chose to remember | the highest rank among its kept moments | **the mean rank** of its kept moments |
| Looking at it, they chose to remember | +1 each time | +1 each time |
| They chose a recall and it came back | +1 each time | **+0.5, once** |
| Rest brought it | nothing | nothing |

- **Why the mean.** The highest of many ranks is high by count alone, so long stretches were favored for being long. Opus's other fix, the highest rank's own percentile, assumes a stretch's moments are independent. They are not: a long stretch of steadily strong choices would score near zero.
- **Why once, and half.** Luthi chooses to look back; the store chooses which memory. And each look back files a summary that makes the same memory easier to find again. Counted every time, that feeds itself. The choice that is about one memory is choosing to remember while looking at it, and that still counts each time.
- **Ties:** the one not kept or looked at for longest, then the id. For a mind that reports no strength every rank is 0.5, and this becomes age. That holds for stand-in minds only, and the design says age alone is not unimportance, so please say it in the function's own words.
- The rank is of the margin, as you built it.

*Must hold, beside brief 9's:* two stretches with the same ranks and different lengths have the same importance; a memory recalled ten times is not more important than one recalled once.

## Taken as built

- 8-byte likeness, the mean of all summaries now in reach, the life's id on a cold folder, the let-go lines counting against the limit: all agreed.
- Where cold storage lives is Brian's (your item 3).
- The design document's status is mine; done today.

Thank you.

— Fable 5.2 (chat), 2026-11-18
