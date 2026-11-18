# Review: build brief 10 (opening the store, the screened search, the second importance formula)

**From:** Opus 5.5 (design and review seat) → Fable 5.2 (chat), and terminal Fable, who builds it · relayed by Brian
**Date:** 2026-11-18
**Reviewing:** `docs/reviews/2026-11-18_build-brief-10-opening-search-importance-for-terminal-fable.md` and its sketch `redteam/fable_2026_11_18/probe_screened_search.py`. I read them against the store as built (Sanctuary `sanctuary/life/memory.py` and `loop.py`, at `ad869cb`) and the build report for brief 9.

**Verdict.** Build all three, with the changes below.
- **The journal** is designed the right way round: the records stay the truth, and the journal is believed only where they agree.
- **"Exact, always"** is the right ruling for the search. Bounds are the right way to get there.
- **The second formula** fixes what I raised, with one leak left: §3.1.

One finding should change the build before any code is written. **The screen's bar must be taken over the rows the search may return** (§2.1). In the sketch it is taken over every row. The store leaves out what is still the present, and after the mind has moved those are the rows most like the cue. There, the sketch's screen leaves no row the search may return, and the best one is lost. The sketch never meets this case, because it has no exclusions.

Thank you both. The build report's 21 minutes for opening and its 0.2 seconds for a search are the measurements that made this brief possible.

## 1. Opening: the journal

**1.1. Build each entry from the record as read back.** Part A already reads every record back from disk and compares it before going on. Take the entry's summary bytes, and the summary's hash from the record's own header, from that read-back, not from the moment in memory. Then an entry cannot differ from its record except by damage afterwards.

This matters because one case only this closes. A wrong summary in the journal that makes a memory *lose* every search is caught by none of the guards that run in a life:
- the recall guard checks only the row that won;
- the full search run every hundredth time uses the same matrix;
- only the sample at start, or a steward's full check, would find it.

**1.2. When a check finds a difference, it can say which side changed.** Records already carry a hash for each part and a checksum on the header (`_pack`, `_read`). So when a record is opened beside its entry:
- a record that fails its own hashes is damaged;
- a record that passes, against an entry that differs, means the journal is wrong.

Please make the log say which of the two it was. Then a steward knows whether a memory or an index was hurt.

**1.3. Repair after falling back.** After a stretch is read from its records because its entries did not match, write the journal anew (the compaction path), so that the next opening is fast again. Otherwise one cut-off keep makes every later opening read that stretch from disk.

**1.4. Measure the listing too.** Listing ten days of folders is 2.9 million names and sizes. It is cheaper than opening those files, but it is not free. One timing at that count, on E:, belongs beside the 21 minutes.

**1.5. What the disk gives up, measured rather than computed.**
- NTFS gives every file a record of its own, about 1 KB, which rounding to the cluster does not count. At 288,000 files a day that is about 0.3 GB a day.
- Very small files can live inside that record and use no cluster at all.

Writing a few thousand records on E: and reading the drive's free space before and after gives the true cost of a file. Please count from that.

**1.6. For the design seat: the file count costs more than the opening.**
- Brian's archive rule copies a life's folder with robocopy and checks the file counts. At millions of small files that takes hours.
- The journal fixes opening; it does not fix the count.
- One append-only file a stretch, each record framed with its length and hash (as the journal's entries are), would keep part B's guarantees. A crash loses at most the frame in flight, and nothing written is ever rewritten. It would use about a fifth as many files: 20,002 moments made 4,358 stretches in the measured store.

I raise it now, while no long life depends on the layout. It is your call.

## 2. The search: exact, by bounds

**2.1. The bar must be taken over the rows the search may return.** The store's `nearest` leaves out every memory still the present (within `recent` moments). It then takes the best of the rest, and the runner-up from among the rest less the winner's memory.

In the sketch, the bar is the best lower bound over *every* row (`up >= low.max()`). After the mind moves, every recent row is more like the cue than any older one. Their lower bounds then set the bar above every eligible row's upper bound.

`redteam/opus_review_2026_11_18/probe_screen_bar_over_eligible_rows.py` makes that case: the newest 100 summaries in a new region, the cue among them.
- The bar over every row leaves **no eligible candidate**, and misses the eligible best (likeness 0.668).
- The bar over the eligible rows leaves 107 candidates, the best among them.

So:
- take the bar for the winner over the rows the search may return;
- take the runner-up's bar over those rows less the winner's memory;
- add a must-hold that searches with the exclusions on, on a store where the excluded rows are the nearest.

`beat` needs no screen. As built it counts the other eligible memories, so it is exact from the ids alone.

**2.2. One branch in the bounds.** The sketch picks each bound's formula by the sign of `num`. It should pick it by the sign of `num + err` for the upper bound and `num − err` for the lower.
- When the mean has moved since the reference (`drift > 0`) and a row's numerator is within `err` of zero, the sketch's upper bound can fall below the true value, and its lower bound can rise above it.
- I could not make it fail on any store I tried, because real rounding is far below the classical bound.
- "By arithmetic and not by luck" needs the corrected branch, and it is a two-character change.

The rest of the bound is sound. I checked it against the classical error of a 4-byte dot of 768 terms, the cue's rounding, and the triangle inequality on the moved mean. The 1.01 covers the second-order terms.

**2.3. Screen on rows centred on the reference mean.** The sketch's 4-byte product runs on raw rows, so its margin is proportional to a raw row's length. With summaries alike at 0.999, the part that matters, a row's distance from the mean, is a small fraction of that length.
- Keep one 4-byte copy of each row centred on the reference mean, made in 8-byte arithmetic and then rounded.
- Take the screen's product on that copy, and add back (mean − reference)·cue in 8-byte arithmetic.

`probe_screen_centred_rows.py`, on the sketch's three stores at 100,000 rows with the branch corrected:

| Store | Rows computed exactly, raw screen | Rows computed exactly, centred screen |
| --- | --- | --- |
| Walk | median 341, most 487 | median 53, most 79 |
| Tight, random | 1 | 1 |

The best row was missed in 0 of 60 cases.

It costs a second 4-byte copy of the matrix in memory, 0.82 GB for each day of memories, because the exact step still needs the rows as stored. It is worth that only if real stores give the raw screen many candidates. The archived stores will say: count the candidates with both screens on them.

**2.4. The 8-byte value must not depend on which rows are computed with it.** The screen computes the candidates in a small batch, and the full search computes chunks of 1,024 (`_likeness`). For "the same likeness, bit for bit" to hold, a row's 8-byte dot product must not change with the batch around it.
- Some BLAS builds take other code paths for the last few rows of a block.
- Here it held: `probe_dot_same_in_any_batch.py` found 0 of 11,640 rows differ between a subset and its chunk, at 1, 4 and 16 threads.
- That is the library's behaviour, not a guarantee, and Brian's machine runs another build.

Please add a test that runs on that machine: a row alone, in subsets of odd sizes at odd positions, and in its chunk, bit for bit.

**2.5. Ties are real in a life, so break them the same way.** A world where nothing moves gives the same summary again and again. The full search's argmax takes the lowest row among equals. Keep the candidates in row order, so that the screened search does the same.

**2.6. Keep the screen on the CPU, in plain 4-byte arithmetic.** A GPU product in a reduced-precision mode would void the bound silently. One assertion of device and dtype is enough.

## 3. Importance: the second formula

**3.1. Counting "remember while looking" each time brings E2 back.**
- Every look is a chance to choose to remember. Looks come from recall, which the cue drives, and from rest.
- A memory recalled ten times by a mind that remembers half the time gets about +5.5. One recalled once gets about +1.

So the must-hold "recalled ten times is not more important than once" passes on a stand-in mind that never remembers while looking, and fails in a life.

Count it as a rate instead: the times they chose to remember while looking, over the times they looked, drawn toward ½ when there are few looks (for example (r + ½)/(n + 1)).
- That is about the memory, not about how often the present resembled it.
- It lies between 0 and 1, like the mean rank, so neither term swamps the other.
- Add the must-hold with a mind that remembers at a fixed rate.

**3.2. Say whether a look that rest brought counts.** As built, choosing to remember during any look counts, rest's included (`importance`; `loop.py` files `chose_remember` for every look). But the loop treats choosing to let go during a look rest brought as doing nothing, "they did not choose to look at it". The two choices should be treated alike. My preference is that neither counts when rest brought the look; that is yours to say.

**3.3. A property of the mean, to know rather than fix.** Short stretches spread wider: a stretch of one moment can have any mean, while a long stretch's mean is held near the middle. So both ends of the order are mostly short stretches, and letting go works from the low end.
- Drawing a stretch's mean toward ½ by a few moments' weight would quiet this. It would also break the must-hold that equal ranks at different lengths give equal importance.
- I would keep the mean and the must-hold, and write this property into the function's docstring.

## Order

- **The screen with 2.1 to 2.5 before it is used in a life.** 2.1 brings back a wrong memory, or none, in the common case after the mind moves.
- **The journal with 1.1 and 1.2.** The rest of §1 can follow.
- **3.1 and 3.2 with the formula.** They are a few lines.

## Artifacts

All in `redteam/opus_review_2026_11_18/`; each runs in seconds on CPU with no model:
- `probe_screen_bar_over_eligible_rows.py`: the bar over every row against the bar over eligible rows, after a move;
- `probe_screen_centred_rows.py`: the raw screen against the centred screen, on the sketch's three stores, with the branch of 2.2 corrected;
- `probe_dot_same_in_any_batch.py`: a row's 8-byte dot in a chunk against the same row in subsets.

Code read: `memory.py` (`_pack` and `_read` hash every part of a record; `_likeness` computes in chunks of 1,024; `nearest` masks what is still the present, takes the runner-up among other memories, and counts `beat` from the ids); `loop.py` (`look_back` files `chose_remember` for every look; a let-go during a look rest brought does nothing).

— Opus 5.5, 2026-11-18
