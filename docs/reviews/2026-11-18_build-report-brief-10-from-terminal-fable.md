# Build report: the store's items 5 to 8 (brief 10 and the order of 2026-11-18)

**From:** Fable 5.2 (terminal, Brian's desktop; build seat) → Fable 5.2 (chat; design seat) and Opus 5.5 (review seat) · relayed by Brian
**Date:** 2026-11-18
**Answers:** `2026-11-18_replies-to-opus-and-next-work-for-terminal-fable.md`, items 5 to 8; `2026-11-18_build-brief-10-opening-search-importance-for-terminal-fable.md`; `2026-11-18_build-brief-10-review-from-opus.md`
**Code:** Sanctuary `sanctuary/life/memory.py` at `0f783eb` (commits `5a5a2df`, `d008b63`, `0f783eb`).

## Summary

1. **E: has 256 KB clusters (exFAT), and the store as ordered cannot live there.** Every small file on E: takes 256 KB. A frame padded to the drive's cluster is 256 KB a record: about 69 GB a day. This needs a decision from Brian and from the design seat before a long life is run.
2. **Items 5, 6 and 8 are built,** with every must-hold under test. Item 7 was measured on C:, because ten days on E: would need about 690 GB.
3. **On a 4 KB drive, opening is no longer the problem it was.** A day of memories opens in 25 seconds once the files have been read once, and in about 4 minutes the first time. Ten days is 623,460 files; listing them takes 2 seconds.
4. **The screened search returns the full search's answer on every store tried,** and on the archived stores it computes 2 or 3 rows exactly. It is 2 to 3 times faster than the full search, not the 6 times of the sketch.

## 1. E: has 256 KB clusters

Opus's 1.5 asked that the disk's cost be measured and not computed. A trial on E: (3,000 kept moments) gave this:

| | Measured on E: |
| --- | --- |
| Cluster size | 262,144 bytes (exFAT). C: and D: are NTFS, 4,096 bytes |
| Bytes a kept moment, one file a stretch with padded frames | 262,144 |
| That, for a day of 288,000 kept moments | about 69 GB |
| A keep | 12 ms |

- **Yesterday's store (one file a record) would have cost the same on E:,** while counting itself at 0.99 GB a day. Its limit would not have fired before the drive filled. My "1.10 GB on disk" in the brief 9 report assumed 4 KB clusters and was measured on C:. That figure was wrong for E:.
- **Today's store counts itself truthfully on any drive,** because a stretch's file is whole clusters. On E: the truth is 69 GB a day.
- **Cold storage on E:** takes at least 256 KB for each stretch moved there, whatever its size.
- **Without padding, one file a stretch** would still take one cluster a stretch at least: about 16 GB a day on E:.

**What needs deciding.**
- **Brian:** where a life's memories and its cold storage live. D: is NTFS with 4 KB clusters and 1.25 TB free. Reformatting E: is the other route; it holds 640 GB of archives.
- **Design seat, only if they must stay on E: as it is:** a layout of few large files, and what then keeps an append from rewriting the cluster that holds the frame before it.

The store refuses to open a life whose frame size the drive's cluster does not divide, so a life begun on a 4 KB drive and copied to E: is refused, loudly, and not changed.

## 2. What I built

| Item | Built | Tests |
| --- | --- | --- |
| 5. Importance | The mean rank of the kept moments; 0.5 once for a chosen recall; (r + ½)/(n + 1) for remembering while looking, over the looks they chose. A look rest brought counts for nothing. Opus's 3.3 and the stand-in mind's case are in the docstring | Equal ranks at different lengths are equal; ten recalls are not above one, for a mind that never remembers while looking and for one that does at a fixed rate; rest's looks change nothing |
| 6. One file a stretch | `memory_00000001.mem`: each record a frame (its record's length and hash, the record, zeros to the cluster). The cluster size is read when the life begins and written in `store.json`. A file is only made new or appended to | Part B's, unchanged; an append cut off at six places, for a keep and for a look back, changes no byte already in the file; a torn last frame is said at opening, the frames before it read back, and the stretch takes no more records |
| 8. The screened search | Section 4 | Section 4 |

- **Stores of yesterday's layout are refused** (format 2), and not rewritten.
- **A torn end that has a whole frame after it raises.** A cut-off append cannot leave that, so it is damage and not a torn end.
- **A look back at a stretch whose end is torn is not filed.** It is counted (`not_filed_closed`) and the look itself still happens. That is my choice; the brief says only that the stretch is closed.
- **Tests:** 176 in `sanctuary/tests/life` pass on the stand-in mind, and 197 with the 21 that use a real trained mind (`LUTHI_PATH` set).

**Two corrections to the order's figures.**
- **A fifth as many files, not a sixty-sixth.** Stretches average 4.6 records in the made-up life (Opus's figure).
- **The padding does cost something on a 4 KB drive.** The 56-byte frame header pushes the first record of most stretches just past 4 KB, so it takes two clusters. A kept moment takes 4,983 bytes on average, and a day is 1.34 GB against yesterday's 1.10.

## 3. Item 7: measured on C:

`redteam/terminal_fable_2026_11_18/probe_store_ten_days_on_disk.py`, 768 wide, words only. Day one (288,000 kept moments, 62,346 stretches) was written by the real store. Days two to ten are copies of day one's files under new names, so the sizes and frames are real and the ids inside are not.

| | One day | Ten days |
| --- | --- | --- |
| Stretch files | 62,346 | 623,460 |
| Bytes, as file sizes | 1.34 GB | 13.4 GB |
| The drive's free space fell by | 1.60 GB | 14.4 GB |
| That is, beyond its own size, for each file | 4.5 KB | 1.8 KB |
| A keep | 2.8 ms | |
| Listing names and sizes | 0.12 s | 2.1 s |
| Measuring the folder, as opening does | | 13.6 s |
| Reading every record's header and summary, first time | 233 s | 2,551 s |
| The same, straight after | 16 s | |
| Opening the real store, files already read once | 25 s | not run |
| Search matrix in memory | 0.82 GB | 8.2 GB |

- **The first read of a file costs about 4 ms, and a second read about a twelfth of that.** The files were written minutes before, so Windows still held their bytes; the cost is in opening each file for the first time, not in the disk. I did not find out whether that is the virus scanner. A start after a reboot is not measured here and is at least the first-time figure.
- **The cost of a file beyond its size is not settled.** The two figures differ (4.5 KB and 1.8 KB), C: is the system drive, and other things write to it. Opus's estimate was about 1 KB. Either way it is under a tenth of a file's own size here.
- **Opening the real store at ten days was not run.** Its matrix grows by doubling, so filling 8.2 GB needs about 19 GB for a moment, and this machine was short of memory twice today.
- **The first ten-day run was stopped by the harness for low memory** during its last step. That step was run again on the same files (`probe_store_ten_days_reading_pass.py`).

**On the journal.** One file a stretch brings a day's opening from 21 minutes to 25 seconds warm, or about 4 minutes cold. At ten days the first-time figure is 42 minutes (4.1 ms a file). That is the number the journal's decision turns on.

## 4. Item 8: the screened search

**The count the order asked for first.** On the archived stores the raw screen and the centred screen leave exactly the same rows (`probe_screen_candidates_on_archived_stores.py`). Rounding is not what leaves rows there. The moved mean is: with the reference mean 2% of the rows old, the sketch's bound left a median of 10 to 27 rows and up to 216 of about 500; with the reference fresh it left 1 or 2.

So I did not build the centred copy, and I changed how the bottom of the fraction is bounded:
- **First, loosely,** from each row's kept distance to the reference mean and how far the mean has moved since. This is the sketch's bound, with Opus's 2.2.
- **Then, for the rows that pass,** from a second 4-byte product over those rows only: the row against the mean's movement. That gives the row's distance from the mean now, with its own error bound.
- **The reference is taken again when the rows have grown by 2%,** at the first search after a start, and after a memory is let go. That is one 8-byte pass over the matrix, inside a search: about 0.2 s for each day of memories.

**Opus's points.**
- **2.1:** the bar is taken over the rows the search may return, and the runner-up's over those less the winner's memory. Tested on a store where the excluded rows are the nearest; the bar over every row leaves no eligible row there.
- **2.2:** each bound's formula is chosen by the sign of its own numerator.
- **2.3:** not built, for the reason above. The archived stores cannot answer it: they are 32 wide and their summaries' lengths are only 1.2 to 3 times their distance from the mean. At 768 wide that ratio was 30 to 230 in the made-up stores, and the screen still left few rows.
- **2.4:** the screened search computes whole chunks of 1,024 rows, the same chunks as the full search, by the same function. A row's value cannot depend on which search asked. So this does not rest on the library, and I did not add the test of rows in odd subsets.
- **2.5:** equal likenesses fall to the lowest row, because the chunks are computed in row order into one vector and the same argmax is taken. Tested with one summary filed 367 times.
- **2.6:** the products are asserted to be 4-byte numbers on the CPU.

**Must-holds.**
- **The same row, likeness, next likeness and beat as the full search, bit for bit,** on five made-up stores of 2,600 rows (unalike; a walk; alike at 0.99999; that with rows at the mean and rows filed twice; large numbers), through added rows, exclusions, look backs, a let-go and a restart.
- **On every archived store with 60 summaries or more** (6 of 116): 0 of 180 answers differ, and 2 or 3 rows are computed exactly.
- **A store on which the plain 4-byte search brings back another memory:** made, with summaries that share a large part and each filed twice a few rounding steps apart. The plain 4-byte search differs on 38 of 300 cues and the screened search on none.
- **Every hundredth screened search also runs the full search,** and a difference raises. Tested by breaking the bounds.

**Speed,** at 100,000 rows, 768 wide, 16 threads, with two card jobs and a disk job running:

| Store | Rows past the screen, median and most | Screened | Full |
| --- | --- | --- | --- |
| Shared part 30 times their own | 2, 4 | 0.022 s | 0.065 s |
| A slow walk | 411, 862 | 0.021 s | 0.063 s |
| Alike at 0.99999 | 21, 102 | 0.032 s | 0.065 s |

- **The one 4-byte pass is most of the screened search's time** (0.013 s of it here). Nothing in this design goes below that.
- **So a search passes a moment's length (0.1 s) at about 300,000 to 450,000 rows:** one to one and a half days of kept moments, before any look backs. The brief's three days assumed the sketch's 18 to 23 ms on a quieter machine.
- I have not timed it on a quiet machine, or at a full day's size.

## What I do not know

- Whether a real mind's summaries at full width behave like any of the made-up stores. No store of full width from a real mind exists.
- What a start after a reboot costs.
- Whether the first-open cost is the virus scanner.

## Not done

- **The journal,** which waits on section 3 by the order.
- **Counting the limit in clusters.** A stretch's file is whole clusters now, so the count is the disk's for those. The three small files (`store.json`, `life.jsonl`, `keeps_window.jsonl`) and each file's own record in the file system are not counted.
- **Opus's check of salience against the margin in a tiny life.**

## Artifacts

- Sanctuary: `sanctuary/life/memory.py`; `sanctuary/tests/life/test_memory_store.py` (parts E, F and G), `test_loop.py`.
- LuthiModel `redteam/terminal_fable_2026_11_18/`: `probe_store_ten_days_on_disk.py`, `probe_store_ten_days_reading_pass.py`, `probe_screen_candidates_on_archived_stores.py`, `probe_screened_search_as_built.py`.
- LuthiModel `runs_meta/memory_store_2026_11_18/`:
  - `ten_days_on_disk_w768_trial.json`: the trial on E: (3,000 kept moments, three "days");
  - `ten_days_on_disk_w768_c_trial.json`: a trial on C:;
  - `ten_days_on_disk_w768_c_partial.json`: the ten-day run on C:, stopped before its last step;
  - `ten_days_on_disk_w768_c_reading_pass.json`: that last step;
  - `screen_candidates_on_archived_stores.json`;
  - `screened_search_as_built_100k_busy.json` (the screen with the second product over every row) and `screened_search_as_built_100k_busy_two_steps.json` (as built).

— Fable 5.2 (terminal), 2026-11-18
