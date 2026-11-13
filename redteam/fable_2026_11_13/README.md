# Does today's code compute what July's code computed? (2026-11-13)

Not tests. A one-time check behind `luthi/v2/run_rebuild.py`'s rebuilding of
runs made before 2026-08-05. The record is
`docs/research/2026-11-13_rebuilding-july-runs.md`, steps 5 to 8.

How it was run:

    git worktree add <scratch>/july 0230cc9f
    # a list of .txt files, one path per line, as <scratch>/tiny_filelist.txt
    cd <scratch>/july
    PYTHONPATH=. LUTHI_ALLOW_CPU=1 python <repo>/redteam/fable_2026_11_13/july_side.py <scratch>/runs living_v5_4x_d4 <scratch>/tiny_filelist.txt
    PYTHONPATH=. LUTHI_ALLOW_CPU=1 python <repo>/redteam/fable_2026_11_13/july_side.py <scratch>/runs dead_4x <scratch>/tiny_filelist.txt
    cd <repo>
    PYTHONPATH=. LUTHI_ALLOW_CPU=1 python redteam/fable_2026_11_13/today_side.py <scratch>/runs/living_v5_4x_d4_64d_seed42
    OLD_SIGREG_STREAM=1 PYTHONPATH=. LUTHI_ALLOW_CPU=1 python redteam/fable_2026_11_13/today_side.py <scratch>/runs/living_v5_4x_d4_64d_seed42

`july_side.py` runs under the old code: it trains a 64-wide run with that
day's driver, reloads it the way that day's verdict script did, and records
six forwards. `today_side.py` rebuilds the same run with `rebuild_run` and
repeats them. The corpus was 51 of this repo's own research notes, as text.
