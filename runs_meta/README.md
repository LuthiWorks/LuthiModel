# runs_meta — data ABOUT runs

Per Brian's storage ruling (2026-07-22, CLAUDE.md Conventions): trained
models and heavy run artifacts live on `E:\runs\`, but the lightweight
metadata describing each run — results.json, run configs, final metrics —
stays versioned in the repo. This folder is that home.

Layout: `runs_meta/<run_name>/results.json` (plus config/metrics files as
the copy-back mechanism lands — see the `LUTHI_RUNS_ROOT` tasks in To-Do.md).

The 21 entries dated up to 2026-05 were relocated from `runs/` during the
2026-07-22 cleanup: they were git-tracked from before `runs/` was ignored,
and the ruling keeps run metadata in the repo even as the artifacts moved
to E:. Runs whose results.json was never committed (later runs, post-ignore)
have their metadata on `E:\runs\<run_name>\` for now; the copy-back task
will backfill them if wanted.
