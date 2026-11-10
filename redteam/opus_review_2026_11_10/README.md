# Probes behind the 2026-11-10 review

Review: `docs/reviews/2026-11-10_small-path-resume-and-rest-response-from-opus.md`, and its follow-up, `docs/reviews/2026-11-10_follow-up-from-opus.md`.

Each probe is a demonstration, not a test: it prints what it finds, and its docstring records what it printed when the review was written. All run on CPU at test size, and each reproduced its numbers exactly on a second run. Times are for an idle two-core machine. Run from the LuthiModel root. The ones that live a Luthi life need Sanctuary importable (`PYTHONPATH=<Sanctuary>`) and `LUTHI_PATH=<LuthiModel>`.

| Probe | Section | Runs in |
|---|---|---|
| `probe_resume_past_the_taper.py` | B1: a resume past the taper's start; a cut at an epoch's end | ~1 min |
| `probe_resume_verdict_on_float_noise.py` | B1: the on-device verdict on float noise | seconds |
| `probe_small_path_loop.py` | A1: what a stop loses; saving after a raise | seconds |
| `probe_small_path_life.py` | A1, B2: the trunk's mode, restarts, the gauge, who writes the record | ~15 s |
| `probe_moment_baseline.py` | A3: the moment measurement with a persistence baseline | ~1 min a seed |
| `probe_rest_on_a_lived_day.py` | B3: rest over a day a real trunk lived | ~1 min |
| `probe_rest_backlog_over_days.py` | follow-up: what rest holds back, over five days | a few min |
| `probe_life_never_tapers.py` | follow-up: the living rate a life starts at | ~15 s |

The probes named `probe_*` are kept out of the suite on purpose (`tests/test_tests_live_in_tests.py`). Pytest still runs one when it is named on the command line.
