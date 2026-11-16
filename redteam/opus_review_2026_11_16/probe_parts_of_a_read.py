"""Do the four routes of a read add up? Read from the card's own results.

Opus 5.5 review of 2026-11-16 (Fable's draft Amendment 5, second version, (b)).
The draft tells the parts of a read apart with "only the write" against
"everything off". The card also ran each route switched off with the others
on (``*_off``), at the registered dose. With everything on, the write's part
in context is "everything on" minus "everything but the write". If the routes
add, that equals "only the write". Where they do not, the write acts through
another route, and "only the write" misses it.

At four windows consolidation has not fired yet (it first fires at window 15
on a July run), so "consolidation off" is the all-on read there.

Run from the repo root:  python redteam/opus_review_2026_11_16/probe_parts_of_a_read.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "runs_meta" / "living_channel_dose_2026_11_16"
RUNS = ("living_v5_4x_d4_512d_seed42", "probe_768_visreg_768d_seed97")


def per_doc(run: str, name: str, key: str = "nmse") -> list[float]:
    """B minus C per document."""
    rows = json.loads((ROOT / run / f"{name}.json").read_text())["per_doc"]
    return [r["B"][key] - r["C"][key] for r in rows]


def main() -> None:
    for run in RUNS:
        on = per_doc(run, "e1_read4_consolidation_off")       # all on, at 4 windows
        but = per_doc(run, "e1_read4_write_off")               # everything but the write
        alone = per_doc(run, "e1_read4_only_the_write")
        episodes = per_doc(run, "e1_read4_only_episodes")
        pull = per_doc(run, "e1_read4_only_the_pull")
        in_context = [a - b for a, b in zip(on, but)]
        n = len(on)
        print(f"{run}, 4 windows, {n} documents; B minus C in nmse (mean, then median):")
        for label, xs in (("everything on", on), ("everything but the write", but),
                          ("the write in context (on minus but)", in_context),
                          ("only the write", alone), ("only episodes", episodes),
                          ("only the pull", pull)):
            print(f"  {label:38s} {sum(xs) / n:+.3e}   {statistics.median(xs):+.3e}")
        top = max(range(n), key=lambda i: abs(in_context[i]))
        share = in_context[top] / sum(in_context) if sum(in_context) else float("nan")
        print(f"  the write in context: document {top} carries {share:.0%} of its sum; "
              f"only the pull's B minus C is exactly zero on {sum(x == 0.0 for x in pull)} of {n}")


if __name__ == "__main__":
    main()
