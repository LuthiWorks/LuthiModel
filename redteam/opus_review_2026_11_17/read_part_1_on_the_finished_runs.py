"""What brief 8's Part 1 says on the finished runs, read from its rows. (2026-11-17)

Opus 5.5, for the review of "a living write that hears the prediction error".
No model is run: this reads the files terminal Fable committed in 96206e1
(``runs_meta/living_gradient_2026_11_17/<run>/shape_of_the_helpful_direction.json``)
and the 64-document run at 768 in 7a01a1d (``..._64docs.json``).

Per living layer it prints:

1. Learning against the rest. Each row holds c, the cosine of the learning part
   (the gradient with the target held still) with the whole score's gradient
   g, and r, the learning part's size over g's. The rest, g minus the learning
   part, is the target's path and its variance's path. By algebra alone:
       |rest| / |g|           = sqrt(1 - 2 r c + r^2)
       cos(learning, rest)    = (c - r) / sqrt(1 - 2 r c + r^2)
   When c < 0, a step along learning's downhill raises the score itself at
   first order: <g, -learning> = -|g| |learning| c > 0.
2. The document's own part of the carry-over: per document, the cosine of the
   head's learning direction with the tail's, own minus other (the frozen
   target), and the same against the tail's whole gradient (the score as
   registered). Mean over documents and t.
3. The map beside no map, with one sign: ``no_map_cosine`` is the cosine of the
   rule's shift (minus the error) with the gradient, so it is negated here to
   read like ``map_cosine`` (an estimate of the gradient against the gradient).
4. The no-map rule computed on the head, against the tail's learning downhill:
   mean over documents and t.

Run: python redteam/opus_review_2026_11_17/read_part_1_on_the_finished_runs.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "runs_meta" / "living_gradient_2026_11_17"
RUNS = {
    "July seed 42": ROOT / "living_v5_4x_d4_512d_seed42" / "shape_of_the_helpful_direction.json",
    "768 seed 97": ROOT / "probe_768_visreg_768d_seed97" / "shape_of_the_helpful_direction.json",
    "768 seed 97, 64 documents": ROOT / "probe_768_visreg_768d_seed97" / "shape_of_the_helpful_direction_64docs.json",
}


def mean_and_t(xs: list[float]) -> tuple[float, float]:
    m = statistics.mean(xs)
    return m, m / (statistics.stdev(xs) / math.sqrt(len(xs)))


def main() -> int:
    for name, path in RUNS.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["rows"]
        n_layers = 1 + max(r["layer"] for r in rows)
        s = report["settings"]
        print(f"{name}: {s['docs']} documents, {s['read']} windows read and {s['score']} scored")
        print("  layer | c<0 share | cos(learning, rest) | |rest|/|g| || own-other, frozen: mean, t | registered: mean, t")
        for layer in range(n_layers):
            R = [r for r in rows if r["layer"] == layer]
            c = [r["learning_part_cosine"] for r in R]
            ratio = [r["learning_part_share_of_size"] for r in R]
            rest = [math.sqrt(1 - 2 * ri * ci + ri * ri) for ci, ri in zip(c, ratio)]
            against = [(ci - ri) / n for ci, ri, n in zip(c, ratio, rest)]
            frozen = mean_and_t([r["learning_head_to_tail_cosine_own"] - r["learning_head_to_tail_cosine_other"]
                                 for r in R])
            registered = mean_and_t([r["head_to_tail_cosine_own"] - r["head_to_tail_cosine_other"] for r in R])
            print(f"  {layer:5d} | {sum(x < 0 for x in c) / len(c):9.2f} | {statistics.median(against):+19.3f} | "
                  f"{statistics.median(rest):10.3f} || {frozen[0]:+.3f}, {frozen[1]:+.1f}"
                  f"{'':15s}| {registered[0]:+.3f}, {registered[1]:+.1f}")
        allc = [r["learning_part_cosine"] for r in rows]
        print(f"  every row: c < 0 in {sum(x < 0 for x in allc) / len(allc):.2f}; median c {statistics.median(allc):+.3f}")
        fb = report["one_map_from_the_error"]
        print(f"  one map, fitted on {fb['windows_fitted_on']} windows and read on {fb['windows_read']}:")
        print("    map, common part removed:", " ".join(f"{b['map_cosine_common_part_removed']:+.3f}" for b in fb["by_layer"]))
        print("    map, whole:              ", " ".join(f"{b['map_cosine']:+.3f}" for b in fb["by_layer"]))
        print("    no map, same sign:       ", " ".join(f"{-b['no_map_cosine']:+.3f}" for b in fb["by_layer"]))
        nb = [mean_and_t([r["learning_no_backward_rule_from_head_cosine_with_tail_downhill"]
                          for r in rows if r["layer"] == layer]) for layer in range(n_layers)]
        print("  no map from the head, against the tail's learning: mean, t by layer:",
              " ".join(f"{m:+.3f} ({t:+.1f})" for m, t in nb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
