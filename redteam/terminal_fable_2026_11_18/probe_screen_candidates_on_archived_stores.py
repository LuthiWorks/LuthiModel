"""How many rows does the screened search still compute exactly, on the
archived stores? The raw screen against the centred one.

The design seat's order of 2026-11-18, item 8: "Count the candidates with the
raw screen and the centred one on the archived stores, and build the centred
copy only if the raw one leaves many." Opus's review of build brief 10, 2.1 to
2.3: the bar is taken over the rows the search may return, the branch of each
bound is chosen by the sign of the bound's own numerator, and the centred
screen takes its product on a 4-byte copy of each row with a reference mean
removed.

The archived stores (``E:/luthi_experiment_archive/2026-11-11_memory_test_lives``,
the zip) are of the old format. Only their ``summaries.jsonl`` is read, straight
from the zip; nothing is extracted or changed. For each store with at least
``--least`` summaries:

- the rows are the summaries in the order filed, as 4-byte numbers;
- the reference mean is the mean of the first 98% of the rows, and the mean
  removed is the mean of all of them, so the mean has moved since the
  reference as it does in a life;
- the newest ``--recent`` rows are the present and may not be returned;
- each cue is one of the newest rows itself (the mind recalling from where it
  is), and then a row picked from anywhere.

Reported for each store: how alike its summaries are as they stand, the
candidates each screen leaves (median and most, of the eligible rows), and
whether the eligible best was among them every time.

    C:/Dev/Sanctuary/.venv/Scripts/python.exe redteam/terminal_fable_2026_11_18/probe_screen_candidates_on_archived_stores.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import zipfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
INTO = REPO_ROOT / "runs_meta" / "memory_store_2026_11_18"
ARCHIVE = r"E:\luthi_experiment_archive\2026-11-11_memory_test_lives\2026-11-11_memory_test_lives.zip"
U = 2.0 ** -24


def exact(rows64: torch.Tensor, mean: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
    c = cue.to(torch.float64) - mean
    r = rows64 - mean
    norms = r.norm(dim=1) * c.norm()
    return torch.where(norms > 0, (r @ c) / norms, torch.zeros_like(norms))


def bounds(dot, length, n0, drift, shift_dot_c, cn, c32n, width):
    """An upper and a lower bound on every row's likeness. ``dot``: the 4-byte
    product; ``length``: the length of the 4-byte row it was taken on."""
    err = (width + 3) * U * torch.maximum(length, n0) * c32n * 1.01
    num = dot - shift_dot_c
    low_den = (n0 * (1 - 1e-9) - drift).clamp_min(0.0) * cn
    high_den = (n0 * (1 + 1e-9) + drift) * cn
    top, bottom = num + err, num - err
    up = torch.where(top >= 0, top / low_den.clamp_min(1e-300), top / high_den).clamp(max=1.0)
    low = torch.where(bottom >= 0, bottom / high_den, bottom / low_den.clamp_min(1e-300)).clamp(min=-1.0)
    return up, low


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--least", type=int, default=60)
    ap.add_argument("--recent", type=int, default=20, help="rows that are still the present")
    ap.add_argument("--cues", type=int, default=30)
    args = ap.parse_args()
    out = INTO / "screen_candidates_on_archived_stores.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    gen = torch.Generator().manual_seed(0)
    stores, skipped = [], 0
    with zipfile.ZipFile(args.archive) as z:
        names = sorted(n for n in z.namelist() if n.endswith("summaries.jsonl"))
        for name in names:
            lines = [json.loads(t) for t in z.read(name).decode("utf-8").splitlines() if t.strip()]
            vectors = [line["summary"] for line in lines if "summary" in line]
            if len(vectors) < args.least or len({len(v) for v in vectors}) != 1:
                skipped += 1
                continue
            rows = torch.tensor(vectors, dtype=torch.float64).to(torch.float32)
            n, width = rows.shape
            rows64 = rows.to(torch.float64)
            m0 = rows64[: max(2, int(n * 0.98))].mean(0)
            mean = rows64.mean(0)
            drift = float((mean - m0).norm())
            n0 = (rows64 - m0).norm(dim=1)
            raw_len = rows64.norm(dim=1)
            centred32 = (rows64 - m0).to(torch.float32)
            centred_len = centred32.to(torch.float64).norm(dim=1)
            able = torch.ones(n, dtype=torch.bool)
            able[n - args.recent:] = False
            alike = torch.nn.functional.cosine_similarity(rows64[:-1], rows64[1:], dim=1)
            counts = {"raw": [], "centred": []}
            missed = 0
            for q in range(args.cues):
                i = n - 1 - q % args.recent if q % 2 == 0 else int(torch.randint(n, (1,), generator=gen))
                cue = rows[i]
                c = cue.to(torch.float64) - mean
                cn, c32 = float(c.norm()), c.to(torch.float32)
                c32n = float(c32.to(torch.float64).norm())
                true = exact(rows64, mean, cue).masked_fill(~able, float("-inf"))
                best = int(true.argmax())
                for kind, matrix, length, shift in (("raw", rows, raw_len, mean), ("centred", centred32, centred_len, mean - m0)):
                    up, low = bounds((matrix @ c32).to(torch.float64), length, n0, drift, float(shift @ c), cn, c32n, width)
                    bar = low.masked_fill(~able, float("-inf")).max()      # over the rows the search may return
                    candidates = (able & (up >= bar)).nonzero().flatten()
                    counts[kind].append(int(candidates.numel()))
                    missed += int(best not in set(candidates.tolist()))
            stores.append({
                "store": name.split("/memories/")[0].split("2026-11-11_memory_test_lives/")[1],
                "summaries": n, "width": width, "eligible": int(able.sum()),
                "alike_one_to_the_next_median": float(alike.median()),
                "length_over_distance_from_the_mean_median": float((raw_len / n0.clamp_min(1e-300)).median()),
                "candidates_raw_median": statistics.median(counts["raw"]), "candidates_raw_most": max(counts["raw"]),
                "candidates_centred_median": statistics.median(counts["centred"]),
                "candidates_centred_most": max(counts["centred"]),
                "eligible_best_missed": missed,
            })
    stores.sort(key=lambda s: -s["summaries"])
    for s in stores[:25]:
        print(f"{s['store'][-34:]:34s} n={s['summaries']:5d} w={s['width']:4d} alike {s['alike_one_to_the_next_median']:.5f} "
              f"len/dist {s['length_over_distance_from_the_mean_median']:8.1f} | raw {s['candidates_raw_median']:6.0f} "
              f"(most {s['candidates_raw_most']}) | centred {s['candidates_centred_median']:5.0f} "
              f"(most {s['candidates_centred_most']}) of {s['eligible']} | missed {s['eligible_best_missed']}")
    share = lambda kind: statistics.median(s[f"candidates_{kind}_median"] / s["eligible"] for s in stores)
    summary = {"stores_read": len(stores), "stores_too_small_or_mixed": skipped,
               "share_of_eligible_rows_left_raw_median_over_stores": share("raw"),
               "share_of_eligible_rows_left_centred_median_over_stores": share("centred"),
               "eligible_best_missed_in_any_store": sum(s["eligible_best_missed"] for s in stores)}
    print(json.dumps(summary, indent=2))
    INTO.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"what": __doc__.splitlines()[0], "settings": vars(args), "torch": torch.__version__,
                               "summary": summary, "stores": stores}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
