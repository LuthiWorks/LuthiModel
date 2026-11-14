"""Review probe (Opus 5.5, 2026-11-14): read the E1, E2 and E4 results again,
from the registry's text alone, without scripts/read_no_training_results.py.

Run from the LuthiModel root:

    python redteam/opus_review_2026_11_14/probe_read_e1_e2_e4.py

It recomputes E1's B minus C from the per-document values (not the summary),
applies amendment 1, E2's four predictions and amendment 4's margin, and
prints one line per result. On 2026-11-14 it agreed with the first read on
every verdict: E1 controls exact (3 of 3, both sets), no checkpoint meets the
criterion or is a candidate, t from -0.84 to +1.51; E2 twelve misses in seed
97; E4 no block exceeds the embedding row at +1 or -1 in any seed.

It also prints, per July run, the living displacement of each block and the
read's mean relative change, for the pull-toward-home arithmetic in the review.
"""

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "runs_meta" / "living_channel_e1_e2_e4"


def _t(xs):
    n, m = len(xs), statistics.fmean(xs)
    se = statistics.stdev(xs) / math.sqrt(n) if n > 1 else 0.0
    return m, (m / se if se > 0 else float("nan"))


def e1():
    print("== E1 (amendment 1: the criterion on phase 0 and on phase 1) ==")
    for run in sorted(ROOT.iterdir()):
        hits = []
        for name in ("adaptation_eval.json", "adaptation_eval_phase1.json", "adaptation_eval_taper_floor.json"):
            path = run / name
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            docs = d["per_doc"]
            m, t = _t([x["B"]["nmse"] - x["C"]["nmse"] for x in docs])
            rank_bc = statistics.fmean(x["B"]["target_eff_rank"] - x["C"]["target_eff_rank"] for x in docs)
            rank_a = statistics.fmean(x["A"]["target_eff_rank"] for x in docs)
            meets = m < 0 and t <= -2 and rank_bc > -0.05 * rank_a
            if name != "adaptation_eval_taper_floor.json":
                hits.append(meets)
            print(f"  {run.name:36s} {name:34s} exact_null={d['summary']['exact_null']!s:5s} "
                  f"B-C {m:+.2e} t {t:+.2f} meets {meets}")
        if hits:
            print(f"  -> {run.name}: {'MEETS' if all(hits) else 'candidate' if any(hits) else 'no'}")


def e2():
    print("== E2 (every living block of the 768 family) ==")
    for run in sorted(ROOT.glob("probe_768*")):
        d = json.loads((run / "living_write_probe.json").read_text())
        misses = 0
        for blk, sp in d["living_change_spectrum"].items():
            st = d["batch_mean_stationarity"]["train_batches"][blk]
            store = d["block_output_store"][blk.replace(".living_ffn", "")]
            misses += not (sp["top_share"] >= 0.9 and sp["out_align"] >= 0.9)
            misses += not (st >= 0.99)
            misses += not (store["recall_duty"] >= 0.9 and store["last_blend_share"] > 0.05)
            print(f"  {run.name} {blk}: top {sp['top_share']:.3f} align {sp['out_align']:.3f} "
                  f"rel {sp['rel_norm']:.1e} stationarity {st:.4f} duty {store['recall_duty']:.2f} "
                  f"blend {store['last_blend_share']:.3f}")
        print(f"  -> {run.name}: {misses} misses")


def e4():
    print("== E4 (amendment 4) ==")
    exceeded = {}
    for run in sorted(ROOT.glob("probe_768*")):
        for off in ("+1", "-1"):
            d = json.loads((run / f"per_block_probe_offset{off}.json").read_text())
            emb = next(b for b in d["blocks"] if b["block"] == "emb")
            p, n = emb["probe_top1"], emb["n_pairs"]
            margin = 3 * math.sqrt(p * (1 - p) / n)
            blocks = [b for b in d["blocks"] if b["block"] != "emb"]
            best = max(blocks, key=lambda b: b["probe_top1"])
            over = [b["block"] for b in blocks if b["probe_top1"] > p + margin]
            for b in over:
                exceeded.setdefault((b, off), []).append(run.name)
            print(f"  {run.name} offset {off}: embedding {p:.4f} margin {margin:.4f} "
                  f"best block {best['block']} {best['probe_top1']:.4f} exceeds {over}")
    print("  refuted cells:", {k: v for k, v in exceeded.items() if len(v) >= 2})


def pull_toward_home():
    print("== The read's change against the living displacement (July runs) ==")
    for run in sorted(ROOT.glob("living_v5*")):
        e2r = json.loads((run / "living_write_probe.json").read_text())
        rel = [v["rel_norm"] for _, v in sorted(e2r["living_change_spectrum"].items())]
        change = statistics.fmean(
            statistics.fmean(abs(x["B"]["nmse"] - x["A"]["nmse"]) / x["A"]["nmse"]
                             for x in json.loads((run / f).read_text())["per_doc"])
            for f in ("adaptation_eval.json", "adaptation_eval_phase1.json"))
        print(f"  {run.name:36s} displacement by block {[f'{r:.1e}' for r in rel]}  "
              f"read's mean relative change {change:.1e}")


if __name__ == "__main__":
    e1(); e2(); e4(); pull_toward_home()
