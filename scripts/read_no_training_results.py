"""Read the collected E1, E2 and E4 results against their registered rules.

``scripts/run_no_training_experiments.py`` measures and collects. This reads
what it collected (``runs_meta/living_channel_e1_e2_e4``) and applies the
rules frozen in ``docs/research/2026-07-15_falsification-preregistration.md``
(E1, E2, E4 and amendments 1 and 4). It measures nothing and needs no card.

    python scripts/read_no_training_results.py
    python scripts/read_no_training_results.py --json

What counts, as registered:

- Admissible: completed runs only. A run stopped early is reported and never
  counted.
- E1: a checkpoint meets the criterion when ``nmse.B_minus_C`` has mean < 0
  with t <= -2 and the effective-rank guard holds, on the primary document
  set AND on the replication set. One set alone is a candidate. Every dead
  control must be an exact null or no living read counts.
- E2, for the 768x8 family's living blocks: top_share >= 0.9 and
  out_align >= 0.9; training-batch stationarity >= 0.99; block store
  recall_duty >= 0.9 and blend_share > 0.05. Any miss is a finding against
  the diagnosis.
- E4: margin m = 3 * sqrt(p_emb * (1 - p_emb) / n_pairs) from the embedding
  row. Refuted if the same block exceeds p_emb + m at the same offset (+1 or
  -1) in 2 or more of 3 seeds; confirmed otherwise.

The reads at the taper's floor are not registered (the registration says
"rate_scale as loaded"). They are reported beside the registered ones.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTED = REPO_ROOT / "runs_meta" / "living_channel_e1_e2_e4"
E1_SETS = {"primary": "adaptation_eval.json", "replication": "adaptation_eval_phase1.json"}
E1_FLOOR = "adaptation_eval_taper_floor.json"
E4_FILES = {
    "+1": "per_block_probe_offset+1.json", "-1": "per_block_probe_offset-1.json",
    "0": "per_block_probe_offset0.json", "causal": "per_block_probe_causal.json",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# The 768x8 family's runs that were stopped before they finished
# (docs/research/2026-08-11_768x8-family-spec.md; the terminal instance's
# report of 2026-11-13: steps 9,100 and 9,300 of 54,077).
STOPPED_EARLY = {"probe_768_visreg_768d_seed46", "probe_768_visreg_768d_seed95"}


def read_e1(run_dir: Path) -> dict | None:
    sets = {name: _load(run_dir / f) for name, f in E1_SETS.items()}
    if not all(sets.values()):
        return None
    out: dict = {"sets": {}}
    for name, result in sets.items():
        s = result["summary"]
        bc, rank = s["nmse"]["B_minus_C"], s["target_eff_rank"]
        t = bc["t"]
        guard = rank["B_minus_C"]["mean"] > -0.05 * rank["A_mean"]
        out["sets"][name] = {
            "exact_null": s["exact_null"],
            "nmse_A": s["nmse"]["A_mean"],
            "any_change": s["nmse"]["B_minus_A"]["mean"],
            "generic": s["nmse"]["C_minus_A"]["mean"],
            "specific": bc["mean"], "specific_t": t,
            "meets": bool(t is not None and t == t and bc["mean"] < 0 and t <= -2 and guard),
        }
    met = [v["meets"] for v in out["sets"].values()]
    out["verdict"] = "meets" if all(met) else "candidate" if any(met) else "does not meet"
    out["exact_null"] = all(v["exact_null"] for v in out["sets"].values())
    floor = _load(run_dir / E1_FLOOR)
    if floor:
        full = {d["doc"]: d["B"]["nmse"] - d["A"]["nmse"] for d in sets["primary"]["per_doc"]}
        low = {d["doc"]: d["B"]["nmse"] - d["A"]["nmse"] for d in floor["per_doc"]}
        ratios = sorted(abs(low[k]) / abs(full[k]) for k in full if full[k] != 0)
        s = floor["summary"]["nmse"]
        out["at_taper_floor"] = {
            "rate_scale": floor["settings"]["rate_scale"],
            "any_change": s["B_minus_A"]["mean"], "specific": s["B_minus_C"]["mean"],
            "specific_t": s["B_minus_C"]["t"],
            # 0.2 if the change came from the write, 1 if it came from elsewhere.
            "median_change_relative_to_full_rate": ratios[len(ratios) // 2] if ratios else None,
        }
    return out


def read_e2(run_dir: Path) -> dict | None:
    result = _load(run_dir / "living_write_probe.json")
    if not result:
        return None
    blocks, misses = [], []
    spectrum, station = result["living_change_spectrum"], result["batch_mean_stationarity"]
    diversity, store = result["write_diversity"], result.get("block_output_store", {})
    for name, s in spectrum.items():
        block = name.split(".")[1]
        st = store.get(f"blocks.{block}", {})
        row = {
            "block": int(block), "top_share": s["top_share"], "out_align": s["out_align"],
            "rel_norm": s["rel_norm"],
            "stationarity": station["train_batches"][name],
            "single_document_write_cosine": diversity["single_sequences"][name],
            "recall_duty": st.get("recall_duty"), "blend_share": st.get("last_blend_share"),
        }
        def at_least(v, bar):
            return v is not None and v >= bar
        if not (at_least(row["top_share"], 0.9) and at_least(row["out_align"], 0.9)):
            misses.append(f"block {block}: spectrum")
        if not at_least(row["stationarity"], 0.99):
            misses.append(f"block {block}: stationarity")
        if st and not (at_least(row["recall_duty"], 0.9) and (row["blend_share"] or 0) > 0.05):
            misses.append(f"block {block}: store")
        blocks.append(row)
    return {"global_step": result["global_step"], "blocks": blocks, "misses": misses}


def read_e4(run_dir: Path) -> dict | None:
    out = {}
    for offset, f in E4_FILES.items():
        result = _load(run_dir / f)
        if not result:
            return None
        rows = result["blocks"]
        emb = next(r for r in rows if r["block"] == "emb")
        p, n = emb["probe_top1"], emb["n_pairs"]
        margin = 3 * math.sqrt(p * (1 - p) / n)
        out[offset] = {
            "embedding": p, "margin": margin, "fit_device": result["settings"].get("fit_device"),
            "blocks": {r["block"]: r["probe_top1"] for r in rows if r["block"] != "emb"},
            "exceed": [r["block"] for r in rows if r["block"] != "emb" and r["probe_top1"] > p + margin],
        }
    return out


def read_all(collected: Path = COLLECTED) -> dict:
    runs = {}
    for run_dir in sorted(p for p in collected.iterdir() if p.is_dir()):
        runs[run_dir.name] = {
            "admissible": run_dir.name not in STOPPED_EARLY,
            "dead": "dead" in run_dir.name.split("_"),
            "e1": read_e1(run_dir), "e2": read_e2(run_dir), "e4": read_e4(run_dir),
        }
    dead = {k: v for k, v in runs.items() if v["dead"] and v["e1"]}
    living = {k: v for k, v in runs.items() if not v["dead"] and v["e1"]}
    counted = {k: v for k, v in living.items() if v["admissible"]}
    e4 = {k: v["e4"] for k, v in runs.items() if v["e4"]}
    exceed_counts: dict = {}
    for name, by_offset in e4.items():
        for offset in ("+1", "-1"):
            for block in by_offset[offset]["exceed"]:
                exceed_counts.setdefault((offset, block), []).append(name)
    refuted = {f"{o} {b}": names for (o, b), names in exceed_counts.items() if len(names) >= 2}
    verdicts = {
        "e1_controls_exact_null": bool(dead) and all(v["e1"]["exact_null"] for v in dead.values()),
        "e1_checkpoints_counted": len(counted),
        "e1_meet": [k for k, v in counted.items() if v["e1"]["verdict"] == "meets"],
        "e1_candidates": [k for k, v in counted.items() if v["e1"]["verdict"] == "candidate"],
        "e1_reported_not_counted": [k for k in living if k not in counted],
        "e2_768_family_misses": {
            k: v["e2"]["misses"] for k, v in runs.items()
            if v["e2"] and k.startswith("probe_768") and v["admissible"]
        },
        "e4": "refuted" if refuted else "confirmed",
        "e4_same_block_exceeds_in_two_seeds": refuted,
        "e4_seeds_read": len(e4), "e4_seeds_admissible": sum(runs[k]["admissible"] for k in e4),
    }
    return {"runs": runs, "verdicts": verdicts}


def _fmt(v, spec=".2f") -> str:
    return "  -  " if v is None else format(v, spec)


def print_report(report: dict) -> None:
    runs, v = report["runs"], report["verdicts"]
    print("E1: does reading change what follows, for the document read?")
    print(f"  dead controls exact null: {v['e1_controls_exact_null']}")
    for name, run in runs.items():
        e1 = run["e1"]
        if not e1 or run["dead"]:
            continue
        mark = "" if run["admissible"] else "  [stopped early: reported, not counted]"
        print(f"  {name}{mark}")
        for label, s in e1["sets"].items():
            print(f"    {label:11s} nmse {s['nmse_A']:.4f}  any {s['any_change']:+.1e}  "
                  f"generic {s['generic']:+.1e}  specific {s['specific']:+.1e} "
                  f"(t {_fmt(s['specific_t'], '+.1f')})")
        if "at_taper_floor" in e1:
            f = e1["at_taper_floor"]
            print(f"    at rate {f['rate_scale']}: any {f['any_change']:+.1e}  specific "
                  f"{f['specific']:+.1e} (t {_fmt(f['specific_t'], '+.1f')})  per document, "
                  f"{_fmt(f['median_change_relative_to_full_rate'])} of the full-rate change")
        print(f"    -> {e1['verdict']}")
    print(f"  counted {v['e1_checkpoints_counted']}; meet {len(v['e1_meet'])}; "
          f"candidates {len(v['e1_candidates'])}")

    print("\nE2: what the living write is")
    for name, run in runs.items():
        e2 = run["e2"]
        if not e2:
            continue
        mark = "" if run["admissible"] else "  [stopped early]"
        print(f"  {name}  step {e2['global_step']}{mark}")
        for key, spec in (("top_share", ".2f"), ("out_align", ".2f"), ("rel_norm", ".0e"),
                          ("stationarity", ".4f"), ("single_document_write_cosine", "+.2f"),
                          ("recall_duty", ".2f"), ("blend_share", ".2f")):
            print(f"    {key:28s}", " ".join(_fmt(b[key], spec) for b in e2["blocks"]))
        print(f"    misses: {', '.join(e2['misses']) or 'none'}")

    print("\nE4: does any block beat the embedding row?")
    for name, run in runs.items():
        if not run["e4"]:
            continue
        mark = "" if run["admissible"] else "  [stopped early]"
        print(f"  {name}{mark}")
        for offset, r in run["e4"].items():
            best = max(r["blocks"], key=r["blocks"].get)
            print(f"    {offset:6s} embedding {r['embedding']:.4f}  needs > "
                  f"{r['embedding'] + r['margin']:.4f}  best block {best} "
                  f"{r['blocks'][best]:.4f}  exceed: {', '.join(r['exceed']) or 'none'}")
    print(f"  -> {v['e4']} ({v['e4_seeds_read']} seeds read, "
          f"{v['e4_seeds_admissible']} admissible)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--collected", default=str(COLLECTED))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    report = read_all(Path(args.collected))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
