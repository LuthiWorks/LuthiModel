"""Check our runs against the JEPA-in-language failure cascade.

arXiv:2607.23531 ("The JEPA Paradox in Language", 2026-07-26) argues that
squared-error latent prediction assumes CONDITIONAL CONCENTRATION -- one
plausible continuation -- and that masked text has many. It documents a
consistent failure cascade over five seeds, comparing I-JEPA (images, healthy)
against T-JEPA (text, degrading):

    1. effective-rank degeneration
    2. cosine collapse
    3. elevated target variance
    4. train/val instability
    5. MI saturation
    6. degraded downstream performance

We train a JEPA on text. This checks each symptom we can measure against the
checkpoints and logs we already own. Eval-only; no training, writes nothing.

THE DISCRIMINATING QUESTION is not "do we show the cascade" -- we plainly show
some of it -- but WHERE:

  * If the cascade appears in the depth-4 families too, it is a property of
    running JEPA on text and we have been living with it since the JEPA move.
  * If it appears ONLY in the depth-8 muPC-on runs, it is our configuration,
    and the paper describes a different animal that happens to share symptoms.

MI saturation is not instrumented and is not estimated here. Five of six.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "jepa_pilot")


def series(recs, path):
    """Pull a nested metric across records; path like ('light','online_std_p5')."""
    out = []
    for r in recs:
        cur = r
        for k in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        if isinstance(cur, (int, float)):
            out.append(float(cur))
    return out


def block_metric(recs, key):
    out = []
    for r in recs:
        for b in (r.get("substrate_blocks") or []):
            v = b.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
    return out


def summarize(run: str) -> dict | None:
    d = os.path.join(BASE, run)
    log = os.path.join(d, "training_log.jsonl")
    res = os.path.join(d, "pilot_result.json")
    if not (os.path.exists(log) and os.path.exists(res)):
        return None
    recs = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
    p = json.load(open(res))
    h = p.get("heldout", {})
    pr = p.get("probe", {})
    fl = p.get("probe_shuffled_floor", {})

    rank = block_metric(recs, "effective_rank")
    std5 = series(recs, ("light", "online_std_p5"))
    std50 = series(recs, ("light", "online_std_p50"))
    cos_c = series(recs, ("light", "predictor_cosine_centered_mean"))
    triv = series(recs, ("light", "predictor_trivial_cosine_mean"))
    loss = [r["loss"] for r in recs if "loss" in r]

    # train/val instability: coefficient of variation of the loss over the back
    # half, after any startup transient.
    tail = loss[len(loss) // 2:] if len(loss) > 4 else loss
    cv = (st.pstdev(tail) / st.mean(tail)) if tail and st.mean(tail) else float("nan")

    floor = fl.get("top1") or 0.0
    return {
        "run": run,
        "n_blocks": len(recs[-1].get("substrate_blocks") or []) if recs else 0,
        "rank_last": rank[-1] if rank else float("nan"),
        "rank_med": st.median(rank) if rank else float("nan"),
        "std_p50": st.median(std50) if std50 else float("nan"),
        "std_p5": st.median(std5) if std5 else float("nan"),
        "cos_centered": st.median(cos_c) if cos_c else float("nan"),
        "cos_trivial": st.median(triv) if triv else float("nan"),
        "loss_cv": cv,
        "nmse": h.get("nmse_mean", float("nan")),
        "lift": (pr.get("top1", 0.0) / floor) if floor else float("nan"),
        "outcome": p.get("outcome", "?"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*")
    args = ap.parse_args()

    runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BASE, args.pattern))
                  if os.path.isdir(p) and not os.path.basename(p).startswith("ledger_"))
    rows = [r for r in (summarize(x) for x in runs) if r]
    if not rows:
        print("no runs matched")
        return 1

    hdr = (f"{'run':<40}{'blk':>4}{'rank':>8}{'std50':>8}{'cos_c':>8}"
           f"{'trivial':>9}{'lossCV':>8}{'NMSE':>8}{'lift':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["n_blocks"], x["run"])):
        print(f"{r['run'][:39]:<40}{r['n_blocks']:>4}{r['rank_med']:>8.1f}"
              f"{r['std_p50']:>8.3f}{r['cos_centered']:>8.3f}"
              f"{r['cos_trivial']:>9.3f}{r['loss_cv']:>8.2f}"
              f"{r['nmse']:>8.3f}{r['lift']:>6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
