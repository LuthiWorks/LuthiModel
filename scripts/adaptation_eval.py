"""Does reading change how the model sees what comes next?

The README's near-term goal -- structural change without a gradient step,
at inference, with behavioral consequence -- measured on a trained
checkpoint, no training. Wraps ``luthi.v2.adaptation_eval``; see its module
docstring for the A / B / C protocol (frozen / read-own / read-other).

Usage:
    python scripts/adaptation_eval.py --run-dir E:\\runs\\jepa_pilot\\<run> \\
        --docs 32 --read 4 --score 2

Pre-registered reading (frozen 2026-10-19, before any production checkpoint
was evaluated; registry entry E1 in
docs/research/2026-07-15_falsification-preregistration.md):

  * Dead arms: summary.exact_null is True. Anything else = harness defect.
  * The living-channel claim "reading produces document-specific change with
    behavioral consequence" is SUPPORTED for a run iff nmse B_minus_C has
    mean < 0 with |t| >= 2 over >= 32 documents AND target_eff_rank
    B_minus_C mean > -5% of its A mean (the improvement is not bought by the
    representation degenerating during reading). Otherwise NOT SUPPORTED for
    that run. Generic drift (C_minus_A) is reported, never credited.

Writes ``adaptation_eval.json`` next to the checkpoint's run dir (or --out);
never modifies the checkpoint or any training artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--docs", type=int, default=32)
    ap.add_argument("--read", type=int, default=4, help="windows read per document")
    ap.add_argument("--score", type=int, default=2, help="windows scored per document")
    ap.add_argument("--repeats", type=int, default=1, help="passes over the read windows")
    ap.add_argument("--read-train-mode", action="store_true",
                    help="read in train mode (adds the top-down sweep)")
    ap.add_argument("--rate-scale", type=float, default=None,
                    help="living rate_scale during reads (default: as loaded, 1.0)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from luthi.v2.adaptation_eval import adaptation_eval, holdout_documents
    from luthi.v2.run_rebuild import rebuild_run
    from scripts.jepa_pilot_driver import _device

    device = _device()
    run = rebuild_run(args.run_dir, device=device, ckpt_name=args.ckpt or None)
    windows = run.text_ds.holdout_window_tokens().to(device)
    docs = holdout_documents(windows, n_read=args.read, n_score=args.score,
                             n_docs=args.docs)
    res = adaptation_eval(
        run.loss_module, docs, n_read=args.read, read_repeats=args.repeats,
        read_train_mode=args.read_train_mode, rate_scale=args.rate_scale,
    )
    res["run"] = {"arm": run.result["arm"], "seed": run.result["seed"],
                  "checkpoint": run.checkpoint.name, "global_step": run.global_step}
    s = res["summary"]
    print(f"arm={run.result['arm']} seed={run.result['seed']} "
          f"docs={s['n_docs']} exact_null={s['exact_null']}")
    for metric in ("nmse", "l_pred", "target_eff_rank"):
        m = s[metric]
        print(f"  {metric:<16} A={m['A_mean']:.5f}  "
              f"B-A={m['B_minus_A']['mean']:+.5f} (t={m['B_minus_A']['t']:+.2f})  "
              f"C-A={m['C_minus_A']['mean']:+.5f} (t={m['C_minus_A']['t']:+.2f})  "
              f"B-C={m['B_minus_C']['mean']:+.5f} (t={m['B_minus_C']['t']:+.2f})")
    out = Path(args.out) if args.out else Path(args.run_dir) / "adaptation_eval.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
