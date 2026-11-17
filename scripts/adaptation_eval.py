"""Does reading change how the model sees what comes next?

The README's near-term goal -- structural change without a gradient step,
at inference, with behavioral consequence -- measured on a trained
checkpoint, no training. Wraps ``luthi.v2.adaptation_eval``; see its module
docstring for the A / B / C protocol (frozen / read-own / read-other).

Usage (two runs per checkpoint -- the second is the replication):
    python scripts/adaptation_eval.py --run-dir E:\\runs\\jepa_pilot\\<run> \\
        --docs 32 --read 4 --score 2
    python scripts/adaptation_eval.py --run-dir E:\\runs\\jepa_pilot\\<run> \\
        --docs 32 --read 4 --score 2 --doc-phase 1

Pre-registered reading (frozen 2026-10-19, before any production checkpoint
was evaluated; registry entry E1 in
docs/research/2026-07-15_falsification-preregistration.md, as amended the
same day before any result was seen):

  * Dead arms: summary.exact_null is True. Anything else = harness defect.
  * A run MEETS the criterion on one document set iff nmse B_minus_C has
    mean < 0 with t <= -2 over >= 32 documents AND target_eff_rank
    B_minus_C mean > -5% of its A mean (the improvement is not bought by the
    representation degenerating during reading).
  * The living-channel claim "reading produces document-specific change
    with behavioral consequence" is SUPPORTED for a run iff it meets the
    criterion on BOTH phase 0 and the disjoint phase 1. One set alone is a
    candidate, not a finding: t <= -2 occurs by chance with p ~ 0.027 per
    set, so over 20 checkpoints at least one chance hit is ~42% likely;
    requiring both sets brings that to ~1.5%. Generic drift (C_minus_A) is
    reported, never credited.

Writes ``adaptation_eval.json`` (phase 0) or ``adaptation_eval_phase<k>.json``
into the run dir (or --out); never modifies the checkpoint or any training
artifact.
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
    ap.add_argument("--doc-phase", type=int, default=0,
                    help="0 = the primary documents; 1 = a disjoint replication "
                         "set (E1 counts a checkpoint only if both agree)")
    # Opt-in switches (2026-11-16, exploratory). Each off by default; see
    # luthi/v2/adaptation_eval.py.
    ap.add_argument("--homeostatic-decay", type=float, default=None,
                    help="the pull toward home during reads (default: as loaded); 0 = off")
    ap.add_argument("--no-read-consolidation", action="store_true",
                    help="consolidation cannot fire during reads")
    ap.add_argument("--no-read-episode-writes", action="store_true",
                    help="no episode is written during reads (block stores frozen, "
                         "the layer's own episode write skipped)")
    ap.add_argument("--read-what", choices=("head", "tail"), default="head",
                    help="tail = a repeat read: B reads the windows it is scored on, "
                         "C another document's; needs --read equal to --score")
    ap.add_argument("--read-by-gradient", type=float, default=None, metavar="RATE",
                    help="replace the plastic read by SGD at this rate on the loss "
                         "of the windows read, plasticity frozen (a positive control)")
    ap.add_argument("--gradient-steps", type=int, default=1,
                    help="SGD steps per window, with --read-by-gradient")
    ap.add_argument("--gradient-read-what", choices=("trained", "living"),
                    default="trained",
                    help="living = the read by gradient steps the living weights "
                         "and nothing else, by hand, restored after (a measurement)")
    ap.add_argument("--read-by-error-map", type=float, default=None, metavar="RATE",
                    help="replace the plastic read by a fixed map of the prediction's "
                         "error onto each living weight, at this rate; no backward pass "
                         "(brief 8 of 2026-11-17; luthi/v2/error_map_read.py)")
    ap.add_argument("--error-map-kind", default="map",
                    choices=("map", "shuffled", "common",
                             "error", "error_turned", "error_shuffled", "bias"),
                    help="map, shuffled, common: fitted. error, error_turned, "
                         "error_shuffled, bias: nothing fitted (addendum 2 to brief 8)")
    ap.add_argument("--error-map-in-context", action="store_true",
                    help="each window is first read with every other way on, then the "
                         "rule's step is taken")
    ap.add_argument("--held-score", action="store_true",
                    help="also score B and C against arm A's targets over A's variance, "
                         "and report how far the read moved the targets")
    ap.add_argument("--error-map-fit-windows", type=int, default=512,
                    help="held-out windows the map is fitted on; none is in a document "
                         "of either document set of this layout")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.read_what == "tail" and args.read != args.score:
        ap.error("--read-what tail reads the scored windows, so --read must equal "
                 "--score (it sets how many leading windows are skipped)")

    from luthi.v2.adaptation_eval import adaptation_eval, holdout_documents
    from luthi.v2 import result_json
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    device = _device()
    run = rebuild_run(args.run_dir, device=device, ckpt_name=args.ckpt or None)
    windows = run.text_ds.holdout_window_tokens().to(device)
    docs = holdout_documents(windows, n_read=args.read, n_score=args.score,
                             n_docs=args.docs, phase=args.doc_phase)
    maps = None
    if args.read_by_error_map is not None and args.error_map_kind in (
            "error", "error_turned", "error_shuffled", "bias"):
        from luthi.v2.error_map_read import unfitted
        maps = unfitted(args.error_map_kind)
    elif args.read_by_error_map is not None:
        from luthi.v2.adaptation_eval import holdout_document_ranges
        from luthi.v2.error_map_read import fit_error_maps, windows_outside_documents
        # Fitted on windows that are in no document of this layout, in
        # either document set, so the fit and the read share nothing.
        taken = [r for phase in (0, 1) for r in holdout_document_ranges(
            int(windows.shape[0]), n_read=args.read, n_score=args.score,
            n_docs=args.docs, phase=phase)]
        fit_at = windows_outside_documents(int(windows.shape[0]), taken,
                                           args.error_map_fit_windows)
        maps = fit_error_maps(run.loss_module.eval(), windows[fit_at],
                              kind=args.error_map_kind)
        print(f"error map ({maps.kind}) fitted on {maps.fitted_on} windows; on "
              f"{maps.check['windows_held_back']} held back: "
              + " ".join(f"{c['cosine']:+.2f}" for c in maps.check["median_by_layer"]),
              flush=True)
    res = adaptation_eval(
        run.loss_module, docs, n_read=args.read, read_repeats=args.repeats,
        read_train_mode=args.read_train_mode, rate_scale=args.rate_scale,
        homeostatic_decay=args.homeostatic_decay,
        read_consolidation=not args.no_read_consolidation,
        read_episode_writes=not args.no_read_episode_writes,
        read_what=args.read_what,
        read_by_gradient=args.read_by_gradient,
        gradient_steps=args.gradient_steps,
        gradient_read_what=args.gradient_read_what,
        read_by_error_map=maps, error_map_rate=args.read_by_error_map,
        error_map_in_context=args.error_map_in_context,
        held_score=args.held_score,
    )
    res["run"] = {"arm": run.result["arm"], "seed": run.result["seed"],
                  "checkpoint": run.checkpoint.name, "global_step": run.global_step}
    res["documents"] = {"n_docs": args.docs, "n_read": args.read,
                        "n_score": args.score, "phase": args.doc_phase,
                        "holdout_windows": int(windows.shape[0])}
    s = res["summary"]
    print(f"arm={run.result['arm']} seed={run.result['seed']} "
          f"docs={s['n_docs']} phase={args.doc_phase} exact_null={s['exact_null']}")
    for metric in ("nmse", "l_pred", "target_eff_rank") + (
            ("held",) if args.held_score else ()):
        m = s[metric]
        print(f"  {metric:<16} A={m['A_mean']:.5f}  "
              f"B-A={m['B_minus_A']['mean']:+.5f} (t={m['B_minus_A']['t']:+.2f})  "
              f"C-A={m['C_minus_A']['mean']:+.5f} (t={m['C_minus_A']['t']:+.2f})  "
              f"B-C={m['B_minus_C']['mean']:+.5f} (t={m['B_minus_C']['t']:+.2f})")
    default_name = ("adaptation_eval.json" if args.doc_phase == 0
                    else f"adaptation_eval_phase{args.doc_phase}.json")
    out = Path(args.out) if args.out else Path(args.run_dir) / default_name
    res["reconstruction"] = reconstruction(run.result["config"])
    out.write_text(result_json.dumps(res), encoding="utf-8")
    print(f"wrote {out}")
    # This script ends on a restore of the model, which is queued on the
    # card and never read back. ``_device()`` registered a wait for the card
    # at exit; without it the process does not exit (see there).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
