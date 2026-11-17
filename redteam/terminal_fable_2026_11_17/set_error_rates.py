"""Set the error read's three rates from size (addenda to brief 8).

The "error" rule's step is not in a gradient's units. So its rates are set
so that one window's step is the same share of the weight as M2's three
rates give:

- M2's share: the median step, by living layer, that
  ``e1_read64_living_gradient_<rate>.json`` recorded at each of its rates.
- The rule's share at rate 1: the median, over ``--windows`` held-out windows
  that are in no document of the layout, of |e| |x| / |W|, by layer.
- A rate: the median over layers of (M2's share / the rule's share at rate
  1). One rate serves every layer, so layers do not all match; what each
  step then is, by layer, is recorded by the read itself.

Nothing is tuned and nothing is moved. Writes ``error_rates.json`` beside
the run's results.

    python redteam/terminal_fable_2026_11_17/set_error_rates.py --run <run name> --docs 48
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
INTO = REPO_ROOT / "runs_meta" / "living_gradient_2026_11_17"
M2_RATES = ("1e-5", "1e-4", "1e-3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--read", type=int, default=64)
    ap.add_argument("--score", type=int, default=8)
    ap.add_argument("--windows", type=int, default=128)
    args = ap.parse_args()

    from luthi.v2 import result_json
    from luthi.v2.adaptation_eval import holdout_document_ranges
    from luthi.v2.error_map_read import step_share_at_rate_one, windows_outside_documents
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    out = INTO / args.run / "error_rates.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    m2 = {}
    for rate in M2_RATES:
        path = INTO / args.run / f"e1_read64_living_gradient_{rate}.json"
        shares = json.loads(path.read_text(encoding="utf-8"))["settings"]["living_step_share_of_weight"]
        m2[rate] = [s["median"] for s in shares]

    device = _device()
    run = rebuild_run(REPO_ROOT / "runs" / "jepa_pilot" / args.run, device=device)
    loss_module = run.loss_module.eval()
    windows = run.text_ds.holdout_window_tokens().to(device)
    taken = [r for phase in (0, 1) for r in holdout_document_ranges(
        int(windows.shape[0]), n_read=args.read, n_score=args.score, n_docs=args.docs, phase=phase)]
    at = windows_outside_documents(int(windows.shape[0]), taken, args.windows)
    unit = step_share_at_rate_one(loss_module, windows[at])
    rates = {rate: statistics.median(m / u for m, u in zip(m2[rate], unit)) for rate in M2_RATES}
    report = {
        "what": "the error read's rates, set so a step is the share of the weight M2's is",
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "m2_median_step_share_by_layer": m2,
        "error_step_share_at_rate_one_by_layer": unit,
        "windows": len(at),
        "error_rate_for_m2_rate": rates,
        "error_step_share_by_layer_at_those_rates": {
            rate: [rates[rate] * u for u in unit] for rate in M2_RATES},
        "reconstruction": reconstruction(run.result["config"]),
    }
    out.write_text(result_json.dumps(report), encoding="utf-8")
    print("M2 share by layer at 1e-5:", " ".join(f"{x:.2e}" for x in m2["1e-5"]))
    print("error share at rate 1:   ", " ".join(f"{x:.2e}" for x in unit))
    print("rates:", {k: f"{v:.4g}" for k, v in rates.items()})
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
