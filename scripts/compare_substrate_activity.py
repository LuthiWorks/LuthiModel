"""Compare PC substrate activity across checkpoints.

Answers external-review round-3 item 3: is `update_ema` still ~5e-9 under the
fixed JEPA objective? If it recovers, the drive-normalization patch (1.1) was
treating a symptom of the collapsed representation. If it stays dead, the
substrate is quiescent for reasons independent of the objective.

Read-only.
"""
import argparse
import os
import sys

import torch

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "jepa_pilot")


def latest_ckpt(run: str) -> str | None:
    d = os.path.join(BASE, run, "checkpoints")
    if not os.path.isdir(d):
        return None
    names = sorted(n for n in os.listdir(d) if n.endswith(".pt"))
    return os.path.join(d, names[-1]) if names else None


def summarize(run: str) -> None:
    path = latest_ckpt(run)
    if path is None:
        print(f"{run}: no checkpoint")
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("online_state_dict") or {}
    print(f"\n=== {run}  ({os.path.basename(path)}, step {ck.get('global_step')})")
    prefixes = sorted({k.rsplit(".", 1)[0] + "."
                       for k in state if k.endswith(".update_ema")})
    hdr = (f"  {'layer':<34} {'|update_ema|':>12} {'|w|':>11} "
           f"{'ratio':>9} {'prec med':>9} {'prec p5':>9} {'prec p95':>9} {'eps':>5}")
    print(hdr)
    for p in prefixes:
        ue = state[p + "update_ema"].float()
        w = state[p + "weight"].float()
        prec = state.get(p + "precision")
        ue_m = float(ue.abs().mean())
        w_m = float(w.abs().mean())
        n_ep = int(state[p + "episode_count"].item()) if p + "episode_count" in state else -1
        if prec is not None:
            prec = prec.float().flatten()
            pm = float(prec.median())
            p5 = float(torch.quantile(prec, 0.05))
            p95 = float(torch.quantile(prec, 0.95))
        else:
            pm = p5 = p95 = float("nan")
        print(f"  {p[:-1]:<34} {ue_m:12.4e} {w_m:11.4e} "
              f"{ue_m / max(w_m, 1e-30):9.2e} {pm:9.4f} {p5:9.4f} {p95:9.4f} {n_ep:5d}")
    print("  (last column = episodes stored)")

    # Surprise-drive readings, if this run had them. `duty` is the whole point:
    # it separates "quiet because nothing is new" from "quiet because broken".
    drive_prefixes = [p for p in prefixes if p + "drive_calls" in state]
    if drive_prefixes:
        print(f"  {'layer':<34} {'duty':>8} {'fires':>8} {'calls':>8} "
              f"{'gain':>9} {'ref':>10} {'dev':>10}")
        for p in drive_prefixes:
            calls = float(state[p + "drive_calls"].item())
            fires = float(state[p + "drive_fire_count"].item())
            # Duty is over post-warmup calls; warmup holds gain at 1.0 for
            # bit-identity with raw and is not counted as firing.
            warm = 200.0
            duty = fires / max(calls - warm, 1.0)
            print(f"  {p[:-1]:<34} {duty:8.4f} {fires:8.0f} {calls:8.0f} "
                  f"{float(state[p + 'drive_gain'].item()):9.4f} "
                  f"{float(state[p + 'drive_ref'].item()):10.4e} "
                  f"{float(state[p + 'drive_dev'].item()):10.4e}")
        print("  (duty over post-warmup calls; warmup assumed 200)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for r in args.runs:
        summarize(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
