"""The M4 STOP GATE, re-run against production checkpoints.

The gate exists in two wordings. Both are tested here, because they are not
the same claim.

  docs/ML_GLOSSARY.md:
    "if consolidation has no measurable effect on prediction quality
     post-replay, v2 has no architectural novelty over 'vanilla transformer +
     episode store' and should be abandoned."

  luthi/v2/consolidation.py docstring (closer to the original brief):
    "if consolidated layer's behavior on the episode's context is not
     measurably closer to the stored snapshot than a control without
     consolidation, v2 has no architectural novelty [...]"

History: the gate ran and passed 2026-05-09 at M4 scale. It has never been
re-run against production checkpoints. This script does that.

Three measurements per layer holding episodes, each against a no-consolidation
control:

  A. weight-space distance to the stored snapshots  (consolidation.py, literal)
  B. behavioural distance to the snapshots, evaluated on each episode's own
     stored input: ||W_now @ x - W_snap @ x||        (consolidation.py, intent)
  C. prediction error on each stored input pattern   (glossary)

"Measurable" needs a yardstick, and the honest one comes from the run itself:
`update_ema` is the layer's own EMA of ordinary per-step weight motion. If a
full consolidation pass over every stored episode moves the layer less than a
single ordinary training step does, consolidation is not a mechanism operating
on the substrate -- it is below the noise floor of normal learning. That
comparison is reported alongside the pre-registered relative-change test.

Read-only: checkpoints are copied aside before loading and never written.
"""
import argparse
import copy
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from luthi.v2.consolidation import consolidate_layer, consolidate_layer_attractor
from luthi.v2.living_layer_pc import PredictiveCodingLayer

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runs", "jepa_pilot")

# Pre-registered "measurable" threshold on relative change: 0.1%.
MEANINGFUL = 1e-3


def snapshots(layer):
    """Stored weight snapshots, dequantized if the store is int8."""
    n = int(layer.episode_count.item())
    out = []
    for i in range(n):
        if layer.episode_values.dtype == torch.int8:
            out.append(layer.episode_values[i].to(torch.float32)
                       * layer.episode_scales[i].float())
        else:
            out.append(layer.episode_values[i].float())
    return out


def measure(layer, snaps):
    """Return (weight_dist, behaviour_dist, pred_err) averaged over episodes."""
    n = int(layer.episode_count.item())
    w = layer.weight.float()
    p = layer.prediction.float()
    wd = bd = pe = 0.0
    with torch.no_grad():
        for i in range(n):
            x = layer.episode_inputs[i].float()
            s = snaps[i]
            wd += float((w - s).abs().mean())
            bd += float((x @ w.T - x @ s.T).abs().mean())
            pe += float((x - (x @ w.T) @ p).abs().mean())
    return wd / n, bd / n, pe / n


def layer_from_state(state, prefix):
    w = state[prefix + "weight"]
    out_f, in_f = w.shape
    n_ep = state[prefix + "episode_saliences"].numel()
    ctx_dim = state[prefix + "episode_contexts"].shape[1]
    layer = PredictiveCodingLayer(in_features=in_f, out_features=out_f,
                                  num_episodes=n_ep, context_dim=ctx_dim)
    own = layer.state_dict()
    loaded = {k[len(prefix):]: v for k, v in state.items()
              if k.startswith(prefix) and k[len(prefix):] in own}
    layer.load_state_dict(loaded, strict=False)
    return layer


def gate_one_run(run, rate_factor):
    ck_dir = os.path.join(BASE, run, "checkpoints")
    if not os.path.isdir(ck_dir):
        print(f"{run}: no checkpoints")
        return None
    name = sorted(n for n in os.listdir(ck_dir) if n.endswith(".pt"))[-1]
    tmp = os.path.join(BASE, "_m4_gate_tmp.pt")
    shutil.copy2(os.path.join(ck_dir, name), tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    state = ck.get("online_state_dict") or {}
    os.remove(tmp)

    print(f"\n=== {run}  ({name}, step {ck.get('global_step')}), "
          f"rate_factor {rate_factor}")
    # Only PC layers qualify: a module can carry `episode_count` without
    # being a consolidatable layer (e.g. the standalone episode_store).
    prefixes = sorted({k.rsplit(".", 1)[0] + "."
                       for k in state if k.endswith(".episode_count")})
    prefixes = [p for p in prefixes
                if p + "weight" in state and p + "episode_inputs" in state]
    results = []
    for prefix in prefixes:
        n_ep = int(state[prefix + "episode_count"].item())
        if n_ep == 0:
            print(f"  {prefix[:-1]}: 0 episodes -- nothing to replay, "
                  f"consolidation is a literal no-op here")
            continue
        base = layer_from_state(state, prefix)
        snaps = snapshots(base)
        dtype = state[prefix + "weight"].dtype
        step_motion = float(state[prefix + "update_ema"].float().abs().mean())

        control = copy.deepcopy(base)
        c_wd, c_bd, c_pe = measure(control, snaps)

        grad = copy.deepcopy(base)
        n_g = consolidate_layer(grad, consolidation_rate_factor=rate_factor)
        g_wd, g_bd, g_pe = measure(grad, snaps)
        g_moved = float((grad.weight.float() - base.weight.float()).abs().mean())

        att = copy.deepcopy(base)
        n_a = consolidate_layer_attractor(att, consolidation_rate_factor=rate_factor)
        a_wd, a_bd, a_pe = measure(att, snaps)
        a_moved = float((att.weight.float() - base.weight.float()).abs().mean())

        def rel(after, ctrl):
            return (after - ctrl) / max(abs(ctrl), 1e-30)

        print(f"  {prefix[:-1]}  ({n_ep} episodes, dtype {dtype}, "
              f"replayed grad={n_g} att={n_a})")
        print(f"     {'metric':<26} {'control':>13} {'gradient':>13} "
              f"{'attractor':>13} {'rel(grad)':>11} {'rel(att)':>11}")
        for label, c, g, a in (
            ("A weight dist to snapshot", c_wd, g_wd, a_wd),
            ("B behaviour dist on input", c_bd, g_bd, a_bd),
            ("C prediction error", c_pe, g_pe, a_pe),
        ):
            print(f"     {label:<26} {c:13.6e} {g:13.6e} {a:13.6e} "
                  f"{rel(g, c):+11.3e} {rel(a, c):+11.3e}")
        print(f"     weight moved by consolidation: gradient {g_moved:.4e}, "
              f"attractor {a_moved:.4e}")
        print(f"     one ordinary step moves it:    {step_motion:.4e} "
              f"(update_ema)")
        ratio_g = g_moved / max(step_motion, 1e-30)
        ratio_a = a_moved / max(step_motion, 1e-30)
        print(f"     full replay pass / one step:   gradient {ratio_g:.2f}x, "
              f"attractor {ratio_a:.2f}x")
        results.append(dict(
            run=run, layer=prefix[:-1], n_ep=n_ep,
            relA=max(abs(rel(g_wd, c_wd)), abs(rel(a_wd, c_wd))),
            relB=max(abs(rel(g_bd, c_bd)), abs(rel(a_bd, c_bd))),
            relC=max(abs(rel(g_pe, c_pe)), abs(rel(a_pe, c_pe))),
            ratio=max(ratio_g, ratio_a),
        ))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rate-factor", type=float, default=0.1)
    args = ap.parse_args()

    allr = []
    for run in args.runs:
        r = gate_one_run(run, args.rate_factor)
        if r:
            allr.extend(r)

    if not allr:
        print("\nGATE NOT EVALUABLE: no layer in any requested checkpoint "
              "holds episodes. Consolidation cannot affect prediction quality "
              "with nothing to replay -- which is itself the finding.")
        return 1

    print("\n=== VERDICT ===")
    print(f"pre-registered threshold: relative change > {MEANINGFUL:.0e}")
    print(f"{'run':<32} {'layer':<20} {'A wt':>10} {'B behav':>10} "
          f"{'C pred':>10} {'vs 1 step':>10}  gate")
    for r in allr:
        best = max(r["relA"], r["relB"], r["relC"])
        status = "PASS" if best > MEANINGFUL else "FAIL"
        print(f"{r['run'][:31]:<32} {r['layer']:<20} {r['relA']:10.3e} "
              f"{r['relB']:10.3e} {r['relC']:10.3e} {r['ratio']:9.2f}x  {status}")
    overall = max(max(r["relA"], r["relB"], r["relC"]) for r in allr)
    print(f"\noverall best relative effect {overall:.3e} -> gate "
          f"{'PASSES' if overall > MEANINGFUL else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
