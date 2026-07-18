"""Stage-1 verdict under the blind-amended criteria (2026-07-16).

Reloads every completed pilot run's FINAL checkpoint, recomputes the
held-out eval with the amended primary metric (NMSE — committed blind at
0fcc92a before any run's NMSE existed), and applies the pre-registered
verdict rule verbatim:

  per axis (NMSE, probe top-1): |mean difference| > pooled sigma = win
  for the better side; within = tie. KF2-strong SURVIVES only if the
  living arm wins >= 1 axis and loses none; the KILL fires if the living
  arm loses any axis or ties both (ON the curve = no advantage).

Probe numbers are taken from each run's pilot_result.json (computed at
run end on the same holdout). NMSE is recomputed here from checkpoints
because pass-2 runs predate the metric.

Writes runs/jepa_pilot/verdict.json and prints the reading.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_ROOT = REPO_ROOT / "runs" / "jepa_pilot"

from scripts.jepa_pilot_driver import _DeviceLoader, _device  # noqa: E402


def _nmse_for_run(run_dir: Path, result: dict) -> float:
    from luthi.living_extra_state import apply_living_extra_state
    from luthi.v2.eval_heldout import heldout_latent_prediction
    from luthi.v2.jepa_loss import JEPALoss
    from luthi.v2.multimodal_data import (
        MultimodalDataLoaderImpl,
        TextDataset,
        TextDatasetConfig,
    )
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    cfg = result["config"]
    device = _device()

    text_ds = TextDataset(TextDatasetConfig(
        source_paths=[cfg["data_dir"]],
        tokenizer_path=REPO_ROOT / "corpus_build" / "tokenizer_32k.json",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        base_seed=result["seed"],
        holdout_fraction=cfg["holdout_fraction"],
    ))
    loader = _DeviceLoader(MultimodalDataLoaderImpl(text=text_ds), device)

    from scripts.jepa_pilot_driver import ARM_CONFIGS
    arm_cfg = ARM_CONFIGS[result["arm"]]
    model = MultimodalPredictiveCodingLM(
        vocab_size=text_ds.vocab_size(),
        d_model=result["d_model"],
        n_blocks=cfg["n_blocks"],
        n_heads=4,
        ffn_expansion=1,
        max_seq_len=cfg["seq_len"],
        # Rebuild the arm EXACTLY as trained (ARM_CONFIGS is the single
        # source of truth) -- the recall gate is live at eval, so a
        # mismatched threshold would evaluate a different model.
        backward_pass_enabled=arm_cfg.get("backward_pass_enabled", False),
        **{k: v for k, v in arm_cfg.items()
           if k != "backward_pass_enabled"},
    ).to(device)
    loss_module = JEPALoss(online_encoder=model).to(device)

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(state["online_state_dict"])
    apply_living_extra_state(
        model, state.get("living_extra_state"),
        source=f"verdict eval {ckpts[-1]}",
    )
    loss_module.predictor.load_state_dict(state["predictor_state_dict"])
    loss_module.projection_heads.load_state_dict(
        state["projection_heads_state_dict"],
    )

    # Determinism guards (2026-07-18, found comparing the 07-17 03:05 and
    # 07-18 09:20 reads): the first ~2 evals in a fresh process on the
    # DML backend came out ~2% high while every later eval reproduced
    # bit-for-bit -- an order-dependent warm-up artifact in the backend's
    # numerics, not a model property. Pin RNG and run one DISCARDED
    # warm-up eval so every measured number comes from a warm device
    # regardless of call order. No verdict flipped (the wobble is far
    # inside every margin), but the primary metric must not depend on
    # evaluation order.
    torch.manual_seed(0)
    warmup = heldout_latent_prediction(
        loss_module, loader.holdout_batches("text", 8), "text",
        max_batches=2,
    )
    del warmup
    r = heldout_latent_prediction(
        loss_module, loader.holdout_batches("text", 8), "text",
        max_batches=50,
    )
    return float(r["nmse_mean"])


def _axis(living: list[float], dead: list[float], lower_is_better: bool):
    lm, dm = statistics.mean(living), statistics.mean(dead)
    pooled = statistics.mean([statistics.stdev(living), statistics.stdev(dead)])
    diff = (dm - lm) if lower_is_better else (lm - dm)  # >0 = living better
    if abs(diff) <= pooled:
        return "tie", lm, dm, pooled
    return ("living" if diff > 0 else "dead"), lm, dm, pooled


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dead-dmodel", type=int, default=256,
                   help="Which dead size to compare against (the frozen "
                        "reads are per-point: 256 = stage 1's matched "
                        "point; 512 = the ratified overshoot; 384 = the "
                        "fallback ceiling). Never pool sizes.")
    p.add_argument("--living-dmodel", type=int, default=None,
                   help="Living-arm size filter. Required once an arm "
                        "exists at more than one size (the bridge made "
                        "living_full exist at 256 AND 512) -- pooling "
                        "sizes is never valid.")
    p.add_argument("--living-arm", type=str, default="living",
                   choices=("living", "living_full", "living_v3"),
                   help="Which living configuration to compare (the "
                        "staged ladder: 'living' = minimal, "
                        "'living_full' = BP + consolidation). Never pool "
                        "configurations.")
    args = p.parse_args()

    runs = []
    for path in sorted(OUTPUT_ROOT.glob("*/pilot_result.json")):
        r = json.loads(path.read_text())
        if not r["admissible"]:
            print(f"[verdict] {path.parent.name}: INADMISSIBLE ({r['outcome']}) -- excluded")
            continue
        runs.append((path.parent, r))

    living = [r for _, r in runs if r["arm"] == args.living_arm
              and (args.living_dmodel is None
                   or r["d_model"] == args.living_dmodel)]
    sizes = {r["d_model"] for r in living}
    if len(sizes) > 1:
        print(f"[verdict] REFUSING: {args.living_arm} exists at sizes "
              f"{sorted(sizes)}; pass --living-dmodel (never pool sizes)")
        return 1
    dead = [r for _, r in runs
            if r["arm"] == "dead" and r["d_model"] == args.dead_dmodel]
    living_dm = living[0]["d_model"] if living else "?"
    print(f"[verdict] comparing {args.living_arm}@{living_dm} vs dead@{args.dead_dmodel}")
    if len(living) < 5 or len(dead) < 5:
        print(f"[verdict] incomplete: living={len(living)}/5 dead={len(dead)}/5 admissible")
        return 1

    # Only the two comparison groups get the (checkpoint-reload) NMSE
    # recompute -- never pool arms or sizes, never waste evals.
    runs = [
        (d, r) for d, r in runs
        if (r["arm"] == args.living_arm
            and (args.living_dmodel is None
                 or r["d_model"] == args.living_dmodel))
        or (r["arm"] == "dead" and r["d_model"] == args.dead_dmodel)
    ]
    print("[verdict] recomputing NMSE from final checkpoints (blind metric)...")
    nmse: dict[str, list[float]] = {args.living_arm: [], "dead": []}
    for run_dir, r in runs:
        value = _nmse_for_run(run_dir, r)
        nmse[r["arm"]].append(value)
        print(f"  {run_dir.name}: nmse={value:.6f} "
              f"(raw l_pred={r['heldout']['l_pred_mean']:.6f})")

    probe = {
        args.living_arm: [r["probe"]["top1"] for r in living],
        "dead": [r["probe"]["top1"] for r in dead],
    }

    nmse_axis = _axis(nmse[args.living_arm], nmse["dead"], lower_is_better=True)
    probe_axis = _axis(probe[args.living_arm], probe["dead"], lower_is_better=False)

    wins = sum(1 for a in (nmse_axis, probe_axis) if a[0] == "living")
    losses = sum(1 for a in (nmse_axis, probe_axis) if a[0] == "dead")
    survives = losses == 0 and wins >= 1
    # Per-point reads, each frozen in the pre-registration BEFORE its
    # runs existed. The comparison point determines the verdict's force.
    if args.living_arm in ("living_full", "living_v3"):
        # Run 2 of the configuration ladder (frozen 2026-07-17 before any
        # living_full run existed): NEW claim, not KF2 revived. Asymmetric
        # rule per the fragility fix -- wins need > 1 sigma; a KILL needs a
        # loss > 2 SIGMA (a 1-2 sigma loss on 5 seeds with 15x variance
        # ratios is noise-of-noise territory); anything else is TRACKED.
        def _ratio(axis, lower_is_better):
            _, lm, dm, pooled = axis
            diff = (dm - lm) if lower_is_better else (lm - dm)
            return diff / max(pooled, 1e-12)
        r_nmse = _ratio(nmse_axis, True)
        r_probe = _ratio(probe_axis, False)
        hard_loss = min(r_nmse, r_probe) < -2.0
        any_win = max(r_nmse, r_probe) > 1.0
        soft = [name for name, r in (("nmse", r_nmse), ("probe", r_probe))
                if -2.0 <= r < -1.0]
        if hard_loss:
            verdict = (f"KILL (ladder rule): {args.living_arm} loses an axis at "
                       ">2 sigma vs matched static capacity")
        elif any_win and not soft:
            verdict = (f"{args.living_arm} SURVIVES: wins an axis, no "
                       "loss beyond noise -- register the full-config claim "
                       "on its own feet")
        elif any_win and soft:
            verdict = (f"{args.living_arm} SURVIVES WITH FLAGS: wins an axis but "
                       f"soft-loses {soft} (1-2 sigma) -- survival stands, "
                       f"the soft loss is tracked prominently for run 3")
        else:
            verdict = ("TRACKED-INCONCLUSIVE: no axis won beyond 1 sigma -- "
                       "ladder continues to run 3, deltas recorded")
        verdict += (f" [ratios: nmse {r_nmse:+.1f} sigma, "
                    f"probe {r_probe:+.1f} sigma]")
    elif args.dead_dmodel == 256:  # stage 1: the matched point
        if survives:
            verdict = "KF2-strong SURVIVES at the matched point (bracket now decisive)"
        elif losses > 0:
            verdict = "KILL: living arm loses >= 1 axis at matched capacity"
        else:
            verdict = "KILL: tie on both axes (ON the curve -- no advantage at matched capacity)"
    elif args.dead_dmodel == 512:  # the ratified overshoot (4x ceiling)
        if survives:
            verdict = ("STRONG-FORM REFUTATION of the capacity explanation: "
                       "living beats the 4x-generous ceiling")
        else:
            verdict = ("INCONCLUSIVE AT THE CEILING: dead@512 wins/ties an "
                       "axis, but 512 exceeds the plausible ceiling (a 4x "
                       "model winning is scale, not an explanation of the "
                       "matched-point result). Pre-committed consequence: "
                       "the 384 run is REQUIRED; no claim-status change.")
    elif args.dead_dmodel == 384:  # the plausible ceiling: full verdict force
        if survives:
            verdict = ("KF2 SURVIVES the plausible ceiling: capacity "
                       "explanation refuted at 384; beyond-ceiling scale "
                       "effects (512) noted, not verdict-bearing")
        else:
            verdict = ("KILL: living arm fails the two-axis rule at the "
                       "plausible capacity ceiling (384) -- the stage-1 "
                       "advantage is explained by effective capacity")
    else:
        verdict = f"no frozen read exists for dead@{args.dead_dmodel}; result reported without verdict force"

    report = {
        "criteria": "blind amendment 0fcc92a, 2026-07-16",
        "nmse": {
            args.living_arm: nmse[args.living_arm], "dead": nmse["dead"],
            "axis": {"winner": nmse_axis[0], "living_mean": nmse_axis[1],
                     "dead_mean": nmse_axis[2], "pooled_sigma": nmse_axis[3]},
        },
        "probe_top1": {
            args.living_arm: probe[args.living_arm], "dead": probe["dead"],
            "axis": {"winner": probe_axis[0], "living_mean": probe_axis[1],
                     "dead_mean": probe_axis[2], "pooled_sigma": probe_axis[3]},
        },
        "verdict": verdict,
    }
    (OUTPUT_ROOT / "verdict.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"NMSE : living {nmse_axis[1]:.6f} vs dead {nmse_axis[2]:.6f} "
          f"(pooled sigma {nmse_axis[3]:.6f}) -> {nmse_axis[0]}")
    print(f"PROBE: living {probe_axis[1]:.4f} vs dead {probe_axis[2]:.4f} "
          f"(pooled sigma {probe_axis[3]:.4f}) -> {probe_axis[0]}")
    print()
    print(f"VERDICT: {verdict}")
    print(f"[verdict] written to {OUTPUT_ROOT / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
