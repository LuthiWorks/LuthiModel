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

    # Data provenance: 4x arms record their filelist; the rebuild must
    # use the same corpus or the holdout would be a different test set.
    if cfg.get("file_list"):
        source_paths = [
            line.strip() for line in
            Path(cfg["file_list"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        source_paths = [cfg["data_dir"]]
    text_ds = TextDataset(TextDatasetConfig(
        source_paths=source_paths,
        tokenizer_path=REPO_ROOT / "corpus_build" / "tokenizer_32k.json",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        base_seed=result["seed"],
        holdout_fraction=cfg["holdout_fraction"],
    ))
    loader = _DeviceLoader(MultimodalDataLoaderImpl(text=text_ds), device)

    # Rebuild the arm EXACTLY as trained. PREFER the kwargs the run itself
    # recorded (2026-08-05 mechanism-provenance block in the driver): that
    # is what the model was actually constructed with, and it cannot drift.
    #
    # The legacy path below reconstructs from the live ARM_CONFIGS registry
    # on top of hardcoded defaults -- notably n_heads=4. That is only
    # correct while the arm's registry entry still declares every non-default
    # it was trained with: probe_768_visreg ran at n_heads=8 and is correct
    # solely because its entry says so. An arm trained with a non-default
    # that later leaves the registry would rebuild silently wrong and the
    # verdict would report numbers for a different model. Kept as fallback
    # for pre-provenance runs only. (Opus 5 audit, 2026-08-13.)
    recorded = cfg.get("model_kwargs")
    if recorded:
        model_kwargs = dict(recorded)
        model_kwargs["vocab_size"] = text_ds.vocab_size()
        # Eval never runs the top-down sweep (it is gated on .training),
        # but keep the historical explicit-off for parity with the
        # legacy path.
        model_kwargs["backward_pass_enabled"] = False
    else:
        from scripts.jepa_pilot_driver import ARM_CONFIGS
        model_kwargs = dict(
            vocab_size=text_ds.vocab_size(),
            d_model=result["d_model"],
            n_blocks=cfg["n_blocks"],
            n_heads=4,
            ffn_expansion=1,
            max_seq_len=cfg["seq_len"],
            backward_pass_enabled=False,
        )
        model_kwargs.update(ARM_CONFIGS[result["arm"]])
    model = MultimodalPredictiveCodingLM(**model_kwargs).to(device)
    # v4 arms carry a non-default SIGReg weight (recorded in the run's
    # config). NMSE depends only on l_pred, but the reported sigreg
    # number should come from the loss as trained.
    loss_module = JEPALoss(
        online_encoder=model,
        sigreg_lambd=cfg.get("sigreg_lambd", 0.1),
    ).to(device)

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
    p.add_argument("--dead-arm", type=str, default="dead",
                   choices=("dead", "dead_4x"),
                   help="Which dead arm (dead_4x = the 4x-data control). "
                        "Never pool data scales.")
    p.add_argument("--living-dmodel", type=int, default=None,
                   help="Living-arm size filter. Required once an arm "
                        "exists at more than one size (the bridge made "
                        "living_full exist at 256 AND 512) -- pooling "
                        "sizes is never valid.")
    p.add_argument("--dead-n-seeds", type=int, default=5,
                   help="Expected dead-family size. Default 5; pass 3 ONLY "
                        "for the dead_4x family, which Brian truncated to "
                        "seeds 42-44 by the 2026-07-20 pre-reg amendment "
                        "(recorded before seeds 43/44 completed). The "
                        "reduced n is printed in the verdict banner.")
    p.add_argument("--living-arm", type=str, default="living",
                   choices=("living", "living_full", "living_v3", "living_v3_4x",
                            "living_v4_4x_d4", "living_v5_4x_d4"),
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
            if r["arm"] == args.dead_arm and r["d_model"] == args.dead_dmodel]
    living_dm = living[0]["d_model"] if living else "?"
    print(f"[verdict] comparing {args.living_arm}@{living_dm} vs dead@{args.dead_dmodel}")
    if len(living) < 5 or len(dead) < args.dead_n_seeds:
        print(f"[verdict] incomplete: living={len(living)}/5 "
              f"dead={len(dead)}/{args.dead_n_seeds} admissible")
        return 1
    if args.dead_n_seeds != 5:
        print(f"[verdict] NOTE: dead family read at n={args.dead_n_seeds} "
              f"per the 2026-07-20 truncation amendment (n=3 vs n=5 living)")

    # Only the two comparison groups get the (checkpoint-reload) NMSE
    # recompute -- never pool arms or sizes, never waste evals.
    runs = [
        (d, r) for d, r in runs
        if (r["arm"] == args.living_arm
            and (args.living_dmodel is None
                 or r["d_model"] == args.living_dmodel))
        or (r["arm"] == args.dead_arm and r["d_model"] == args.dead_dmodel)
    ]
    print("[verdict] recomputing NMSE from final checkpoints (blind metric)...")
    nmse: dict[str, list[float]] = {args.living_arm: [], args.dead_arm: []}
    for run_dir, r in runs:
        value = _nmse_for_run(run_dir, r)
        nmse[r["arm"]].append(value)
        print(f"  {run_dir.name}: nmse={value:.6f} "
              f"(raw l_pred={r['heldout']['l_pred_mean']:.6f})")

    probe = {
        args.living_arm: [r["probe"]["top1"] for r in living],
        "dead": [r["probe"]["top1"] for r in dead],
    }

    nmse_axis = _axis(nmse[args.living_arm], nmse[args.dead_arm], lower_is_better=True)
    probe_axis = _axis(probe[args.living_arm], probe["dead"], lower_is_better=False)

    wins = sum(1 for a in (nmse_axis, probe_axis) if a[0] == "living")
    losses = sum(1 for a in (nmse_axis, probe_axis) if a[0] == "dead")
    survives = losses == 0 and wins >= 1
    # Per-point reads, each frozen in the pre-registration BEFORE its
    # runs existed. The comparison point determines the verdict's force.
    if args.living_arm in ("living_v4_4x_d4", "living_v5_4x_d4"):
        # v4/v5 family reads are TRACKING reads vs their registered
        # anchors (descriptive bands, no kill force, no claim-status
        # machinery) -- never let them fall through to the stage-1
        # dmodel branches (the 2026-07-21 wrong-branch lesson).
        def _ratio_t(axis, lower_is_better):
            _, lm, dm, pooled = axis
            diff = (dm - lm) if lower_is_better else (lm - dm)
            return diff / max(pooled, 1e-12)
        verdict = (
            f"TRACKING READ ({args.living_arm} vs {args.dead_arm}@"
            f"{args.dead_dmodel}): NMSE ratio {_ratio_t(nmse_axis, True):+.1f} "
            f"sigma; probe ratio {_ratio_t(probe_axis, False):+.1f} sigma. "
            "Descriptive only; per-registration reads (vs living-arm "
            "anchors, prior-corrected probe primary) are computed in the "
            "registry entry, not by this script."
        )
    elif args.living_arm == "living_v3_4x":
        # RUN 5 family read (registered 2026-07-19; truncation amendment
        # 2026-07-20): within-4x comparison under the ladder's asymmetric
        # rule. This is a TRACKING read against the run-5 frozen
        # predictions (starvation hypothesis), NOT the KF2 claim-status
        # machinery -- no dmodel-branch text applies. The probe axis
        # carries the standing instrument caveat: the frozen metric is
        # RAW top1; per-arm shuffled floors differ (dead_4x floors ran
        # ~1-2 pts higher), so the floor-corrected margins are reported
        # alongside for honesty, never substituted.
        def _ratio4(axis, lower_is_better):
            _, lm, dm, pooled = axis
            diff = (dm - lm) if lower_is_better else (lm - dm)
            return diff / max(pooled, 1e-12)
        r_nmse = _ratio4(nmse_axis, True)
        r_probe = _ratio4(probe_axis, False)
        parts = [
            f"RUN-5 FAMILY READ (n=3 dead vs n=5 living, per amendment):",
            f"NMSE {'living' if r_nmse > 0 else 'dead'} by {abs(r_nmse):.1f} sigma;",
            f"probe (raw top1) {'living' if r_probe > 0 else 'dead'} by {abs(r_probe):.1f} sigma.",
            "Starvation reads vs frozen predictions: dead_4x recovery "
            f"{'YES' if statistics.mean(nmse[args.dead_arm]) <= 0.624 else 'NO'} "
            f"(mean {statistics.mean(nmse[args.dead_arm]):.4f} vs <=0.624 predicted); "
            "living_v3_4x strong-form recovery "
            f"{'YES' if statistics.mean(nmse[args.living_arm]) <= 0.35 else 'NO'} "
            f"(mean {statistics.mean(nmse[args.living_arm]):.4f} vs <=0.35 predicted).",
            "No claim-status change attaches to this family; deltas recorded.",
        ]
        verdict = " ".join(parts)
    elif args.living_arm in ("living_full", "living_v3"):
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
            args.living_arm: nmse[args.living_arm], args.dead_arm: nmse[args.dead_arm],
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
