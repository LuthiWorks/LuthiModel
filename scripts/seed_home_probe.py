"""Stage 0 probe for seed S3, the living layer's home
(docs/research/2026-11-03_seeding-luthi.md).

A capacity check (standing rule 1), not a registered experiment: small CPU
runs of the current lineage's living package (the LC-768 configuration, block
store off, at smoke width and depth), one per home, three seeds each. It
reads, per living block, from each run's final checkpoint:

- what the home learned: its movement from its starting value, and how many
  directions that movement spans (top-direction share, participation ratio);
- the live displacement from home and its top-direction share (E2's measure);
- the weight's conditioning (smallest over largest singular value);

and from each run's pilot_result.json, held-out NMSE and the standardized
probe above the embedding baseline (rule 3). At this size those two are
health gauges, not evidence: 512 and below under-show collapse-class
effects (2026-08-15 ruling).

It runs no model forwards of its own; the driver's held-out and probe reads
already freeze plasticity.

Usage:
    LUTHI_ALLOW_CPU=1 python scripts/seed_home_probe.py --data_dir <corpus> \
        --out <dir> [--steps 300] [--seeds 42,43,44]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.jepa_pilot_driver as driver  # noqa: E402
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM  # noqa: E402

D_MODEL = 64
BASE = dict(driver.ARM_CONFIGS["lc768_living"])
BASE.update(n_blocks=2, n_heads=4, block_episode_store="off")

ARMS = {
    "random": {},
    "orthogonal": {"living_home": "orthogonal"},
    "identity": {"living_home": "identity"},
    # The first grown arms raised set_point_adapt_rate, which also raises the
    # rank-1 top-down push into the home; it ran away in block 0 (step 5 of
    # the research log). These keep the shared rate and raise only the
    # follow rate.
    "grown_shared": {"set_point_adapt_rate": 1e-3},
    "grown": {"home_follow_rate": 1e-3},
    "random_cov": {"write_rule": "covariance"},
    "grown_shared_cov": {"set_point_adapt_rate": 1e-3, "write_rule": "covariance"},
    "grown_cov": {"home_follow_rate": 1e-3, "write_rule": "covariance"},
    "orthogonal_cov": {"living_home": "orthogonal", "write_rule": "covariance"},
    "grown_orthogonal_cov": {
        "home_follow_rate": 1e-3, "living_home": "orthogonal", "write_rule": "covariance",
    },
}

_initial_homes: dict[int, torch.Tensor] = {}


def _record_initial_homes():
    real_init = MultimodalPredictiveCodingLM.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        _initial_homes.clear()
        for i, block in enumerate(self.blocks):
            _initial_homes[i] = block.living_ffn.set_point.detach().clone()

    MultimodalPredictiveCodingLM.__init__ = init


def _spectrum(m: torch.Tensor) -> dict:
    sv = torch.linalg.svdvals(m.double())
    e = sv.pow(2)
    total = float(e.sum())
    if total == 0.0:
        return {"top_share": float("nan"), "participation": 0.0}
    return {
        "top_share": float(e[0] / total),
        "participation": float(total ** 2 / e.pow(2).sum()),
    }


def _read(run_dir: Path) -> dict:
    [ckpt] = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-1:]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["online_state_dict"]
    blocks = []
    for i, home0 in sorted(_initial_homes.items()):
        w = state[f"blocks.{i}.living_ffn.weight"].double()
        home = state[f"blocks.{i}.living_ffn.set_point"].double()
        moved = home - home0.double()
        sv = torch.linalg.svdvals(w)
        blocks.append({
            "home_moved_rel": float(moved.norm() / home0.double().norm()),
            "home_moved": _spectrum(moved),
            "displacement_rel": float((w - home).norm() / home.norm()),
            "displacement": _spectrum(w - home),
            "conditioning": float(sv.min() / sv.max()),
        })
    result = json.loads((run_dir / "pilot_result.json").read_text())
    return {
        "blocks": blocks,
        "heldout_nmse": result["heldout"]["nmse_mean"],
        "probe_trunk_minus_embedding": result["probe_standardized_trunk_minus_embedding"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seeds", default="42,43,44")
    args = p.parse_args()
    out = Path(args.out)
    driver.OUTPUT_ROOT = out / "runs"
    _record_initial_homes()

    run_args = Namespace(
        data_dir=args.data_dir,
        tokenizer=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=2, epochs=1, batch_size=8, seq_len=64, stride=64, lr=3e-4,
        holdout_fraction=0.05, heldout_batches=8, probe_batches=8,
        kill_warmup=10**9, checkpoint_interval=10**9, checkpoint_slots=3,
        max_batches_per_epoch=args.steps, no_m9=True,
    )
    readings: dict[str, list[dict]] = {}
    for name, extra in ARMS.items():
        arm = f"seedhome_{name}"
        driver.ARM_CONFIGS[arm] = {**BASE, **extra}
        for seed in (int(s) for s in args.seeds.split(",")):
            driver._run_one(arm, D_MODEL, seed, run_args)
            reading = _read(out / "runs" / driver._run_name(arm, D_MODEL, seed))
            readings.setdefault(name, []).append(reading)
            print(name, seed, json.dumps(reading["blocks"][0]), flush=True)

    summary = {}
    for name, runs in readings.items():
        def ms(values):
            values = [v for v in values if v == v]
            return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)
        n_blocks = len(runs[0]["blocks"])
        summary[name] = {
            "heldout_nmse": ms([r["heldout_nmse"] for r in runs]),
            "probe_trunk_minus_embedding": ms([r["probe_trunk_minus_embedding"] for r in runs]),
            "blocks": [
                {
                    key: ms([r["blocks"][b][k1][k2] if k2 else r["blocks"][b][k1] for r in runs])
                    for key, k1, k2 in (
                        ("home_moved_rel", "home_moved_rel", None),
                        ("home_moved_top_share", "home_moved", "top_share"),
                        ("home_moved_participation", "home_moved", "participation"),
                        ("displacement_rel", "displacement_rel", None),
                        ("displacement_top_share", "displacement", "top_share"),
                        ("conditioning", "conditioning", None),
                    )
                }
                for b in range(n_blocks)
            ],
        }
    (out / "summary.json").write_text(json.dumps({"readings": readings, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
