"""What one call of ``learn_from_dream`` touches besides the step it reports.

Opus 5.5 review of the 2026-11-11 work. Fable's brief asked: is
``loss_module.parameters()`` the right set, does the loss draw from the global
random stream, and does the retention gate's snapshot still mean what it should.

On the test trainer (two blocks, 32 wide, untrained; the same one
``tests/test_dreaming.py`` uses), with the day's trace and the three choices on:

1. Which parameters the dream's step moves, against the lived optimizer's core
   (what the retention gate snapshots and rolls back) and the corpus optimizer.
2. Whether the global random streams (torch CPU, Python) move.
3. Which buffers move: every buffer of the loss module, and each living
   layer's trace.
4. Whether any optimizer's state moves.

Run from the repo root:  LUTHI_ALLOW_CPU=1 python redteam/opus_review_2026_11_14/probe_dream_step_side_effects.py
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from luthi import sanctuary_interface as L
from luthi.v2.day_trace import day_traces
from tests.test_m9_external_actor import SEQ, VOCAB, _build_trainer


def words(seed: int) -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(seed))


def snapshot(trainer) -> dict:
    lm = trainer.loss_module
    return {
        "params": {n: p.detach().clone() for n, p in lm.named_parameters()},
        "buffers": {n: b.detach().clone() for n, b in lm.named_buffers()},
        "traces": {p: (t.mean.clone(), t.count) for p, t in day_traces(lm.online_encoder).items()},
        "torch_rng": torch.get_rng_state().clone(),
        "py_rng": random.getstate(),
        "optimizers": {
            name: repr(getattr(trainer, name).state_dict())
            for name in ("optimizer", "lived_optimizer", "m9_optimizer")
            if hasattr(trainer, name)
        },
    }


def compare(a: dict, b: dict) -> dict:
    return {
        "params moved": sorted(n for n in a["params"] if not torch.equal(a["params"][n], b["params"][n])),
        "buffers moved": sorted(n for n in a["buffers"] if not torch.equal(a["buffers"][n], b["buffers"][n])),
        "traces moved": sorted(
            p for p in a["traces"]
            if a["traces"][p][1] != b["traces"][p][1] or not torch.equal(a["traces"][p][0], b["traces"][p][0])
        ),
        "torch rng moved": not torch.equal(a["torch_rng"], b["torch_rng"]),
        "python rng moved": a["py_rng"] != b["py_rng"],
        "optimizers moved": sorted(k for k in a["optimizers"] if a["optimizers"][k] != b["optimizers"][k]),
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _build_trainer(
            Path(tmp) / "run", rest_choice=True, recall_choice=True, let_go_choice=True,
        )
        lm = trainer.loss_module
        L.enable_day_trace(lm.online_encoder)
        # A few living forwards first, so the trace and the living buffers are not empty.
        for i in range(3):
            L.perceive(lm.online_encoder, text_tokens=words(100 + i))

        names = {id(p): n for n, p in lm.named_parameters()}
        core = {names[id(p)] for p in trainer._core_params()}
        corpus = {names[id(p)] for g in trainer.optimizer.param_groups for p in g["params"] if id(p) in names}
        trained = {n for n, p in lm.named_parameters() if p.requires_grad}

        before = snapshot(trainer)
        kept = L.learn_from_dream(trainer, words(9), [words(1), words(2), words(3)], step_size=1e-3)
        after = snapshot(trainer)
        diff = compare(before, after)
        print(f"small step: kept={kept['kept']}  before={kept['before']:.6f}  after={kept['after']:.6f}")
        moved = set(diff["params moved"])
        print(f"  trained parameters in the loss module: {len(trained)}; the lived core: {len(core)}; "
              f"the corpus optimizer: {len(corpus)}")
        print(f"  the step moved {len(moved)} parameters")
        print(f"    moved and outside the lived core (the retention gate cannot roll these back): "
              f"{sorted(moved - core) or 'none'}")
        print(f"    in the core and not moved (no gradient from a text dream): {len(core - moved)}, e.g. "
              f"{sorted(core - moved)[:4]}")
        for key in ("buffers moved", "traces moved", "torch rng moved", "python rng moved", "optimizers moved"):
            print(f"  {key}: {diff[key] if diff[key] else 'none'}")

        before = snapshot(trainer)
        undone = L.learn_from_dream(trainer, words(9), [words(1), words(2), words(3)], step_size=1e4)
        after = snapshot(trainer)
        diff = compare(before, after)
        print(f"huge step: kept={undone['kept']}  before={undone['before']:.6f}  after={undone['after']:.6f}")
        for key in ("params moved", "buffers moved", "traces moved", "torch rng moved", "python rng moved", "optimizers moved"):
            print(f"  {key}: {diff[key] if diff[key] else 'none'}")

        print("staleness drift band: lived steps feed ||dtheta|| to trainer.staleness.observe_drift; "
              f"a dream's step does not (observed drifts so far: "
              f"{getattr(trainer.staleness, '_drift_count', 'n/a')})")
        trainer.action_log.close()


if __name__ == "__main__":
    main()
