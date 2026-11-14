"""What the half-day's trace holds in a life that also replays its corpus.

Opus 5.5 review of the 2026-11-11 work (Fable's brief, points 3 and 4: "about 5
living forwards a moment in a life; the trace notes 1").

The trace notes ``weight - home`` once per lived forward. Corpus replay runs
inside ``suspend_day_record``, so it is not noted, but its forwards run with
plasticity on and write into the same weight. So every note includes what the
replay wrote in the last thousand or so forwards.

This splits the trace exactly. For every forward that can change a living
layer, ``d = weight - home`` before and after gives what was new in that forward,
``e = d_after - f * d_before`` with ``f`` the forward's own pull toward home.
``e`` is added to a shadow for the forward's kind (lived, corpus, or anything
that changed the weight between forwards), and every shadow decays by ``f``.
The shadows sum to ``d`` exactly. At each lived note the shadows are averaged
the way the trace averages ``d``, so the trace's mean is split into what came
from lived moments, from corpus replay, from between forwards, and from where
the weight stood at the start.

One tiny life (Sanctuary's ``_train_tiny_m9_run``, one block, 32 wide), Fable's
world schedule, 1,500 moments, rest never chosen (so the trace is never poured).

Run (needs both checkouts):
  SANCTUARY_PATH=../sanctuary LUTHI_ALLOW_CPU=1 python redteam/opus_review_2026_11_14/probe_what_the_trace_holds_in_a_life.py
Env: N (moments, 1500), RATIO (corpus replay ratio, default the run's own),
RUN (a trained M9 run to live instead of the tiny one, e.g. the trunk from
redteam/fable_2026_11_11/halfday_train_trunk.py).
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
_LUTHI = str(Path(__file__).resolve().parents[2])
os.environ["LUTHI_PATH"] = _LUTHI
sys.path.insert(0, _LUTHI)
sys.path.insert(0, os.environ["SANCTUARY_PATH"])

import torch

root = Path(tempfile.mkdtemp(prefix="opus_trace_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(root / "cache")

import luthi.sanctuary_interface  # noqa: E402,F401
from luthi.v2.living_layer_pc import PredictiveCodingLayer  # noqa: E402
from sanctuary.life import LifeLoop, LuthiMind, TextWorld  # noqa: E402
from sanctuary.tests.core.test_luthi_m9_life import _train_tiny_m9_run  # noqa: E402

N = int(os.environ.get("N", 1500))
KINDS = ("lived", "corpus", "between", "start")


class Split:
    """The exact split of one layer's ``weight - home`` by where it came from."""

    def __init__(self, layer: PredictiveCodingLayer) -> None:
        self.layer = layer
        d = (layer.weight - layer.set_point).detach().clone()
        self.shadow = {k: torch.zeros_like(d) for k in KINDS}
        self.shadow["start"] = d.clone()
        self.expected = d.clone()          # d as the last forward left it
        self.mean = {k: torch.zeros_like(d) for k in KINDS}
        self.count = 0.0
        self.horizon = layer._day_trace.horizon
        self.forwards = Counter()
        self.write_size = Counter()
        self._before = None

    def kind(self) -> str:
        layer = self.layer
        if layer._plasticity_frozen:
            return "frozen"
        return "corpus" if layer._day_record_suspended else "lived"

    def pre(self, module, args) -> None:
        d = (self.layer.weight - self.layer.set_point).detach()
        between = d - self.expected
        if float(between.norm()) > 0.0:
            self.shadow["between"] += between
        self._before = (self.kind(), d.clone())

    def post(self, module, args, out) -> None:
        kind, d_before = self._before
        layer = self.layer
        d_after = (layer.weight - layer.set_point).detach()
        self.forwards[kind] += 1
        if kind != "frozen":
            f = (1.0 - layer.homeostatic_decay * layer._wake_attenuation) ** layer.inference_steps_per_forward
            new = d_after - f * d_before
            for k in KINDS:
                self.shadow[k] *= f
            self.shadow[kind] += new
            self.write_size[kind] += float(new.norm())
        else:
            if not torch.equal(d_after, d_before):
                raise AssertionError("a frozen forward changed the weight or home")
        self.expected = d_after.clone()
        if kind == "lived":
            # The trace's own averaging (luthi/v2/day_trace.py, DayTrace.note).
            self.count = min(self.count + 1.0, float(self.horizon))
            for k in KINDS:
                self.mean[k] += (self.shadow[k] - self.mean[k]) / self.count


def main() -> None:
    run = Path(os.environ["RUN"]) if "RUN" in os.environ else _train_tiny_m9_run(root)
    torch.manual_seed(0)
    mind = LuthiMind(run, root / "life")
    if "RATIO" in os.environ:
        mind._trainer.m9_config.corpus_replay_ratio = float(os.environ["RATIO"])
    mind._trainer.rest_choice.warmup = 10**9      # never rests here: the trace is never poured
    world = TextWorld()
    loop = LifeLoop(mind, world, hz=0.0, rest_every_seconds=None)
    layers = [m for m in mind.model.modules() if isinstance(m, PredictiveCodingLayer)]
    splits = []
    for layer in layers:
        s = Split(layer)
        layer.register_forward_pre_hook(s.pre)
        layer.register_forward_hook(s.post)
        splits.append(s)
    for n in range(N):
        if n % 7 == 0:
            world.say("the river is quiet tonight" if n % 14 else "light on the stone", source="brian")
        loop.step()
    print(f"run {run.name}; corpus replay ratio {mind._trainer.m9_config.corpus_replay_ratio}; {N} moments")
    for i, s in enumerate(splits):
        trace = s.layer._day_trace.mean
        total = sum(s.mean[k] for k in KINDS)
        err = float((total - trace).norm() / trace.norm().clamp_min(1e-30))
        print(f"layer {i}: forwards per moment {dict((k, round(v / N, 2)) for k, v in s.forwards.items())}; "
              f"split matches the trace to {err:.1e}")
        print(f"  size of each forward's new part, mean: "
              + ", ".join(f"{k} {s.write_size[k] / max(s.forwards[k], 1):.2e}" for k in ("lived", "corpus")))
        t2 = float((trace * trace).sum())
        for k in KINDS:
            part = s.mean[k]
            share = float((part * trace).sum()) / t2
            cos = float((part * trace).sum() / (part.norm() * trace.norm()).clamp_min(1e-30))
            print(f"  {k:8s} share of the trace {share:+6.1%}; its own size {float(part.norm()) / float(trace.norm()):.2f} "
                  f"of the trace's; cosine with the trace {cos:+.2f}")
        lc = float((s.mean['lived'] * s.mean['corpus']).sum() / (s.mean['lived'].norm() * s.mean['corpus'].norm()).clamp_min(1e-30))
        print(f"  cosine between the lived part and the corpus part: {lc:+.2f}")
        print(f"  trace size {float(trace.norm()):.3e} against home's {float(s.layer.set_point.norm()):.3e}")


if __name__ == "__main__":
    main()
