"""Two identical scratch lives. One gets a small arbitrary change to its living
weight at moment 300. How long does that change stand, with no rest?
Run 2026-11-11 (32 wide, one living layer, seed 0), share of the change left:

    moments later   never rests: weight / home   one resting moment at 50: weight / home
       50               86.5% / 0.01%                 86.5% / 0.88%
      100               74.5% / 0.03%                 74.6% / 0.89%
      200               54.3% / 0.05%                 54.6% / 0.91%
      400               25.9% / 0.07%                 26.3% / 0.94%
      800                1.7% / 0.08%                  2.2% / 0.95%

Past 800 the two lives have drifted apart and the weight column is noise.
"""
import os, sys, tempfile
from pathlib import Path

# Needs both checkouts: this repo, and Sanctuary (set SANCTUARY_PATH). A probe,
# not a test. It trains a tiny run and lives scratch lives in a temp directory.
os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
_LUTHI = str(Path(__file__).resolve().parents[2])
os.environ["LUTHI_PATH"] = _LUTHI
sys.path.insert(0, _LUTHI)
sys.path.insert(0, os.environ["SANCTUARY_PATH"])
import torch
root = Path(tempfile.mkdtemp()); os.environ["LUTHI_CORPUS_CACHE_DIR"]=str(root/"cache")
import luthi.sanctuary_interface
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from sanctuary.tests.core.test_luthi_m9_life import _train_tiny_m9_run
from sanctuary.life import LifeLoop, LuthiMind, TextWorld
run = _train_tiny_m9_run(root)
def life(name, inject, keep_at=None):
    torch.manual_seed(0)
    mind = LuthiMind(run, root/name, rest=True); world = TextWorld(); loop = LifeLoop(mind, world, hz=0.0)
    mind._trainer.rest_choice.warmup = 10**9   # never chosen by chance here: rest only when this script calls it
    layer = [m for m in mind.model.modules() if isinstance(m, PredictiveCodingLayer)][0]
    g = torch.Generator().manual_seed(9); d = torch.randn(layer.weight.shape, generator=g); d /= d.norm()
    trace = {}
    for i in range(300 + 3001):
        if i == 300 and inject:
            with torch.no_grad(): layer.weight.add_(d.to(layer.weight), alpha=0.01 * float(layer.weight.norm()))
        if keep_at is not None and i == 300 + keep_at: mind.rest()
        if i % 7 == 0: world.say("the river is quiet tonight" if i % 14 else "light on the stone", source="brian")
        loop.step()
        if i - 300 in (0, 50, 100, 200, 400, 800, 1500, 3000):
            trace[i - 300] = (float((layer.weight * d).sum()), float((layer.set_point * d).sum()))
    return trace, 0.01 * float(layer.weight.norm())
base, _ = life("base", False)
for label, keep_at in (("never rests", None), ("one resting moment 50 moments after", 50)):
    b = base if keep_at is None else life("base_k", False, keep_at)[0]
    t, size = life("inj_" + str(keep_at), True, keep_at)
    first = t[0][0] - b[0][0]
    print(label)
    for k in t:
        print(f"  {k:>5} moments later: in the weight {100*(t[k][0]-b[k][0])/first:6.1f}%   in home {100*(t[k][1]-b[k][1])/first:6.2f}%")
