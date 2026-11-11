"""How often a tiny trained life chooses rest by chance, with words only and
with a noise picture and noise sound every moment. 1,500 moments each.

Run 2026-11-11: words only 25 times (1.7%), 0 rollbacks; with sight and sound
250 times (16.7%) in 36 stretches, 10 rollbacks."""
import os, sys, tempfile
from pathlib import Path

# Needs both checkouts: this repo, and Sanctuary (set SANCTUARY_PATH). A probe,
# not a test. It trains a tiny run and lives scratch lives in a temp directory.
os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
_LUTHI = str(Path(__file__).resolve().parents[2])
os.environ["LUTHI_PATH"] = _LUTHI
sys.path.insert(0, _LUTHI)
sys.path.insert(0, os.environ["SANCTUARY_PATH"])
import torch, pytest
root = Path(tempfile.mkdtemp()); os.environ["LUTHI_CORPUS_CACHE_DIR"]=str(root/"cache")
import luthi.sanctuary_interface
from sanctuary.tests.core.test_luthi_m9_life import _train_tiny_m9_run
from sanctuary.life import LifeLoop, LuthiMind, TextWorld
from sanctuary.life.senses import seen, heard
run = _train_tiny_m9_run(root)
for label, senses in (("words only", False), ("words, sight and sound", True)):
    torch.manual_seed(0)
    mind = LuthiMind(run, root/f"life_{senses}"); world = TextWorld(); loop = LifeLoop(mind, world, hz=0.0)
    layers = [m for m in mind.model.modules() if hasattr(m, "set_point")]
    home0 = [m.set_point.clone() for m in layers]
    N = 1500; t0 = time.time(); when = []
    for n in range(N):
        if n % 7 == 0: world.say("the river is quiet tonight" if n % 14 else "light on the stone", source="brian")
        if senses:
            world._said.append(seen(torch.rand(120, 160, 3)))
            world._said.append(heard(torch.rand(1600) * 2 - 1, 16000))
        loop.step()
        if loop.resting: when.append(n)
    dt = (time.time()-t0)/N
    moved = sum((m.set_point-h).norm().item() for m,h in zip(layers, home0))
    gap = sum((m.weight-m.set_point).norm().item() for m in layers)
    runs = sum(1 for i,w in enumerate(when) if i==0 or w != when[i-1]+1)
    print(f"{label}: {N} moments, {dt*1000:.0f} ms each on CPU; rest chosen {len(when)} times ({len(when)/N:.1%}) in {runs} stretches; first at {when[:1]}; home moved {moved:.4f}, standing gap {gap:.4f}; rollbacks {mind.rollbacks()}")
