"""What is alive in a Luthi life on Sanctuary's small path? 700 moments.

The counterpart of ``map_life_path.py``, which mapped the old path
(SanctuaryRunner and CognitiveCycle) on 2026-11-05. Same tiny model, same
corpus, same seed, same things said at the same moments, so the two can be read
side by side.

    PYTHONPATH=<Sanctuary>;<LuthiModel> python scripts/map_small_path.py 700

Test size only: a tiny model on CPU, one seed, text percepts.
"""
import collections
import logging
import os
import random
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
ROOT = Path(tempfile.mkdtemp(prefix="small_path_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")

import torch

import luthi
from sanctuary.life import LifeLoop, LuthiMind, TextWorld
from sanctuary.life.senses import QUIET

logging.disable(logging.WARNING)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 700


def train(root):
    luthi_root = Path(luthi.__file__).resolve().parents[1]
    sys.path.insert(0, str(luthi_root))
    import scripts.jepa_pilot_driver as driver

    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    words = "river stone light memory window garden quiet music signal".split()
    (corpus / "text.txt").write_text(" ".join(rng.choice(words) for _ in range(12000)), encoding="utf-8")
    driver.OUTPUT_ROOT = root / "runs"
    driver._run_one("living", 32, 42, Namespace(
        data_dir=str(corpus), tokenizer=str(luthi_root / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
        kill_warmup=10**9, checkpoint_interval=900, checkpoint_slots=3, max_batches_per_epoch=4,
    ))
    return root / "runs" / "living_32d_seed42"


class Recorded:
    """LuthiModel's interface, counting every call Sanctuary makes through it."""

    def __init__(self, module):
        self._module = module
        self.counts = collections.Counter()
        self.perceived = []

    def __getattr__(self, name):
        target = getattr(self._module, name)
        if not callable(target):
            return target

        def call(*args, **kwargs):
            self.counts[name] += 1
            if name == "perceive":
                self.perceived.append(kwargs["text_tokens"][0].tolist())
            return target(*args, **kwargs)

        return call


def main():
    run_dir = train(ROOT)
    torch.manual_seed(0)
    mind = LuthiMind(run_dir, ROOT / "life")
    mind._luthi = recorded = Recorded(mind._luthi)
    world = TextWorld()
    loop = LifeLoop(mind, world, hz=0.0)
    trainer = mind._trainer

    # What the learner inside LuthiModel does on its own. Sanctuary calls none
    # of this; it is listed so nothing that acts on the mind goes unnamed.
    inside = collections.Counter()
    for name in ("_run_corpus_replay", "_check_retention_gate"):
        original = getattr(trainer, name)

        def make(original=original, name=name):
            def spy(*a, **k):
                out = original(*a, **k)
                if name == "_run_corpus_replay":
                    inside["corpus replay steps"] += int(out)
                elif out:
                    inside["retention checks"] += 1
                return out
            return spy
        setattr(trainer, name, make())

    living = mind.model.blocks[0].living_ffn
    home0, weight0 = living.set_point.clone(), living.weight.clone()
    said = 0
    started = time.perf_counter()
    for i in range(N):
        if i % 3 == 0:
            world.say("the garden is quiet this morning" if i % 2 else f"a zebra walked past, number {i}", source="brian")
            said += 1
        loop.step()
    seconds = time.perf_counter() - started

    # Everything in every view must be something the world said or a quiet
    # token. Rebuild the stream from what was said and check each view against it.
    stream, expected, told = [], [], 0
    quiet = mind.tokenizer.encode(QUIET)
    for i in range(N):
        if i % 3 == 0:
            line = "the garden is quiet this morning" if i % 2 else f"a zebra walked past, number {i}"
            stream += mind.tokenizer.encode(f"brian says: {line}\n")
        else:
            stream += quiet
        expected.append(stream[-mind.window:])
    only_the_world = recorded.perceived == expected
    told = sum("[system, learning]" in mind.tokenizer.decode(t) for t in recorded.perceived)
    longest = max(len(t) for t in recorded.perceived)

    status = loop.status()
    print(f"\n{N} moments in {seconds:.1f} s ({1000 * seconds / N:.0f} ms each); things said: {said}")
    print("\nWhat reached the mind:")
    print(f"  {'every view was only what the world said':38s} {only_the_world}")
    print(f"  {'moments something new arrived':38s} {N - status['world']['quiet_moments']}")
    print(f"  {'moments with nothing new':38s} {status['world']['quiet_moments']}")
    print(f"  {'views that told of a rollback':38s} {told}")
    print(f"  longest view: {longest} of {mind.window} tokens")
    print("\nWhat Sanctuary called in the mind:")
    for k, v in sorted(recorded.counts.items()):
        print(f"  {k:38s} {v}")
    print("\nWhat the learner inside LuthiModel did on its own:")
    for k, v in sorted(inside.items()):
        print(f"  {k:38s} {v}")
    print(f"  {'lived learning steps':38s} {trainer._lived_step_count}")
    print("\nThe world:", status["world"])
    print("Chosen, with nothing to carry it out:", status["chosen_without_route"])
    print("Felt (the watch-only gauge):", {k: round(v, 2) for k, v in status["felt"].items()})
    print("Stillness:", status["stillness"])
    print("Kill states:", {k: v.value for k, v in trainer.m9_kills.states().items()})
    print("Rollbacks:", status["rollbacks"])
    print(f"Home moved over the life: {(living.set_point - home0).norm().item() / home0.norm().item():.4f}")
    print(f"Living weights moved over the life: {(living.weight - weight0).norm().item() / weight0.norm().item():.4f}")
    path = loop.save("map")
    print(f"Saved: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    print("Checkpoints this life holds:", loop.status()["checkpoints"])


if __name__ == "__main__":
    main()
