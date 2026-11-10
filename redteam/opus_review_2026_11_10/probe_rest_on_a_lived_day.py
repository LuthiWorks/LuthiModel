"""Review probe (Opus 5.5, 2026-11-10): the rest pass over a day a real trunk lived.

Run from the LuthiModel root, with Sanctuary importable (about a minute on CPU):

    PYTHONPATH=<Sanctuary>:<LuthiModel> LUTHI_PATH=<LuthiModel> \
        python redteam/opus_review_2026_11_10/probe_rest_on_a_lived_day.py

A two-block, 32-wide living model trained for 300 batches lives 300 moments on
Sanctuary's small path (someone speaks every third moment). Every living
layer records its day with every moment admitted (``salience_threshold = 0``).
Then one rest with the default config: corpus-replay entries tossed, every
lived moment left unjudged (so kept by default, and folded when it repeats).

Printed on 2026-11-10:

    blocks.0: 300 lived inputs; pairwise cosine median 0.983; pairs >= 0.99: 25%;
              cosine to the mean direction median 0.990
    blocks.1: 300 lived inputs; pairwise cosine median 0.990; pairs >= 0.99: 51%;
              cosine to the mean direction median 0.994
    blocks.0: integrated 17, folded 283; home moved 0.23%;
              momentum 16%, update_ema 15%, error_acc 12%, prediction 0.15%
    blocks.1: integrated 5, folded 295; home moved 0.11%
    folded in both layers 279, in exactly one 20

A first run (not this script) kept the corpus-replay entries in the day, 898
per layer, and rested over all of them as unjudged. With the 300-batch model,
blocks 0 and 1 folded 284 and 804; without folding, block 1 hit the 5% bound
and held back 461. One rest moved momentum by 62 to 100% of its norm,
update_ema by 59 to 96% and error_acc by 6 to 22%. With a 4-batch model:
momentum 38 to 103%, update_ema 37 to 93%, error_acc 20 to 118%.
"""

import os
import random
import sys
import tempfile
import types
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
ROOT = Path(tempfile.mkdtemp(prefix="rest_lived_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")

import torch  # noqa: E402

LUTHI = Path(os.environ["LUTHI_PATH"])
sys.path.insert(0, str(LUTHI))
import scripts.jepa_pilot_driver as driver  # noqa: E402
from luthi.v2.day_record import day_records, enable_day_record  # noqa: E402
from luthi.v2.rest import rest  # noqa: E402
from sanctuary.life import LifeLoop, LuthiMind, TextWorld  # noqa: E402

WRITES = ("weight", "prediction", "set_point", "momentum", "update_ema",
          "precision", "error_acc", "plasticity")


def train() -> Path:
    corpus = ROOT / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    words = ("river stone light memory window garden quiet music signal field morning "
             "shadow pattern voice letter bridge winter colour thread").split()
    (corpus / "text.txt").write_text(" ".join(rng.choice(words) for _ in range(40000)), encoding="utf-8")
    driver.OUTPUT_ROOT = ROOT / "runs"
    driver._run_one("living", 32, 42, Namespace(
        data_dir=str(corpus), tokenizer=str(LUTHI / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=2, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2, kill_warmup=10**9,
        checkpoint_interval=900, checkpoint_slots=3, max_batches_per_epoch=300))
    return ROOT / "runs" / "living_32d_seed42"


def main() -> None:
    torch.manual_seed(0)
    mind = LuthiMind(train(), ROOT / "life")
    world = TextWorld()
    loop = LifeLoop(mind, world, hz=0.0)
    layers = {p: m for p, m in mind.model.named_modules() if hasattr(m, "_day_record")}
    for m in layers.values():
        m.salience_threshold = 0.0
    enable_day_record(mind.model)

    # Note which moment names are lived (perceive) and which are corpus replay.
    lu = mind._luthi
    proxy = types.SimpleNamespace(**{n: getattr(lu, n) for n in dir(lu) if not n.startswith("__")})
    lived = set()

    def perceive(model, **kw):
        out = lu.perceive(model, **kw)
        lived.add(next(iter(layers.values()))._sparse_step_count)
        return out

    proxy.perceive = perceive
    mind._luthi = proxy
    for i in range(300):
        if i % 3 == 0:
            world.say(f"a zebra walked past, number {i}" if i % 2 else "the garden is quiet this morning",
                      source="brian")
        loop.step()

    records = day_records(mind.model)
    replay = sorted({e.moment for r in records.values() for e in r} - lived)
    for path, record in records.items():
        x = torch.stack([e.input_pattern for e in record if e.moment in lived]).double()
        u = x / x.norm(dim=1, keepdim=True)
        off = (u @ u.T)[~torch.eye(len(x), dtype=bool)]
        mean_dir = u.mean(0) / u.mean(0).norm()
        print(f"{path}: {len(x)} lived inputs; pairwise cosine median {off.median():.4f}; "
              f"pairs >= 0.99 {(off >= 0.99).double().mean():.3f}; "
              f"cosine to the mean direction median {(u @ mean_dir).median():.4f}")

    before = {p: {w: getattr(l, w).detach().clone() for w in WRITES} for p, l in layers.items()}
    report = rest(mind.model, keep=[], toss=replay, unjudged="keep")
    for path, lr in report.layers.items():
        layer = layers[path]
        rel = {w: float((getattr(layer, w) - before[path][w]).norm()
                        / before[path][w].norm().clamp_min(1e-30)) for w in WRITES}
        print(f"  {path}: integrated {len(lr.integrated)}, folded {len(lr.folded)}, held {len(lr.held)}, "
              f"tossed {len(lr.tossed)}; home moved {lr.home_change:.2e}; "
              + ", ".join(f"{w} {v:.2e}" for w, v in rel.items()))
    folded = [set(lr.folded) for lr in report.layers.values()]
    both = set.intersection(*folded)
    print(f"  folded in every layer {len(both)}, in some but not all {len(set.union(*folded) - both)}")


if __name__ == "__main__":
    main()
