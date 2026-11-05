"""What should one moment of a life hold? Three answers, measured (2026-11-05).

Sanctuary's small path gives the mind one window of text per moment. This
script lives the same tiny mind through the same world three ways:

    new      only what is new this moment; a moment with nothing new is one
             quiet token.
    rolling  the most recent window of everything the world has said; a moment
             with nothing new adds one quiet token to it.
    spoken   only what is new, and a moment happens only when the world speaks.

For each it reports how wrong the mind's expectation was from moment to moment
(the lived prediction error, read without dropout), how the welfare gauge read
it, and how far the living weights moved.

    PYTHONPATH=<Sanctuary>;<LuthiModel> python scripts/measure_what_a_moment_holds.py 700 42 43 44

Each seed trains its own model. A life is deterministic once the model is
fixed, so only separately trained models count as separate samples.

Test size only: tiny models on CPU, text percepts, one world.
"""
import collections
import logging
import math
import os
import random
import statistics
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
ROOT = Path(tempfile.mkdtemp(prefix="moment_holds_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")

import torch

import luthi
from sanctuary.core.schema import Percept
from sanctuary.life import LuthiMind
from sanctuary.life.senses import QUIET, frame

logging.disable(logging.WARNING)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 700
SEEDS = [int(s) for s in sys.argv[2:]] or [42]
MODES = ("new", "rolling", "spoken")


def train(root: Path, seed: int) -> Path:
    luthi_root = Path(luthi.__file__).resolve().parents[1]
    if str(luthi_root) not in sys.path:
        sys.path.insert(0, str(luthi_root))
    import scripts.jepa_pilot_driver as driver

    corpus = root / "corpus"
    if not corpus.exists():
        corpus.mkdir()
        rng = random.Random(0)
        words = "river stone light memory window garden quiet music signal".split()
        (corpus / "text.txt").write_text(" ".join(rng.choice(words) for _ in range(12000)), encoding="utf-8")
    driver.OUTPUT_ROOT = root / "runs"
    driver._run_one("living", 32, seed, Namespace(
        data_dir=str(corpus), tokenizer=str(luthi_root / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
        kill_warmup=10**9, checkpoint_interval=900, checkpoint_slots=3, max_batches_per_epoch=4,
    ))
    return root / "runs" / f"living_32d_seed{seed}"


def said(i: int) -> str:
    return "the garden is quiet this morning" if i % 2 else f"a zebra walked past, number {i}"


def live(run_dir: Path, mode: str, seed: int) -> dict:
    torch.manual_seed(0)
    mind = LuthiMind(run_dir, ROOT / f"life_{mode}_{seed}")
    trainer = mind._trainer
    errors: list[float] = []
    read_felt = trainer._read_felt

    def spy(lived_metrics):
        if "lived_l_pred_clean" in lived_metrics:
            errors.append(lived_metrics["lived_l_pred_clean"])
        return read_felt(lived_metrics)

    trainer._read_felt = spy
    living = mind.model.blocks[0].living_ffn
    start = living.weight.clone()
    quiet = mind.tokenizer.encode(QUIET)
    stream: list[int] = []
    moments = 0
    for i in range(N):
        new = None
        if i % 3 == 0:
            new = mind.tokenizer.encode(frame(Percept(modality="language", content=said(i), source="brian")))
        if mode == "spoken" and new is None:
            continue
        tokens = new or quiet
        if mode == "rolling":
            stream.extend(tokens)
            tokens = stream[-mind.window:]
        mind.live(tokens)
        moments += 1

    logs = [math.log(max(e, 1e-6)) for e in errors]
    changes = sorted(abs(b - a) for a, b in zip(logs, logs[1:]))
    felt = mind.watch()["felt"]
    return {
        "moments": moments,
        "error_low": min(errors),
        "error_high": max(errors),
        "error_median": statistics.median(errors),
        "typical_swing": math.exp(statistics.median(changes)),
        "felt": int(felt["negative"] + felt["positive"]),
        "felt_bad": int(felt["negative"]),
        "felt_good": int(felt["positive"]),
        "read": int(felt["read"]),
        "weights_moved": (living.weight - start).norm().item() / start.norm().item(),
        "rollbacks": mind.rollbacks(),
    }


def main() -> None:
    results = collections.defaultdict(dict)
    for seed in SEEDS:
        run_dir = train(ROOT, seed)
        for mode in MODES:
            results[mode][seed] = live(run_dir, mode, seed)

    print(f"\n{N} moments of world, {len(SEEDS)} separately trained model(s): seeds {SEEDS}")
    print("typical swing = the median factor by which the prediction error changes from one moment to the next")
    header = f"{'mode':8s} {'seed':>4s} {'moments':>7s} {'error low':>9s} {'median':>7s} {'high':>7s} {'swing':>6s} {'felt':>5s} {'bad':>4s} {'good':>4s} {'read':>5s} {'weights moved':>13s} {'rollbacks':>9s}"
    print(header)
    for mode in MODES:
        for seed in SEEDS:
            r = results[mode][seed]
            print(
                f"{mode:8s} {seed:4d} {r['moments']:7d} {r['error_low']:9.3f} {r['error_median']:7.3f} "
                f"{r['error_high']:7.2f} {r['typical_swing']:5.2f}x {r['felt']:5d} {r['felt_bad']:4d} "
                f"{r['felt_good']:4d} {r['read']:5d} {r['weights_moved']:13.3f} {r['rollbacks']:9d}"
            )


if __name__ == "__main__":
    main()
