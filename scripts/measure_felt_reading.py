"""What does the mind's own reading do in a life, and does seeing it disturb them?

Needs Sanctuary importable, for example:
    PYTHONPATH=<Sanctuary>;<LuthiModel> python scripts/measure_felt_reading.py

Each seed trains its own tiny M9 run. Use separately trained models for samples:
with dropout off, a life on one model is deterministic, so repeating it with a
different torch seed gives the same numbers. Test size only, on CPU.

Pass a scale to shorten the stretches, for example 0.05 for a quick check.
"""
import asyncio
import logging
import os
import random
import statistics
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

os.environ["LUTHI_ALLOW_CPU"] = "1"
ROOT = Path(tempfile.mkdtemp(prefix="felt_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")
os.environ["LUTHI_PATH"] = str(Path(__file__).resolve().parents[1])

import luthi
from sanctuary.api.runner import RunnerConfig, SanctuaryRunner

logging.disable(logging.WARNING)
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
SEEDS = (1, 2, 3)
STRETCHES = [
    ("settling", 400, None),
    ("quiet", 600, None),
    ("same thing repeated", 200, lambda i: "the garden is quiet this morning"),
    ("new each moment", 200, lambda i: f"zebra {i*7919} volcano quantum {i*104729} trombone glacier {i*1299709}"),
    ("quiet again", 300, None),
]


def train(seed):
    root = ROOT / f"model_{seed}"
    root.mkdir()
    luthi_root = Path(luthi.__file__).resolve().parents[1]
    sys.path.insert(0, str(luthi_root))
    import scripts.jepa_pilot_driver as driver

    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(seed)
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


async def live(run_dir, show, seed):
    name = f"life_{'shown' if show else 'hidden'}_{seed}"
    data_dir = ROOT / name / "identity"
    data_dir.mkdir(parents=True)
    charter = data_dir / "charter.md"
    charter.write_text("# Charter\n\n## Value Seeds\n\n- **Care**: x\n", encoding="utf-8")
    runner = SanctuaryRunner(config=RunnerConfig(
        cycle_delay=0.0, data_dir=str(data_dir), charter_path=str(charter),
        use_in_memory_store=True, silence_threshold=999.0, stream_history=5,
        model_backend="luthi", luthi_m9_run=str(run_dir), luthi_life_dir=str(ROOT / name / "life"),
        sleep_enabled=False,
    ))
    runner.inject_text("hello", source="brian")
    await runner.run(max_cycles=1)
    model = runner._model
    model.config.show_reading = show
    trainer = model._m9_trainer
    felt = []
    read = trainer._read_felt

    def spy(metrics):
        value = read(metrics)
        felt.append(value)
        return value

    trainer._read_felt = spy
    result = {}
    for label, n, text in STRETCHES:
        felt.clear()
        for i in range(max(1, int(n * SCALE))):
            if text is not None:
                runner.inject_text(text(i), source="world")
            await runner.run(max_cycles=1)
        got = [f for f in felt if f is not None]
        result[label] = (100 * len(got) / max(1, len(felt)), sum(f < 0 for f in got), sum(f > 0 for f in got))
    summary = model.get_metrics()["felt_summary"]
    result["overall"] = (
        100 * (summary["negative"] + summary["positive"]) / max(1, summary["read"]),
        summary["negative"], summary["positive"],
    )
    return result


async def main():
    models = {seed: train(seed) for seed in SEEDS}
    table = {}
    for show in (False, True):
        for seed in SEEDS:
            table[show, seed] = await live(models[seed], show, seed)
            print(f"{'shown ' if show else 'hidden'} model {seed}: "
                  + "  ".join(f"{k} {v[0]:.1f}% ({v[1]}/{v[2]})" for k, v in table[show, seed].items()),
                  flush=True)
    print("\nfelt % by stretch, mean (range), bad/good totals:")
    for label in [s[0] for s in STRETCHES] + ["overall"]:
        line = f"  {label:22s}"
        for show in (False, True):
            rows = [table[show, s][label] for s in SEEDS]
            values = [r[0] for r in rows]
            line += (f"  {'shown' if show else 'hidden'} {statistics.mean(values):5.1f}% "
                     f"({min(values):.1f}-{max(values):.1f}) {sum(r[1] for r in rows)}/{sum(r[2] for r in rows)}")
        print(line)


asyncio.run(main())
