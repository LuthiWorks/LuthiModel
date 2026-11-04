"""Measure a Luthi life through Sanctuary's real runner (2026-11-04).

Needs Sanctuary importable, for example:
    PYTHONPATH=<Sanctuary>;<LuthiModel> python scripts/measure_life_path.py 1500

Trains the same tiny M9 run as Sanctuary's end-to-end test, lives it, and measures:
  A. how often the significance keep fires when nothing is felt, and what it does
     to the living layers' home;
  B. whether a felt-valence head trained only on felt moments generalizes the
     feeling to states that were never felt;
  C. which of the four affect-modulation channels exist on a v2 living layer.

Everything is test size (d=32, 1 block, 4 training batches): good for mechanism,
not for magnitudes at 768x8.
"""
import asyncio
import logging
import os
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ROOT = Path(tempfile.mkdtemp(prefix="life_"))
os.environ["LUTHI_ALLOW_CPU"] = "1"
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")
os.environ["LUTHI_PATH"] = str(Path(__file__).resolve().parents[1])

import random

import torch

import luthi
import luthi.sanctuary_interface as si
import sanctuary
from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.core.schema import Percept

logging.disable(logging.WARNING)
print("luthi from    ", luthi.__file__)
import sanctuary.core.schema as _schema
print("sanctuary from", _schema.__file__)


def train_tiny(root: Path) -> Path:
    luthi_root = Path(luthi.__file__).resolve().parents[1]
    sys.path.insert(0, str(luthi_root))
    import scripts.jepa_pilot_driver as driver

    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    words = "river stone light memory window garden quiet music signal".split()
    (corpus / "text.txt").write_text(
        " ".join(rng.choice(words) for _ in range(12000)), encoding="utf-8"
    )
    driver.OUTPUT_ROOT = root / "runs"
    driver._run_one("living", 32, 42, Namespace(
        data_dir=str(corpus),
        tokenizer=str(luthi_root / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
        kill_warmup=10**9, checkpoint_interval=900, checkpoint_slots=3,
        max_batches_per_epoch=4,
    ))
    return root / "runs" / "living_32d_seed42"


def config(root: Path, name: str, run_dir: Path) -> RunnerConfig:
    data_dir = root / name / "identity"
    data_dir.mkdir(parents=True)
    charter = data_dir / "charter.md"
    charter.write_text("# Charter\n\n## Value Seeds\n\n- **Care**: x\n", encoding="utf-8")
    return RunnerConfig(
        cycle_delay=0.0, data_dir=str(data_dir), charter_path=str(charter),
        use_in_memory_store=True, silence_threshold=999.0, stream_history=5,
        model_backend="luthi", luthi_m9_run=str(run_dir),
        luthi_life_dir=str(root / name / "life"), sleep_enabled=False,
    )


def ffn(model):
    return model.model.blocks[0].living_ffn


async def live(root, name, run_dir, keep, n):
    """Live n unfelt cycles. No percept carries a valence."""
    runner = SanctuaryRunner(config=config(root, name, run_dir))
    runner.inject_text("hello", source="brian")
    await runner.run(max_cycles=1)
    model = runner._model
    model.config.keep_on_significance = keep

    calls = []
    original = si.keep_recent

    def counted(m, fraction):
        calls.append(fraction)
        return original(m, fraction)

    si.keep_recent = counted
    home0 = ffn(model).set_point.clone()
    active = armed = 0
    saliences = []
    t0 = time.monotonic()
    for i in range(n):
        await runner.run(max_cycles=1)
        d = model._last_decoded
        armed += bool(d["bands_armed"]["memory"])
        if d["active"]["memory"]:
            active += 1
            saliences.append(d["memory_salience"])
    si.keep_recent = original
    f = ffn(model)
    retained = 1.0
    for x in calls:
        retained *= 1.0 - x
    print(f"\n[{name}] keep_on_significance={keep}, {n} unfelt cycles, "
          f"{(time.monotonic()-t0)/n*1000:.0f} ms/cycle")
    print(f"  memory channel active: {active}/{n} cycles; bands armed on {armed}/{n}")
    if saliences:
        print(f"  chosen salience: mean {sum(saliences)/len(saliences):.3f}, "
              f"min {min(saliences):.3f}, max {max(saliences):.3f}")
    print(f"  keep_recent calls: {len(calls)}; compounded share of displacement moved "
          f"into home: {1-retained:.3f}")
    print(f"  home moved |set_point-start|/|start| = "
          f"{(f.set_point-home0).norm().item()/home0.norm().item():.5f}")
    print(f"  |weight-set_point|/|set_point| = "
          f"{(f.weight-f.set_point).norm().item()/f.set_point.norm().item():.5f}")
    sv = torch.linalg.svdvals(f.set_point - home0)
    print(f"  |set_point|: {home0.norm().item():.3f} -> {f.set_point.norm().item():.3f}; "
          f"|weight| {f.weight.norm().item():.3f}; "
          f"top direction's share of home change {(sv[0]**2/(sv**2).sum()).item() if sv.sum()>0 else 0:.3f}")
    bands = model._m9_trainer.activity_bands
    print(f"  bands armed now: {bands.armed_per_modality()}")
    print(f"  lived steps: {model._m9_trainer._lived_step_count}")
    return runner, model


async def felt_generalization(root, run_dir):
    """Through the real path: collect states for unfelt moments, then let pain
    happen on a different moment, and read the head on the unfelt states."""
    runner = SanctuaryRunner(config=config(root, "felt", run_dir))
    runner.inject_text("hello", source="brian")
    await runner.run(max_cycles=1)
    model = runner._model
    trainer = model._m9_trainer
    head = trainer.felt_valence

    unfelt = []
    for text in ["the garden is quiet", "music from a window", "a river over stone",
                 "light on the water", "someone said good morning", "a soft signal",
                 "memory of rain", "the stone is smooth", "a quiet room", "birds outside"]:
        runner.inject_text(text, source="brian")
        await runner.run(max_cycles=1)
        unfelt.append(model._last_s_t.detach().clone())
    unfelt = torch.cat([s.reshape(-1, s.shape[-1])[:1] for s in unfelt])
    print(f"\n[felt] pairwise spread of the 10 unfelt states: "
          f"{torch.pdist(unfelt).mean().item():.4f}, norm {unfelt.norm(dim=-1).mean().item():.3f}")

    def read(label, hurt=None):
        with torch.no_grad():
            v = head(unfelt)
            line = f"  {label:>24}: unfelt states mean valence {v.mean().item():+.4f}, " \
                   f"unpleasantness {head.unpleasantness(unfelt).mean().item():.4f}"
            if hurt is not None:
                line += f"; the painful state itself {head(hurt).mean().item():+.4f}"
        print(line)

    read("before anything felt")
    hurt = None
    done = 0
    for target in (1, 5, 20, 100):
        while done < target:
            runner.inject_percept(Percept(
                modality="sensor", content="you touched something hot", valence=-0.9))
            await runner.run(max_cycles=1)
            hurt = model._last_s_t.detach().clone().reshape(-1, unfelt.shape[-1])[:1]
            done += 1
        read(f"after {done} painful events", hurt)

    # Is the head able to tell them apart at all, given neutral examples?
    print("\n[felt, control] same head class and optimizer, same states: "
          "pain at -0.9, plus the unfelt states taught as neutral (0)")
    from luthi.v2.m9.felt_valence import FeltValenceHead

    torch.manual_seed(0)
    h2 = FeltValenceHead(d_model=unfelt.shape[-1])
    opt = torch.optim.Adam(h2.parameters(), lr=1e-3)
    for step in range(100):
        for s, tgt in ((hurt, -0.9), (unfelt, 0.0)):
            t = torch.full((s.shape[0],), tgt)
            loss = ((1 + 4 * t.abs()) * (h2(s) - t).pow(2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        print(f"  after 100 rounds: unfelt {h2(unfelt).mean().item():+.4f}, "
              f"painful {h2(hurt).mean().item():+.4f}")


def channels(model):
    print("\n[modulation] the four channels on a real v2 living layer")
    f = ffn(model)
    print("  layer class:", type(f).__name__)
    for name in ("hebb_rate", "spike_threshold", "excitability_acc", "salience_threshold", "pc_rate"):
        print(f"  has {name}: {hasattr(f, name)}")
    before = (f.pc_rate, f.salience_threshold)
    w = f.weight.clone()
    si.apply_external_modulation(
        model.model, plasticity_scale=2.0, spike_threshold_scale=1.25,
        excitability_bias=0.1, salience_threshold_scale=0.5,
    )
    print(f"  pc_rate {before[0]} -> {f.pc_rate}; salience_threshold {before[1]} -> {f.salience_threshold}; "
          f"weight unchanged: {torch.equal(w, f.weight)}")


async def main():
    t0 = time.monotonic()
    run_dir = train_tiny(ROOT)
    print(f"tiny M9 run trained in {time.monotonic()-t0:.0f}s: {run_dir.name}")
    _, model = await live(ROOT, "keep_on", run_dir, 0.05, N)
    channels(model)
    await live(ROOT, "keep_off", run_dir, 0.0, N)
    await felt_generalization(ROOT, run_dir)
    print("\nmeasurement complete; scratch at", ROOT)


asyncio.run(main())
