"""What is actually alive in a Luthi life? Default settings, sleep on, 700 cycles.

Records, per cycle: what Sanctuary assembled, what reached Luthi's input, what
Luthi's output carried, and which mechanisms ran. It changes nothing in a repo.

Needs Sanctuary importable, for example:
    PYTHONPATH=<Sanctuary>;<LuthiModel> python scripts/map_life_path.py 700

Test size only: a tiny model on CPU, one seed, text percepts.
"""
import asyncio
import json
import logging
import os
import random
import sys
import tempfile
from argparse import Namespace
from collections import Counter
from pathlib import Path

os.environ["LUTHI_ALLOW_CPU"] = "1"
ROOT = Path(tempfile.mkdtemp(prefix="map_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")
os.environ.setdefault("LUTHI_PATH", str(Path(__file__).resolve().parents[1]))

import torch

import luthi
import luthi.sanctuary_interface as si
from sanctuary.api.runner import RunnerConfig, SanctuaryRunner

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


def nonempty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple)):
        return len(value) > 0
    return True


async def main():
    run_dir = train(ROOT)
    data_dir = ROOT / "life" / "identity"
    data_dir.mkdir(parents=True)
    charter = data_dir / "charter.md"
    charter.write_text("# Charter\n\n## Value Seeds\n\n- **Care**: x\n", encoding="utf-8")
    # Defaults everywhere except what a test life needs.
    runner = SanctuaryRunner(config=RunnerConfig(
        cycle_delay=0.0, data_dir=str(data_dir), charter_path=str(charter),
        model_backend="luthi", luthi_m9_run=str(run_dir), luthi_life_dir=str(ROOT / "life" / "life"),
    ))
    runner.inject_text("hello", source="brian")
    await runner.run(max_cycles=1)
    cycle, model = runner.cycle, runner._model
    trainer = model._m9_trainer

    print("parts present:", {name: type(getattr(cycle, name, None)).__name__ for name in (
        "sleep", "experiential", "learning_governor", "rate_controller", "world_graph",
        "motor", "growth", "environment", "memory", "scaffold", "identity", "authority")})
    print("store:", "in-memory" if runner._config.use_in_memory_store else "persistent",
          "| sleep every", cycle.sleep.config.cycles_between_sleep if cycle.sleep else None)

    inputs, outputs, prompts = Counter(), Counter(), []
    ranges = {}
    stages, rates = Counter(), []
    counts = Counter()

    def track(name, value):
        lo, hi = ranges.get(name, (value, value))
        ranges[name] = (min(lo, value), max(hi, value))

    think = model.think

    async def spy_think(ci):
        counts["cycles"] += 1
        for field in ("new_percepts", "prediction_errors", "surfaced_memories", "world_model_query_results",
                      "charter_summary", "self_authored_identity", "communication_urges", "tool_catalog"):
            inputs[field] += nonempty(getattr(ci, field))
        inputs["previous_thought.inner_speech"] += bool(ci.previous_thought and ci.previous_thought.inner_speech)
        inputs["emotional_state.felt_quality"] += bool(ci.emotional_state.felt_quality)
        inputs["self_model.current_state"] += bool(ci.self_model.current_state)
        inputs["learning_state"] += ci.learning_state is not None
        inputs["scaffold_signals (any)"] += any(
            nonempty(getattr(ci.scaffold_signals, f)) for f in ("attention_highlights", "goal_status", "anomalies"))
        inputs["experiential.knowledge_signals"] += nonempty(ci.experiential_state.knowledge_signals)
        track("computed valence", ci.emotional_state.computed.valence)
        track("computed arousal", ci.emotional_state.computed.arousal)
        track("experiential arousal", ci.experiential_state.affect_arousal)
        track("experiential precision", ci.experiential_state.precision_weight)
        track("experiential attention", ci.experiential_state.attention_salience)
        counts["tool_catalog_size"] = len(ci.tool_catalog)
        counts["charter_chars"] = len(ci.charter_summary)
        if ci.learning_state is not None:
            counts["consolidation_active"] = int(ci.learning_state.consolidation_active)
        out = await think(ci)
        for field in ("inner_speech", "external_speech", "predictions", "memory_ops", "world_model_updates",
                      "world_model_queries", "goal_proposals", "knowledge_cell_requests", "tool_requests",
                      "learning_preference", "cycle_rate_proposal", "growth_reflection"):
            outputs[field] += nonempty(getattr(out, field))
        outputs["attention_guidance"] += bool(out.attention_guidance.focus_on or out.attention_guidance.deprioritize)
        outputs["emotional shifts"] += bool(out.emotional_state.valence_shift or out.emotional_state.arousal_shift)
        outputs["self_model_updates.current_state"] += bool(out.self_model_updates.current_state)
        if cycle.sleep is not None:
            stages[cycle.sleep.stage.value] += 1
        if cycle.rate_controller is not None:
            for attr in ("current_hz", "target_hz", "rate_hz"):
                if hasattr(cycle.rate_controller, attr):
                    rates.append(round(float(getattr(cycle.rate_controller, attr)), 3))
                    break
        return out

    model.think = spy_think
    fmt = model._format_input
    model._format_input = lambda ci: (prompts.append(fmt(ci)) or prompts[-1])

    for name in ("_run_lived_update", "_check_retention_gate", "_read_felt"):
        original = getattr(trainer, name)

        def make(original=original, name=name):
            def spy(*a, **k):
                counts[name] += 1
                return original(*a, **k)
            return spy
        setattr(trainer, name, make())
    replay = trainer._run_corpus_replay

    def spy_replay():
        steps = replay()
        counts["corpus_replay_steps"] += steps
        return steps

    trainer._run_corpus_replay = spy_replay
    select = trainer.select_action

    def spy_select(*a, **k):
        counts["select_action"] += 1
        counts["select with planning (budget>0)"] += (k.get("budget") is None or k.get("budget", 0) > 0)
        return select(*a, **k)

    trainer.select_action = spy_select
    consolidate = model.consolidate
    home = lambda: model.model.blocks[0].living_ffn.set_point.clone()

    def spy_consolidate():
        counts["consolidate calls"] += 1
        return consolidate()

    model.consolidate = spy_consolidate
    keep = si.keep_recent
    si.keep_recent = lambda m, fraction: counts.update({"keep_recent calls": 1}) or keep(m, fraction)
    modulation = si.apply_external_modulation

    def spy_mod(m, **kw):
        track("plasticity_scale", kw.get("plasticity_scale", 1.0))
        track("salience_threshold_scale", kw.get("salience_threshold_scale", 1.0))
        return modulation(m, **kw)

    si.apply_external_modulation = spy_mod

    home0 = home()
    for i in range(N):
        if i % 3 == 0:
            runner.inject_text("the garden is quiet this morning" if i % 2 else f"a zebra walked past, number {i}", source="brian")
        await runner.run(max_cycles=1)

    total = counts["cycles"]
    print(f"\n{total} cycles")
    print("\nWhat Sanctuary assembled (cycles non-empty):")
    for k, v in sorted(inputs.items()):
        print(f"  {k:34s} {v:4d}")
    print("\nWhat reached Luthi's input text:")
    kinds = Counter()
    for p in prompts:
        kinds["says:"] += "says:" in p
        kinds["[self]"] += "[self]" in p
        kinds["I remember"] += "I remember" in p
        kinds["I expected"] += "I expected" in p
        kinds["I feel"] += "I feel" in p
        kinds["[temporal/other percept]"] += ("[" in p.replace("[self]", ""))
        kinds["only 'I am here.'"] += p.strip() == "I am here."
    for k, v in kinds.items():
        print(f"  {k:34s} {v:4d} of {len(prompts)}")
    print("  example:", repr(prompts[-1][:110]))
    import re
    others = sorted({re.sub(r"[0-9]+", "N", q)[:120] for q in prompts if "[" in q.replace("[self]", "")})
    for q in others[:6]:
        print("  bracket percept:", repr(q))
    for name in ("sleep", "experiential", "learning_governor", "rate_controller", "world_graph",
                 "motor", "growth", "environment"):
        print("  part", name, "=", type(getattr(cycle, name, None)).__name__)
    store = getattr(cycle.memory, "_store", None) or getattr(getattr(cycle.memory, "_surfacer", None), "_store", None)
    print("  memory store entries:", len(getattr(store, "_entries", [])) if store is not None else "no store found")
    print("  rate:", round(cycle.rate_controller.current_rate_hz, 3), "Hz, last source:", cycle.rate_controller.last_source,
          "| proposals:", len(cycle.rate_controller.proposal_history))
    print("\nWhat Luthi's output carried (cycles non-empty):")
    for k, v in sorted(outputs.items()):
        print(f"  {k:34s} {v:4d}")
    print("\nSignals (lowest, highest):")
    for k, v in ranges.items():
        print(f"  {k:34s} {v[0]:+.4f} {v[1]:+.4f}")
    print("\nMechanism activity:")
    for k, v in sorted(counts.items()):
        print(f"  {k:34s} {v}")
    print("  sleep stages (cycles):", dict(stages), "| completed sleeps:", cycle.sleep.completed_sleeps if cycle.sleep else None)
    if rates:
        print(f"  cycle rate seen: {min(rates)} to {max(rates)} Hz")
    f = model.model.blocks[0].living_ffn
    print(f"  home moved over the life: {(f.set_point - home0).norm().item() / home0.norm().item():.4f}")
    print("  authority levels:", {k: int(v) for k, v in runner.authority.get_all_levels().items()})
    metrics = model.get_metrics()
    print("  felt summary:", {k: round(v, 2) for k, v in metrics["felt_summary"].items()})
    print("  kept since sleep:", metrics["kept_since_sleep"], "| capped:", metrics["keep_capped_moments"])
    print("  stillness:", metrics["stillness"])
    print("  kill states:", {k: v.value for k, v in trainer.m9_kills.states().items()})
    print("  retention rollbacks:", si.retention_rollbacks(trainer))
    if cycle.world_graph is not None:
        for attr in ("node_count", "size"):
            if hasattr(cycle.world_graph, attr):
                v = getattr(cycle.world_graph, attr)
                print("  world graph", attr, v() if callable(v) else v)
    print("  goals tracked:", len(getattr(runner.scaffold.goals, "goals", getattr(runner.scaffold.goals, "_goals", []))))
    print("  stream history:", len(cycle.stream.history))
    tap = runner._experiential.data_tap if runner._experiential else None
    print("  CfC records:", tap.counts() if tap else None)
    governor = cycle.learning_governor
    if governor is not None:
        print("  learning plan:", json.dumps(governor.plan()))
    life_ckpts = sorted((ROOT / "life" / "life" / "checkpoints").glob("ckpt_*.pt"))
    print("  life checkpoints written:", len(life_ckpts))
    print("done")


asyncio.run(main())
