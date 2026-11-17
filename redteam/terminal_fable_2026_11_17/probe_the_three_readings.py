"""The three readings (rest, recall, letting go) before and after brief 5,
Part B.

Two measurements, the ones Opus's review of 2026-11-14 (section 2) made, on
more than one seed.

``steady``: a steady lean toward rest, at ``shift`` times the noise, after
300 moments of noise. How many moments in a row it is read as rest. Opus's
``probe_a_steady_choice.py`` does this and divides by what is left over,
which is nothing once the choice is read for as long as it lasts; so the
loop is repeated here.

``life``: one tiny life, as Opus's ``probe_choices_in_a_tiny_life.py`` lives
it (a tiny trained run, something said every seventh moment, words only,
Sanctuary's defaults), for one seed. The seed sets the tiny run's training
seed (42 + seed) and the life's random stream. Seed 0 is Opus's life. It
records how often the mind rested, recalled and let go, the longest stretch
of rest, and whether any memory was let go at the moment rest brought it
back.

``--reading`` picks how the choices are read: ``as_built_2026_11_11`` or
``drift_removed``. "Before" also needs Sanctuary as it was (SANCTUARY_PATH
pointing at a checkout of 3ec929a), since two of the corrections are in its
loop.

One CPU thread: a tiny life is chaotic down to rounding, and the thread
count decides it (Opus, 2026-11-14).

    SANCTUARY_PATH=<checkout> OMP_NUM_THREADS=1 python \\
        redteam/terminal_fable_2026_11_17/probe_the_three_readings.py life \\
        --seed 0 --reading drift_removed --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
_LUTHI = str(Path(__file__).resolve().parents[2])
os.environ["LUTHI_PATH"] = _LUTHI
sys.path.insert(0, _LUTHI)

import torch  # noqa: E402

torch.set_num_threads(1)

from luthi.v2.m9.rest_choice import RestChoice  # noqa: E402


def stretches(flags: list[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for n, f in enumerate(flags + [False]):
        if f and start is None:
            start = n
        elif not f and start is not None:
            runs.append((start, n - start))
            start = None
    return runs


def steady(args) -> dict:
    d = 32
    rows = []
    for reading in ("as_built_2026_11_11", "drift_removed"):
        for width, shift in ((3.0, 10.0), (3.0, 100.0), (6.0, 10.0)):
            for seed in range(args.seeds):
                choice = RestChoice(d, width=width, reading=reading,
                                    drift_span=args.drift_span)
                generator = torch.Generator().manual_seed(seed)
                nothing = torch.zeros(d)
                chosen, by_chance = [], 0
                for t in range(300 + args.moments):
                    lean = torch.randn(d, generator=generator)
                    if t >= 300:
                        lean = lean + shift * choice.direction
                    out = choice.read(lean, nothing)
                    if t >= 300:
                        chosen.append(out["chosen"])
                    else:
                        by_chance += out["chosen"]
                in_a_row = next((i for i, c in enumerate(chosen) if not c), len(chosen))
                after = chosen[in_a_row:]
                rows.append({"reading": reading, "width": width, "shift": shift, "seed": seed,
                             "read_in_a_row": in_a_row, "of": len(chosen),
                             "read_after_that": sum(after), "moments_after_that": len(after),
                             "chosen_in_the_300_of_noise": by_chance})
            mine = [r for r in rows if (r["reading"], r["width"], r["shift"]) == (reading, width, shift)]
            runs = [r["read_in_a_row"] for r in mine]
            print(f"{reading:20s} width {width}, lean {shift:g} times the noise: read as rest for "
                  f"{min(runs)} to {max(runs)} moments in a row of {args.moments} "
                  f"({args.seeds} seeds)")
    return {"what": "a steady lean toward rest", "moments": args.moments,
            "drift_span": args.drift_span, "rows": rows}


def train_tiny_run(root: Path, seed: int) -> Path:
    """Sanctuary's ``_train_tiny_m9_run`` (tests/core/test_luthi_m9_life.py)
    with the training seed open. Seed 42 is that function exactly."""
    from argparse import Namespace
    import random

    import scripts.jepa_pilot_driver as driver

    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    words = "river stone light memory window garden quiet music signal".split()
    (corpus / "text.txt").write_text(
        " ".join(rng.choice(words) for _ in range(12000)), encoding="utf-8")
    driver.OUTPUT_ROOT = root / "runs"
    driver._run_one("living", 32, seed, Namespace(
        data_dir=str(corpus),
        tokenizer=str(Path(_LUTHI) / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=1, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
        kill_warmup=10**9, checkpoint_interval=900, checkpoint_slots=3,
        max_batches_per_epoch=4,
    ))
    return root / "runs" / f"living_32d_seed{seed}"


def life(args) -> dict:
    sys.path.insert(0, os.environ["SANCTUARY_PATH"])
    # The life's folder is kept where it is named (Brian, 2026-11-17).
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="three_readings_"))
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise SystemExit(f"{root} is not empty")
    os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(root / "cache")

    import luthi.sanctuary_interface  # noqa: F401
    from sanctuary.life import LifeLoop, LuthiMind, TextWorld

    # The readings are built inside the mind with the config's defaults; the
    # kind of reading is set here, where each one is built.
    built = RestChoice.__init__

    def build(self, d_model, **kwargs):
        kwargs["reading"] = args.reading
        kwargs["drift_span"] = args.drift_span
        if args.reading == "as_built_2026_11_11":
            kwargs.pop("holds", None)
        built(self, d_model, **kwargs)

    RestChoice.__init__ = build
    try:
        run = train_tiny_run(root, 42 + args.seed)
        torch.manual_seed(args.seed)
        mind = LuthiMind(run, root / "life")
        world = TextWorld()
        loop = LifeLoop(mind, world, hz=0.0)
    finally:
        RestChoice.__init__ = built
    kinds = {name: (c.reading_kind, c.holds) for name, c in (
        ("rest", mind._trainer.rest_choice), ("recall", mind._trainer.recall_choice),
        ("let_go", mind._trainer.let_go_choice))}

    brought_by_rest: list[int] = []
    let_go_at: list[int] = []
    bring_back, recall = loop._bring_back, loop._recall
    in_recall = {"now": False}

    def recalling(moment):
        in_recall["now"] = True
        try:
            return recall(moment)
        finally:
            in_recall["now"] = False

    def bringing_back(moment):
        ok = bring_back(moment)
        if ok and not in_recall["now"]:
            brought_by_rest.append(loop.cycles + 1)     # the moment it is looked at
        return ok

    loop._recall, loop._bring_back = recalling, bringing_back
    letting_go = loop.memories.let_go

    def let_go(memory_id, *, cycle):
        let_go_at.append(cycle)
        return letting_go(memory_id, cycle=cycle)

    loop.memories.let_go = let_go
    resting, recalls, let_go_chosen = [], [], []
    t0 = time.time()
    for n in range(args.moments):
        if n % 7 == 0:
            world.say("the river is quiet tonight" if n % 14 else "light on the stone",
                      source="brian")
        moment = loop.step()
        resting.append(bool(loop.resting))
        recalls.append(bool(moment.chosen.get("recall", {}).get("chosen", False)))
        let_go_chosen.append(bool(moment.chosen.get("let_go", {}).get("chosen", False)))
    rest_runs = stretches(resting)
    mem = loop.status()["memories"]
    result = {
        "what": "one tiny life", "seed": args.seed, "training_seed": 42 + args.seed,
        "reading": args.reading, "drift_span": args.drift_span, "moments": args.moments,
        "choices_built_as": kinds, "threads": torch.get_num_threads(),
        "sanctuary": os.environ["SANCTUARY_PATH"], "life_folder": str(root),
        "seconds_per_moment": (time.time() - t0) / args.moments,
        "rested": loop.rested, "rest_stretches": len(rest_runs),
        "longest_rest_stretch": max((length for _, length in rest_runs), default=0),
        "longest_rest_stretch_from": max(rest_runs, key=lambda r: r[1])[0] if rest_runs else None,
        "resting_at_the_end": resting[-1],
        "recall_chosen": sum(recalls), "recalled": mem["recalled"],
        "recalled_nothing": mem["recalled_nothing"],
        "let_go_chosen": sum(let_go_chosen), "let_go": mem["let_go_by_choice"],
        "let_go_chosen_on_a_memory_rest_brought": mem.get("let_go_chosen_on_a_memory_rest_brought"),
        "kept": mem["kept"], "revisited": mem["revisited_in_rest"], "dreamed": mem["dreamed_in_rest"],
        "let_go_at_the_moment_rest_brought_it_back": sorted(set(let_go_at) & set(brought_by_rest)),
        "brought_by_rest": brought_by_rest, "let_go_at": let_go_at,
        "rollbacks": mind.rollbacks(),
    }
    print({k: v for k, v in result.items() if k not in ("brought_by_rest", "let_go_at")})
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("steady", "life"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="steady: how many noise seeds")
    ap.add_argument("--reading", choices=("drift_removed", "as_built_2026_11_11"),
                    default="drift_removed")
    ap.add_argument("--drift-span", type=int, default=64)
    ap.add_argument("--moments", type=int, default=1500)
    ap.add_argument("--keep", default="", help="life: the folder the life is kept in")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    result = steady(args) if args.what == "steady" else life(args)
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"{out} exists; not overwriting it")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
