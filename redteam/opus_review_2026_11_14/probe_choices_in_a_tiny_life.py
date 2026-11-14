"""Why a tiny life recalled 71 times with dream learning and 4 without.

Opus 5.5 review of the 2026-11-11 work (Fable's brief, point 1, last bullet).

Three lives of the same tiny trained run, the same world and the same seed
(Fable's ``probe_rest_by_chance.py`` schedule: something is said every seventh
moment), 1,500 moments each, words only:

- ``off``: rest dreams, and the mind does not learn from its dreams.
- ``on``: as Sanctuary runs it by default: one checked step on each dream.
- ``on, undone``: the step is taken and checked as in ``on``, and then the
  trained parameters are put back exactly. If this life matches ``off`` moment
  for moment, the only way dream learning changes a life is through the
  parameters it moves.

Per moment it records each choice's reading (rest, recall, let go), the size of
the action's lean away from doing nothing, and the size of each lived step.
It prints the counts, the moments around each dream, and the share of recalls
that fall soon after a dream.

Run (needs both checkouts):
  SANCTUARY_PATH=../sanctuary LUTHI_ALLOW_CPU=1 OMP_NUM_THREADS=1 python redteam/opus_review_2026_11_14/probe_choices_in_a_tiny_life.py
Env: N (moments, 1500), ARMS (comma list of off,on,undone), OUT (a JSON path for the per-moment record).

The thread count decides the life (2026-11-14): with OMP_NUM_THREADS=1 the
counts are Fable's (recall 71 with dream learning, 4 without); with 2 threads
the same code and seed give 4 and 2. Each setting repeats exactly.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
_LUTHI = str(Path(__file__).resolve().parents[2])
os.environ["LUTHI_PATH"] = _LUTHI
sys.path.insert(0, _LUTHI)
sys.path.insert(0, os.environ["SANCTUARY_PATH"])

import torch

root = Path(tempfile.mkdtemp(prefix="opus_tiny_life_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(root / "cache")

import luthi.sanctuary_interface  # noqa: E402,F401
from luthi.v2.m9.rest_choice import RestChoice  # noqa: E402
from sanctuary.life import LifeLoop, LuthiMind, TextWorld  # noqa: E402
from sanctuary.tests.core.test_luthi_m9_life import _train_tiny_m9_run  # noqa: E402

N = int(os.environ.get("N", 1500))
ARMS = os.environ.get("ARMS", "off,on,undone").split(",")


def watch_reads(record: list) -> None:
    """Record, for each read, which choice it was, the reading, the size of the
    action's lean from doing nothing, and that lean's cosine with the direction."""
    original = RestChoice.read

    def read(self, action, do_nothing):
        out = original(self, action, do_nothing)
        lean = (action.reshape(-1).to(self.direction) - do_nothing.reshape(-1).to(self.direction))
        record.append({
            "seed_dir": float(self.direction[0]),   # tells the three choices apart
            "reading": out["reading"], "chosen": out["chosen"], "armed": out["armed"],
            "lean": float(lean.norm()),
            "cos": float(torch.dot(self.direction, lean) / lean.norm().clamp_min(1e-30)),
        })
        return out

    RestChoice.read = read
    return original


def live(run: Path, arm: str) -> dict:
    torch.manual_seed(0)
    mind = LuthiMind(run, root / f"life_{arm}")
    world = TextWorld()
    loop = LifeLoop(mind, world, hz=0.0, learn_from_dreams=(arm != "off"))
    dirs = {
        "rest": float(mind._trainer.rest_choice.direction[0]),
        "recall": float(mind._trainer.recall_choice.direction[0]),
        "let_go": float(mind._trainer.let_go_choice.direction[0]),
    }
    dreams_at: list[int] = []
    dream = mind.dream

    def dreaming(words, real):
        dreams_at.append(loop.cycles)
        if arm != "undone":
            return dream(words, real)
        params = [p for p in mind._trainer.loss_module.parameters() if p.requires_grad]
        saved = [p.detach().clone() for p in params]
        out = dream(words, real)
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)
        return out

    mind.dream = dreaming
    # When memories come back and when one is let go, recorded where it happens.
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
    reads: list = []
    original = watch_reads(reads)
    rows = []
    t0 = time.time()
    try:
        for n in range(N):
            if n % 7 == 0:
                world.say("the river is quiet tonight" if n % 14 else "light on the stone", source="brian")
            k = len(reads)
            moment = loop.step()
            row = {"n": n, "resting": loop.resting,
                   "recall": bool(moment.chosen.get("recall", {}).get("chosen", False)),
                   "let_go": bool(moment.chosen.get("let_go", {}).get("chosen", False)),
                   "lived_dtheta": moment.learned.get("lived_delta_theta_norm"),
                   "rollbacks": mind.rollbacks()}
            for r in reads[k:]:
                name = next(c for c, d in dirs.items() if d == r["seed_dir"])
                row[f"{name}_reading"], row[f"{name}_lean"], row[f"{name}_cos"] = r["reading"], r["lean"], r["cos"]
            rows.append(row)
    finally:
        RestChoice.read = original
    status = loop.status()
    mem = status["memories"]
    return {
        "arm": arm, "seconds_per_moment": (time.time() - t0) / N, "rows": rows, "dreams_at": dreams_at,
        "brought_by_rest": brought_by_rest, "let_go_at": let_go_at,
        "counts": {
            "rested": loop.rested, "recalled": mem["recalled"], "recalled_nothing": mem["recalled_nothing"],
            "kept": mem["kept"], "let_go": mem["let_go_by_choice"], "revisited": mem["revisited_in_rest"],
            "dreamed": mem["dreamed_in_rest"], "dreams_learned_from": mem["dreams_learned_from"],
            "dreams_undone": mem["dreams_undone"], "rollbacks": mind.rollbacks(),
        },
    }


def recall_moments(result: dict) -> list[int]:
    return [r["n"] for r in result["rows"] if r["recall"]]


def classify(readings: list[float], window: int, width: float, warmup: int = 32) -> list[bool]:
    """RestChoice's rule replayed on recorded readings, for other window lengths."""
    from collections import deque
    q: deque = deque(maxlen=window)
    out = []
    for x in readings:
        q.append(x)
        if len(q) < warmup:
            out.append(False)
            continue
        values = torch.tensor(list(q))
        median = values.median()
        mad = (values - median).abs().median()
        out.append(bool(x > 0.0 and x > float(median + width * mad)))
    return out


def stretches(flags: list[bool]) -> list[tuple[int, int]]:
    """(first moment, length) of each run of chosen moments."""
    runs, start = [], None
    for n, f in enumerate(flags + [False]):
        if f and start is None:
            start = n
        elif not f and start is not None:
            runs.append((start, n - start))
            start = None
    return runs


def describe(res: dict) -> None:
    rows = res["rows"]
    print(f"--- {res['arm']}: how the readings behave")
    print("  each hundred moments: median lean size; median cosine of the lean with rest's direction (spread)")
    for b in range(0, len(rows), 100):
        blk = rows[b:b + 100]
        lean = [r["rest_lean"] for r in blk if "rest_lean" in r]
        cos = [r["rest_cos"] for r in blk if "rest_cos" in r]
        if lean:
            print(f"    {b:5d}: {statistics.median(lean):6.2f}  {statistics.median(cos):+.3f} ({statistics.pstdev(cos):.3f})")
    for name, width in (("rest", 3.0), ("recall", 6.0), ("let_go", 6.0)):
        readings = [r[f"{name}_reading"] for r in rows]
        for window in (256, 64, 32):     # 256 is the built window; the rule replays it exactly
            runs = stretches(classify(readings, window, width))
            longest = max(runs, key=lambda r: r[1]) if runs else (None, 0)
            print(f"  {name:6s} window {window:3d}: read as chosen on {sum(n for _, n in runs):3d} moments in "
                  f"{len(runs)} stretches; longest {longest[1]} from moment {longest[0]}")
    print(f"  memories brought back by rest, looked at on moments: {res['brought_by_rest']}")
    print(f"  memories let go on moments: {res['let_go_at']}")


def main() -> None:
    run = _train_tiny_m9_run(root)
    results = {arm: live(run, arm) for arm in ARMS}
    for arm, res in results.items():
        print(f"{arm:7s} {res['seconds_per_moment'] * 1000:.0f} ms a moment; {res['counts']}")
    for arm in ("off", "on"):
        if arm in results:
            describe(results[arm])
    if "off" in results and "undone" in results:
        same = all(
            a.get("recall_reading") == b.get("recall_reading") and a["resting"] == b["resting"]
            for a, b in zip(results["off"]["rows"], results["undone"]["rows"])
        )
        print(f"'on, undone' matches 'off' moment for moment: {same}")
    if "on" in results:
        res = results["on"]
        rec = recall_moments(res)
        lived = [r["lived_dtheta"] for r in res["rows"] if r["lived_dtheta"]]
        print(f"lived step size, median {statistics.median(lived):.2e}, max {max(lived):.2e}; a dream's step is 1.00e-02")
        print(f"dreams at moments {res['dreams_at']}")
        print(f"recalls at moments {rec}")
        for window in (25, 50, 100):
            soon = sum(any(0 < m - d <= window for d in res["dreams_at"]) for m in rec)
            covered = sum(any(0 < m - d <= window for d in res["dreams_at"]) for m in range(N))
            print(f"  recalls within {window} moments after a dream: {soon}/{len(rec)} "
                  f"(those moments are {covered / N:.0%} of the life)")
        for d in res["dreams_at"]:
            before = [r["recall_reading"] for r in res["rows"][max(0, d - 30):d] if "recall_reading" in r]
            after = [r["recall_reading"] for r in res["rows"][d + 1:d + 31] if "recall_reading" in r]
            lean_b = [r["recall_lean"] for r in res["rows"][max(0, d - 30):d] if "recall_lean" in r]
            lean_a = [r["recall_lean"] for r in res["rows"][d + 1:d + 31] if "recall_lean" in r]
            n_rec = sum(1 for m in rec if d < m <= d + 30)
            if before and after:
                print(f"  dream at {d}: recall reading median {statistics.median(before):+.4f} -> "
                      f"{statistics.median(after):+.4f}; lean {statistics.median(lean_b):.4f} -> "
                      f"{statistics.median(lean_a):.4f}; recalls in the next 30: {n_rec}")
    out = os.environ.get("OUT")
    if out:
        Path(out).write_text(json.dumps({a: {k: v for k, v in r.items()} for a, r in results.items()}))
        print(f"per-moment record: {out}")


if __name__ == "__main__":
    main()
