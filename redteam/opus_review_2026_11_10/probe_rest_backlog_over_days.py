"""Follow-up probe (Opus 5.5, 2026-11-10): with folding off, what happens to
what rest's bound holds back, day after day?

Run from the LuthiModel root, with Sanctuary importable (a few minutes on CPU):

    PYTHONPATH=<Sanctuary>:<LuthiModel> LUTHI_PATH=<LuthiModel> \
        python redteam/opus_review_2026_11_10/probe_rest_backlog_over_days.py

Same small trained trunk and world as ``probe_rest_on_a_lived_day.py``. Five
days of 300 moments each, every moment admitted, a default rest after each
day, nothing judged.

Printed on 2026-11-10, at LuthiModel 4c99fb1 (block 1; block 0 integrated
nearly everything):

    day 1: considered 300, integrated 194, waiting 106, oldest waiting = the first moment lived; home +5.0%
    day 2: considered 406, integrated 187, waiting 219, oldest waiting = the first moment lived; home +5.0%
    day 3: considered 519, integrated 119, waiting 400, oldest waiting = the first moment lived; home +5.0%
    day 4: considered 700, integrated 145, waiting 555, oldest waiting = the first moment lived; home +5.0%
    day 5: considered 855, integrated 186, waiting 669, oldest waiting = the first moment lived; home +5.0%

Over the same five rests, block 1's precision (not put back by rest) moved
0.2%, 5.8%, 6.9%, 7.4% and 9.1%, and its prediction 3.0% to 5.9%.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_rest_on_a_lived_day as lived  # noqa: E402  (its corpus, training and temp root)
import torch  # noqa: E402
from luthi.v2.day_record import day_records, enable_day_record  # noqa: E402
from luthi.v2.rest import rest  # noqa: E402
from sanctuary.life import LifeLoop, LuthiMind, TextWorld  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    mind = LuthiMind(lived.train(), lived.ROOT / "life")
    world = TextWorld()
    loop = LifeLoop(mind, world, hz=0.0)
    layers = {p: m for p, m in mind.model.named_modules() if hasattr(m, "_day_record")}
    for m in layers.values():
        m.salience_threshold = 0.0
    enable_day_record(mind.model)
    first = None
    for day in range(1, 6):
        for i in range(300):
            if i % 3 == 0:
                world.say(f"day {day}: a zebra walked past, number {i}" if i % 2
                          else f"day {day}: the garden is quiet", source="brian")
            loop.step()
        if first is None:
            first = min(e.moment for r in day_records(mind.model).values() for e in r)
        report = rest(mind.model, keep=[], toss=[], unjudged="keep")
        for path, lr in report.layers.items():
            record = layers[path]._day_record
            oldest = min(record.moments()) if len(record) else None
            print(f"day {day} {path}: considered {lr.taken}, integrated {len(lr.integrated)}, "
                  f"waiting {len(record)}, oldest waiting {oldest} (first lived {first}); "
                  f"home moved {lr.home_change:.3f}; also moved "
                  + ", ".join(f"{k} {v:.3f}" for k, v in lr.also_moved.items()), flush=True)


if __name__ == "__main__":
    main()
