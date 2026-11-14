"""Three things about Sanctuary's memories that only show across moments or
restarts. Opus 5.5 review of the 2026-11-11 work (Fable's brief, points 5 and 6).

Uses the loop tests' stand-in mind (``sanctuary/tests/life/test_loop.py``), so
no model is needed.

1. Recall after a restart. ``LifeLoop.cycles`` starts at 0 on every start and
   is not saved, while every memory row carries the cycle it was written at. A
   memory counts as "still the present" while ``now - cycle < recent``, so after
   a restart every memory from before it is "still the present" until the new
   count passes the old one.
2. Letting go and keeping in one moment. The moment that looks back on a memory
   can let it go and, by the memory channel, keep what it perceives: the
   memory's own words, picture and sound.
3. A crash between a memory's file and its line in ``summaries.jsonl``. The next
   start counts kept memories from the lines, so the next keep picks the id
   whose file already exists, and refuses.

Run:  SANCTUARY_PATH=../sanctuary python redteam/opus_review_2026_11_14/probe_memories_across_a_restart.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["SANCTUARY_PATH"])

import torch

import sanctuary.life.memory as memory_module
from sanctuary.core.atomic_io import PersistenceError
from sanctuary.life import LifeLoop, TextWorld
from sanctuary.life.memory import Memories
from sanctuary.tests.life.test_loop import _Mind, _text


def remember_for(loop: LifeLoop, mind: _Mind, moments: int) -> None:
    mind.active = {"text": False, "attention": False, "memory": True}
    for _ in range(moments):
        loop.step()
    mind.active = {"text": False, "attention": False, "memory": False}


def live(loop: LifeLoop, moments: int) -> None:
    for _ in range(moments):
        loop.step()


def first_recall(loop: LifeLoop, mind: _Mind, limit: int) -> int | None:
    """Choose recall every moment until something comes back; return how many
    moments that took, or None."""
    mind.chooses_recall = True
    try:
        for n in range(limit):
            before = loop.recalled
            loop.step()
            if loop.recalled > before:
                return n
        return None
    finally:
        mind.chooses_recall = False


def part_1() -> None:
    print("1. Recall after a restart (recent = 100, the default)")
    for restart in (False, True):
        mind = _Mind()
        loop = LifeLoop(mind, TextWorld(), hz=0.0)
        live(loop, 200)
        remember_for(loop, mind, 50)          # kept at cycles 200 to 249
        live(loop, 150)                       # now 400: all of them are long past
        loop.save("stop")
        if restart:
            loop = LifeLoop(mind, TextWorld(), hz=0.0)   # the same life, started again
        took = first_recall(loop, mind, 1000)
        print(f"   {'after a restart' if restart else 'same session  '}: cycles now {loop.cycles - (took or 0) - 1:4d}; "
              f"{len(loop.memories)} memories held; first recall came back after "
              f"{took if took is not None else 'more than 1000'} moments of choosing it")


def part_2() -> None:
    print("2. Letting go and keeping in the same moment")
    mind = _Mind()
    loop = LifeLoop(mind, TextWorld(), hz=0.0)
    loop.memories.recent = 2
    loop.world.say("the kettle is on")
    remember_for(loop, mind, 1)
    live(loop, 3)
    mind.chooses_recall = True
    loop.step()                               # chooses to look back: memory 1 comes back next moment
    mind.chooses_recall = False
    mind.chooses_let_go = True
    mind.active = {"text": False, "attention": False, "memory": True}
    loop.step()                               # looking at it, let it go, and keep this moment
    mind.chooses_let_go = False
    report = loop.memories.report()
    newest = loop.memories.open(loop.memories.kept)
    print(f"   let go: {report['let_go']}; held: {report['held']}; "
          f"the memory kept in that moment holds the let-go words: {'the kettle is on' in _text(newest.tokens)}")


def part_3() -> None:
    print("3. A crash between a memory's file and its line")
    mind = _Mind()
    store = Memories(mind.life_dir / "memories")
    store.keep(cycle=1, summary=torch.ones(3), tokens=[1, 2, 3])
    real_append = memory_module.append_jsonl

    def crash(*args, **kwargs):
        raise PersistenceError("simulated crash before the line was written")

    memory_module.append_jsonl = crash
    try:
        store.keep(cycle=2, summary=torch.ones(3), tokens=[4, 5, 6])
    except PersistenceError as e:
        print(f"   second keep crashed after its file was written: {e}")
    finally:
        memory_module.append_jsonl = real_append
    files = sorted(p.name for p in store.directory.glob("memory_*.pt"))
    again = Memories(mind.life_dir / "memories")
    print(f"   on disk: {files}; the restarted store counts {again.kept} kept")
    try:
        again.keep(cycle=3, summary=torch.ones(3), tokens=[7, 8, 9])
        print("   the next keep worked")
    except PersistenceError as e:
        print(f"   the next keep refuses, and every keep after it will: {e}")


if __name__ == "__main__":
    part_1()
    part_2()
    part_3()
