"""Review probes (Opus 5.5, 2026-11-10) for Sanctuary's small-path loop, with the
test suite's stand-in mind. No luthi needed.

Run from the LuthiModel root, with Sanctuary importable:

    PYTHONPATH=<Sanctuary> python -m pytest redteam/opus_review_2026_11_10/probe_small_path_loop.py -s -p no:cacheprovider

Printed on 2026-11-10:

- clean stop with 753 tokens still waiting: saved, ``dropped`` 0, nothing
  logged. The line is not saved with the life, so after a restart those tokens
  are gone. A rollback notice still in the line at a stop goes the same way,
  and after the restart it counts as already told.
- after a moment raised, ``loop.save()`` succeeded. Nothing in the loop stops
  a caller from saving a half-finished moment; only the callers' discipline
  does (today, ``run()`` and ``__main__`` both get it right).
"""

from __future__ import annotations

import logging

import pytest

from sanctuary.tests.life.test_loop import _loop


def test_what_is_waiting_at_a_clean_stop(caplog):
    loop, mind, world = _loop()
    world.say("x" * 1000, source="b")          # about four windows of 256
    with caplog.at_level(logging.WARNING):
        loop.run(max_cycles=1)
    print("\nafter a clean stop:", loop.status()["world"], "saved:", mind.saved,
          "logged:", [r.message for r in caplog.records])


def test_a_save_after_a_raise_is_not_refused():
    loop, mind, _ = _loop()
    mind.fail_at = 1
    with pytest.raises(RuntimeError):
        loop.run(max_cycles=3)
    loop.save("after the raise")
    print("\nsaved after a raise:", mind.saved)
