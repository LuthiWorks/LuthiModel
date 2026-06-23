"""Tests for the LuthiScope run-registry handshake (luthi/v2/run_registry.py).

The registry is how a live trainer tells LuthiScope where it is logging. Two
properties matter most and are tested hardest:

  1. **Best-effort isolation.** No public call may ever raise -- a broken or
     unwritable registry must degrade to "console didn't see the run," never to
     "training crashed writing a status file." The contract is load-bearing:
     registry I/O sits inside a real training process.
  2. **Liveness check is read-only.** ``_pid_alive`` must never signal the
     process it inspects (on Windows ``os.kill`` would *terminate* it). We assert
     it reports our own live pid and a surely-dead pid correctly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from luthi.v2 import run_registry as rr


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Point the registry at a temp file via the documented env override."""
    path = tmp_path / "runs.json"
    monkeypatch.setenv("LUTHISCOPE_REGISTRY", str(path))
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_register_then_unregister_roundtrip(registry, tmp_path):
    run_dir = tmp_path / "runs" / "m8_x"
    run_dir.mkdir(parents=True)
    key = str(run_dir.resolve())

    rr.register_run(run_dir, started_at=123.0)
    data = _read(registry)
    assert key in data
    assert data[key]["pid"] == os.getpid()
    assert data[key]["started_at"] == 123.0

    rr.unregister_run(run_dir)
    assert key not in _read(registry)


def test_register_creates_parent_dir(tmp_path, monkeypatch):
    # The ~/.luthiscope dir typically does not exist yet on first run.
    nested = tmp_path / "deep" / "nested" / "runs.json"
    monkeypatch.setenv("LUTHISCOPE_REGISTRY", str(nested))
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    rr.register_run(run_dir, started_at=1.0)
    assert nested.exists()
    assert str(run_dir.resolve()) in _read(nested)


def test_dead_pid_entries_are_pruned(registry, tmp_path):
    # A crashed run that never unregistered leaves a stale entry; the next
    # writer prunes it. PID 999999 is effectively never a live process.
    stale_dir = str((tmp_path / "crashed").resolve())
    registry.write_text(
        json.dumps({stale_dir: {"pid": 999999, "started_at": 1.0}}), encoding="utf-8"
    )
    live = tmp_path / "live"
    live.mkdir()
    rr.register_run(live, started_at=2.0)
    data = _read(registry)
    assert stale_dir not in data           # pruned
    assert str(live.resolve()) in data     # ours survives (live pid)


def test_unregister_missing_key_is_noop(registry, tmp_path):
    # Early-exit path: run dir never registered (e.g. missing-input return).
    run_dir = tmp_path / "never"
    rr.unregister_run(run_dir)             # must not raise
    # File may or may not exist; if it does, it's a valid empty object.
    if registry.exists():
        assert _read(registry) == {}


def test_malformed_registry_is_tolerated(registry, tmp_path):
    registry.write_text("{ this is not json", encoding="utf-8")
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    rr.register_run(run_dir, started_at=1.0)   # must recover, not raise
    assert str(run_dir.resolve()) in _read(registry)


def test_non_object_registry_is_tolerated(registry, tmp_path):
    registry.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON, wrong shape
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    rr.register_run(run_dir, started_at=1.0)
    assert _read(registry) == {str(run_dir.resolve()): {
        "pid": os.getpid(), "started_at": 1.0
    }}


def test_calls_never_raise_even_when_path_is_unwritable(monkeypatch, tmp_path):
    # Point the registry at a path whose parent is a *file*, so mkdir/replace
    # cannot succeed. The functions must still return quietly.
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    bad = afile / "runs.json"  # parent is a file -> mkdir fails
    monkeypatch.setenv("LUTHISCOPE_REGISTRY", str(bad))
    rr.register_run(tmp_path / "r", started_at=1.0)   # no exception
    rr.unregister_run(tmp_path / "r")                 # no exception


def test_pid_alive_is_readonly_and_correct():
    # Our own process is alive...
    assert rr._pid_alive(os.getpid()) is True
    # ...and a surely-dead / invalid pid is not. Critically, calling this must
    # not have signalled or terminated anything (we're still running to assert).
    assert rr._pid_alive(999999) is False
    assert rr._pid_alive(-1) is False
    assert rr._pid_alive(None) is False
    assert rr._pid_alive("nonsense") is False


def test_pid_alive_rejects_non_int_pids():
    # A float would truncate to a real unrelated pid; bool is an int subclass
    # (True -> 1). Both must be rejected outright, not coerced.
    assert rr._pid_alive(3.7) is False
    assert rr._pid_alive(float(os.getpid())) is False  # even our own, as a float
    assert rr._pid_alive(True) is False
    assert rr._pid_alive(False) is False


def test_non_finite_started_at_is_coerced_finite(registry, tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    rr.register_run(run_dir, started_at=float("nan"))
    val = _read(registry)[str(run_dir.resolve())]["started_at"]
    import math as _m
    assert _m.isfinite(val)  # not NaN/Infinity -> standard JSON round-trips


def test_no_tmp_leak_when_serialization_fails(registry, tmp_path, monkeypatch):
    # Force the write to fail with a non-OSError mid-dump. register_run must
    # still not raise, and must leave no orphan .tmp file behind.
    def boom(*a, **k):
        raise TypeError("non-serializable")

    monkeypatch.setattr(rr.json, "dump", boom)
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    rr.register_run(run_dir, started_at=1.0)  # must not raise
    leftovers = list(registry.parent.glob("*.tmp"))
    assert leftovers == [], f"orphan temp files left behind: {leftovers}"
