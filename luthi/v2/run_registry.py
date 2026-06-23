"""Announce a live training run to LuthiScope (the observation console).

LuthiScope discovers runs two ways: by scanning a configured runs dir, and by
reading this registry -- a tiny JSON file an active trainer writes its run dir
to. The registry lets a run logging *anywhere* on disk be found with zero config
on the LuthiScope side, and badged "live" while it is writing.

Contract (see LuthiScope docs/RUN_REGISTRY.md):
  - File at ``~/.luthiscope/runs.json`` (override: ``$LUTHISCOPE_REGISTRY``).
  - A JSON object keyed by absolute run-dir path; value = ``{"pid", "started_at"}``.
  - LuthiScope *only reads* it. We are the sole writer. The same read-only
    discipline the metric logs follow: the trainer writes only its own files
    (logs + this announcement); the console never writes back.

Design rule, non-negotiable: **registry I/O must never crash training.** Every
public function swallows its own errors -- a missing dir, a locked file, a
malformed registry, a disk hiccup -- and returns quietly. The console degrades
to "didn't see the run," never the trainer to "died writing a status file."
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

try:  # filelock is in the lock tree; if it's somehow absent we degrade to no lock
    from filelock import FileLock, Timeout
    _HAVE_LOCK = True
except Exception:  # pragma: no cover - defensive
    _HAVE_LOCK = False

_LOCK_TIMEOUT_S = 5.0


def _registry_path() -> Path:
    env = os.environ.get("LUTHISCOPE_REGISTRY")
    return Path(env).expanduser() if env else Path.home() / ".luthiscope" / "runs.json"


def _pid_alive(pid) -> bool:
    """True if a process with ``pid`` currently exists. Read-only on both OSes.

    Critically, this must NOT signal the process. On POSIX, ``os.kill(pid, 0)``
    is the canonical existence probe (signal 0 is never delivered). On Windows,
    ``os.kill`` with any non-CTRL signal calls ``TerminateProcess`` -- it would
    *kill* the very run we are checking -- so we query a handle instead.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k = ctypes.windll.kernel32
        handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False  # no such process (or access denied -> treat as gone)
        try:
            code = wintypes.DWORD()
            if not k.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, just not ours to signal
        except OSError:
            return False


def _rewrite(mutate) -> None:
    """Read the registry, prune dead-pid entries, apply ``mutate``, write atomically.

    Locked (best-effort) so two runs starting at once don't lose each other's
    update. The write goes to a temp file in the same dir then ``os.replace`` --
    atomic, so a reader never sees a half-written file. Any failure is swallowed.
    """
    path = _registry_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    lock_cm = None
    if _HAVE_LOCK:
        try:
            lock_cm = FileLock(str(path) + ".lock", timeout=_LOCK_TIMEOUT_S)
            lock_cm.acquire()
        except Timeout:
            lock_cm = None  # proceed unlocked rather than block training
        except Exception:
            lock_cm = None
    try:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}

        # Prune entries whose process is gone (crash / kill -9 that skipped exit).
        data = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and _pid_alive(v.get("pid"))
        }

        mutate(data)

        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    finally:
        if lock_cm is not None:
            try:
                lock_cm.release()
            except Exception:
                pass


def register_run(run_dir, started_at: float) -> None:
    """Announce this run. Best-effort; never raises. Call once at run start."""
    try:
        key = str(Path(run_dir).resolve())
        _rewrite(lambda d: d.__setitem__(
            key, {"pid": os.getpid(), "started_at": float(started_at)}
        ))
    except Exception:  # pragma: no cover - the whole point is to never escape
        pass


def unregister_run(run_dir) -> None:
    """Remove this run's announcement. Best-effort; never raises. Call in finally."""
    try:
        key = str(Path(run_dir).resolve())
        _rewrite(lambda d: d.pop(key, None))
    except Exception:  # pragma: no cover
        pass
