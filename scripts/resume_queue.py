"""Queue supervisor: restart the active experiment queue from wherever it
left off. Safe to run at ANY time -- completed work is skipped, an
interrupted seed resumes from its latest rolling checkpoint (<=15 min
lost), and a second copy of this script exits immediately.

Task Scheduler runs this at logon and every 30 minutes (see To-Do.md's
operational-queue block), so shutdowns, terminal closures, and power
loss all self-heal without anyone remembering a command. Running it by
hand is also always safe:

    python scripts/resume_queue.py

The queue itself lives in runs/jepa_pilot/queue.json (data, not code --
edit the JSON to change the plan). All output appends to
runs/jepa_pilot/supervisor.log, the witness when no console exists
(the LuthiScope desktop.log lesson, 2026-07-19).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = REPO_ROOT / "runs" / "jepa_pilot" / "queue.json"
LOG_PATH = REPO_ROOT / "runs" / "jepa_pilot" / "supervisor.log"

# Single-instance mutex: bind a fixed localhost port for the lifetime of
# the supervisor (and hold it while child drivers run). A port cannot go
# stale the way a lockfile can -- the OS releases it the instant the
# process dies, which is exactly the crash-safety this exists for.
MUTEX_PORT = 8877


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {msg}"
    print(line)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def main() -> int:
    mutex = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        mutex.bind(("127.0.0.1", MUTEX_PORT))
    except OSError:
        # Another supervisor (or a driver launched by one) owns the
        # queue. Exiting silently is the correct behavior -- this is
        # the watchdog finding everything already alive.
        return 0

    if not QUEUE_PATH.is_file():
        _log("no queue.json -- nothing to do")
        return 0
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    stages = queue.get("stages", [])
    if not stages:
        _log("queue.json has no stages -- nothing to do")
        return 0

    _log(f"supervisor start: {len(stages)} stage(s) in queue")
    # Which interpreter drives the runs is the single fact that decides the
    # backend (the ROCm stack lives in its own environment). Log it so a
    # queue that was launched from the wrong python is diagnosable from
    # supervisor.log alone. The driver enforces: with LUTHI_BACKEND unset it
    # declares ROCm (2026-08-21 default) and refuses to start anywhere else.
    _log(f"interpreter: {sys.executable}  "
         f"LUTHI_BACKEND={os.environ.get('LUTHI_BACKEND') or '<unset -> rocm>'}")
    for entry in stages:
        cmd = [
            sys.executable,
            "-u",  # unbuffered: driver output reaches the log live
            str(REPO_ROOT / "scripts" / "jepa_pilot_driver.py"),
            "--stage", str(entry["stage"]),
        ]
        if "n_seeds" in entry:
            cmd += ["--n-seeds", str(entry["n_seeds"])]
        if "seeds" in entry:
            cmd += ["--seeds", str(entry["seeds"])]
        _log(f"stage {entry['stage']}: {entry.get('note', '')} -> {' '.join(cmd[1:])}")
        with open(LOG_PATH, "a", encoding="utf-8") as logf:
            rc = subprocess.run(
                cmd, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT,
            ).returncode
        if rc != 0:
            _log(f"stage {entry['stage']} exited rc={rc} -- stopping the "
                 f"chain (driver output above has the reason)")
            return rc
        _log(f"stage {entry['stage']} complete")
    _log("queue complete -- all stages done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
