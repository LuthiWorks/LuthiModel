"""Repo hygiene for launch scripts: no committed credentials, no user names.

This repository is public. `test_no_hardcoded_paths` deliberately skips
launch scripts (.bat/.sh may be machine-specific), and two things leaked
through exactly there:

* a checkpoint password passed literally as ``--checkpoint_password``
  (``train_vision.bat`` until 2026-07-22, ``generate.bat`` until
  2026-10-19);
* a Windows user-home path, which publishes the account name
  (``generate.bat``, ``run_m8.bat`` until 2026-10-19).

Launchers may still name machine paths such as ``E:\\runs`` or
``C:\\Dev\\rocm-probe``. A password comes from ``LUTHI_CHECKPOINT_KEY`` (or no
password at all: plaintext copies from scripts/decrypt_checkpoints.py), and a
home folder comes from ``%USERPROFILE%`` / ``$HOME``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_SUFFIXES = {".bat", ".cmd", ".ps1", ".sh"}

# A value is fine when it is a variable reference: %VAR%, $VAR, ${VAR},
# $(command), or $env:VAR.
_VARIABLE = r"[\"']?(?:%\w+%|\$\{?\w+\}?|\$\(|\$env:\w+)"
LITERAL_PASSWORD = re.compile(
    r"--checkpoint[_-]password(?:\s+|=)(?!" + _VARIABLE + r")\S+", re.IGNORECASE
)
LITERAL_KEY = re.compile(
    r"LUTHI_CHECKPOINT_KEY\s*=\s*(?!" + _VARIABLE + r"|[\"']?\s*$)\S", re.IGNORECASE
)
USER_HOME = re.compile(
    r"[Cc]:[\\/]Users[\\/][A-Za-z0-9 _.\-]+|/home/[A-Za-z0-9_.\-]+|/Users/[A-Za-z0-9 _.\-]+"
)


def _launchers() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.splitlines()
        paths = [REPO_ROOT / line for line in out if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        paths = [p for p in REPO_ROOT.rglob("*") if ".git" not in p.parts]
    return [p for p in paths if p.suffix.lower() in LAUNCHER_SUFFIXES and p.is_file()]


def _scan(pattern: re.Pattern) -> list[str]:
    hits = []
    for path in _launchers():
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"  {path.relative_to(REPO_ROOT)}:{n}")
    return hits


def test_there_are_launchers_to_check():
    assert _launchers(), "no tracked launch scripts found; is git available?"


@pytest.mark.parametrize("label,pattern", [
    ("a literal --checkpoint_password value", LITERAL_PASSWORD),
    ("a literal LUTHI_CHECKPOINT_KEY assignment", LITERAL_KEY),
])
def test_no_literal_checkpoint_password(label, pattern):
    hits = _scan(pattern)
    assert not hits, (
        f"Launch script(s) contain {label}. This repo is public: read the "
        "password from LUTHI_CHECKPOINT_KEY instead, or load a plaintext "
        "copy made by scripts/decrypt_checkpoints.py.\n" + "\n".join(hits)
    )


def test_no_user_home_path_in_launchers():
    hits = _scan(USER_HOME)
    assert not hits, (
        "Launch script(s) embed a user-home path, which publishes the account "
        "name. Use %USERPROFILE% / $HOME, or %~dp0 for the script's own "
        "folder.\n" + "\n".join(hits)
    )


@pytest.mark.parametrize("line,bad", [
    ('python -m luthi.generate --checkpoint_password "hunter2"', True),
    ("python x.py --checkpoint_password=hunter2", True),
    ('python x.py --checkpoint_password "%LUTHI_CHECKPOINT_KEY%"', False),
    ("python x.py --checkpoint_password $LUTHI_CHECKPOINT_KEY", False),
    ("python x.py --checkpoint_password ${LUTHI_CHECKPOINT_KEY}", False),
])
def test_password_pattern_itself(line, bad):
    assert bool(LITERAL_PASSWORD.search(line)) is bad


@pytest.mark.parametrize("line,bad", [
    ('set "LUTHI_CHECKPOINT_KEY=hunter2"', True),
    ("export LUTHI_CHECKPOINT_KEY=hunter2", True),
    ('$env:LUTHI_CHECKPOINT_KEY = "hunter2"', True),
    ('if "%LUTHI_CHECKPOINT_KEY%"=="" echo unset', False),
    ("export LUTHI_CHECKPOINT_KEY=$(pass show luthi)", False),
    ('set "LUTHI_CHECKPOINT_KEY="', False),
])
def test_key_pattern_itself(line, bad):
    assert bool(LITERAL_KEY.search(line)) is bad
