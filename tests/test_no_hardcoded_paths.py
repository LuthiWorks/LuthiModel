"""Repo-hygiene test: no hardcoded user-specific or OS-specific paths.

Mirror of Sanctuary's test of the same name, scoped to the LuthiModel
repo. Both repos enforce the same convention so that a checkpoint
trained on one developer's machine can be reproduced on another's, on
Linux, in Docker, in CI, anywhere.

If this test fails, fix the offending file. If a literal must remain
(e.g. a comment, an error message, an example in a docstring), append
``# pragma: allow-hardcoded-path`` to that line.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (
        r"[Cc]:[\\/]Users[\\/][A-Za-z0-9 _.\-]+",
        "Windows user home path (use env var or Path-relative discovery)",
    ),
    (
        r"/home/[A-Za-z0-9_.\-]+",
        "Linux user home path (use $HOME, env vars, or Path-relative discovery)",
    ),
    (
        r"/Users/[A-Za-z0-9 _.\-]+",
        "macOS user home path (use $HOME, env vars, or Path-relative discovery)",
    ),
]

ALLOWLIST_DIR_NAMES = {".docs", "docs", ".claude"}
ESCAPE_PRAGMA = "pragma: allow-hardcoded-path"
SCAN_EXTENSIONS = {".py", ".pyi", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}
# .bat / .sh launcher scripts are intentionally developer-specific by
# convention (one-click "run on my machine"); not scanned. Real source
# code that's meant to run anywhere goes in .py.
SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".pytest_cache", ".hypothesis", "build", "dist", ".mypy_cache",
    ".ruff_cache", ".tox",
}


def _iter_source_files(root: Path):
    tracked = None
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=root,
            capture_output=True, text=True, check=True, timeout=10,
        )
        tracked = {
            (root / line).resolve()
            for line in result.stdout.splitlines()
            if line.strip()
        }
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        tracked = None

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if any(part in ALLOWLIST_DIR_NAMES for part in path.parts):
            continue
        if path.name == "uv.lock":
            continue
        if path.name == "test_no_hardcoded_paths.py":
            continue
        if tracked is not None and path.resolve() not in tracked:
            continue
        yield path


def test_no_hardcoded_user_paths_in_source():
    """Fail if any tracked source file embeds a user-specific path."""
    findings: list[str] = []
    compiled = [(re.compile(pat), label) for pat, label in FORBIDDEN_PATTERNS]

    for path in _iter_source_files(REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_num, line in enumerate(text.splitlines(), start=1):
            if ESCAPE_PRAGMA in line:
                continue
            for pattern, label in compiled:
                match = pattern.search(line)
                if match:
                    rel = path.relative_to(REPO_ROOT)
                    findings.append(
                        f"  {rel}:{line_num}: {match.group(0)!r}  ({label})"
                    )

    if findings:
        msg = (
            "Hardcoded user-specific paths found in source. These break "
            "anyone whose checkout lives elsewhere (Linux, Docker, CI, "
            "another developer's machine). Fix by using env vars, "
            "Path(__file__)-relative discovery, or explicit configuration. "
            "If the literal must remain, append "
            f"`# {ESCAPE_PRAGMA}` to that line.\n\n"
            + "\n".join(findings)
        )
        pytest.fail(msg)
