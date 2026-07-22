"""Repo-hygiene test: all tests live in tests/, and only in tests/.

Brian's ruling (2026-07-22), same practice as the storage policy and the
hardcoded-path guard: standards are documented in CLAUDE.md *and* enforced
by a test, so they cannot be quietly forgotten. Test files scattered
through the source tree don't run under the standard command
(``python -m pytest tests/`` — pyproject pins ``testpaths = ["tests"]``),
which is exactly how the 19 m9 tests silently fell out of the suite
before 2026-07-22.

If this test fails, ``git mv`` the offending file into ``tests/``
(subfolders are fine — e.g. ``tests/m9/``) and fix its imports. Red-team
probes in ``redteam/`` are not tests (different naming, different
purpose) and are not scanned.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_FILE_PATTERN = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True, timeout=10,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_all_tests_live_in_tests_folder():
    """Fail if any tracked test file lives outside tests/."""
    strays = [
        path for path in _tracked_files()
        if TEST_FILE_PATTERN.search(path) and not path.startswith("tests/")
    ]

    if strays:
        pytest.fail(
            "Test files found outside tests/. They will not run under the "
            "standard suite (pyproject testpaths = ['tests']). git mv each "
            "into tests/ (subfolders fine) and fix imports:\n\n  "
            + "\n  ".join(strays)
        )
