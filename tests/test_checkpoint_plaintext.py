"""Encryption is optional (2026-10-19): plaintext checkpoints load with no
password, and an encrypted checkpoint can be given a plaintext copy without
its original ever being touched.

Pins `luthi.checkpoint.load_checkpoint`'s plaintext detection,
`write_plaintext_copy`, and `scripts/decrypt_checkpoints.py` end to end.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from luthi.checkpoint import (  # noqa: E402
    is_plaintext_torch,
    load_checkpoint,
    plaintext_copy_path,
    save_checkpoint,
    write_plaintext_copy,
)

PW = "test-password-do-not-use-in-production"
PAYLOAD = {"epoch": 3, "w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    monkeypatch.delenv("LUTHI_CHECKPOINT_KEY", raising=False)


@pytest.fixture
def encrypted(tmp_path) -> Path:
    return save_checkpoint(dict(PAYLOAD), tmp_path / "run" / "model_final", password=PW)


def _same(a: dict, b: dict) -> bool:
    return a["epoch"] == b["epoch"] and torch.equal(a["w"], b["w"])


def test_plaintext_file_loads_without_a_password(tmp_path):
    path = tmp_path / "plain.pt"
    torch.save(dict(PAYLOAD), path)
    assert is_plaintext_torch(path.read_bytes())
    assert _same(load_checkpoint(path), PAYLOAD)


def test_encrypted_file_still_needs_its_password(encrypted):
    assert not is_plaintext_torch(encrypted.read_bytes())
    with pytest.raises(ValueError, match="decrypt_checkpoints"):
        load_checkpoint(encrypted)
    assert _same(load_checkpoint(encrypted, PW), PAYLOAD)


def test_plaintext_copy_round_trip_leaves_the_original_alone(encrypted):
    before = _sha(encrypted)
    rec = write_plaintext_copy(encrypted, [("wrong", "nope"), ("right", PW)])
    assert rec["status"] == "written"
    assert rec["password_label"] == "right"
    assert rec["payload_is_torch"] is True
    assert _sha(encrypted) == before, "the original must never change"

    copy = plaintext_copy_path(encrypted)
    assert copy.name == "model_final.decrypted.pt"
    assert rec["copy_sha256"] == _sha(copy)
    assert _same(load_checkpoint(copy), PAYLOAD)  # no password anywhere

    sidecar = copy.with_suffix(".json")
    text = sidecar.read_text(encoding="utf-8")
    assert PW not in text, "a password must never be written down"
    assert json.loads(text)["source_sha256"] == before


def test_second_run_skips_and_touches_nothing(encrypted):
    write_plaintext_copy(encrypted, [("right", PW)])
    copy = plaintext_copy_path(encrypted)
    stamp = (_sha(copy), copy.stat().st_mtime_ns)
    rec = write_plaintext_copy(encrypted, [("right", PW)])
    assert rec["status"] == "exists"
    assert (_sha(copy), copy.stat().st_mtime_ns) == stamp


def test_no_matching_password_writes_nothing(encrypted):
    rec = write_plaintext_copy(encrypted, [("a", "nope"), ("b", "also-nope")])
    assert rec["status"] == "no_password_matched"
    assert "password_label" not in rec
    assert not plaintext_copy_path(encrypted).exists()


def test_plaintext_named_luthi_is_left_alone(tmp_path):
    path = tmp_path / "old_fallback.luthi"
    torch.save(dict(PAYLOAD), path)
    rec = write_plaintext_copy(path, [("right", PW)])
    assert rec["status"] == "already_plaintext"
    assert not plaintext_copy_path(path).exists()


def test_structural_damage_is_not_mistaken_for_a_wrong_password(encrypted, tmp_path):
    damaged = tmp_path / "damaged.luthi"
    damaged.write_bytes(encrypted.read_bytes()[:-7])
    with pytest.raises(ValueError, match="truncated"):
        write_plaintext_copy(damaged, [("right", PW)])
    assert not plaintext_copy_path(damaged).exists()


def test_script_end_to_end(encrypted, tmp_path, capsys):
    script = importlib.import_module("scripts.decrypt_checkpoints")
    root = str(tmp_path)

    assert script.main([root, "--password", "nope"]) == 1
    assert not plaintext_copy_path(encrypted).exists()

    assert script.main([root, "--password", "nope", "--password", PW]) == 0
    assert _same(load_checkpoint(plaintext_copy_path(encrypted)), PAYLOAD)
    assert PW not in capsys.readouterr().out
