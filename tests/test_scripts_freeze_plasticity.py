"""Repo-hygiene test: measurement scripts must not rewrite what they measure.

A living layer has no ``self.training`` gate -- perception is never stateless
-- so ``model.eval()`` + ``torch.no_grad()`` does NOT stop
``pc_self_modify`` or the block-level episode store from writing. Until
2026-10-19 six forensic scripts in ``scripts/`` (per_block_probe,
soloist_forensic, measure_input_sensitivity, measure_offset_by_block,
localize_offset_in_block, depth_ladder_probe) measured checkpoints that
self-modified between, and during, their own measurement batches. The
registered evaluator (``luthi/v2/eval_heldout.py``) had always frozen; the
forensic instruments had not.

Same practice as the tests-live-in-tests and hardcoded-path guards: the
standard is enforced by a test so it cannot be quietly forgotten. Any
script that runs a model encode or calls a block directly must reference
``freeze_plasticity`` -- or appear in ``ALLOWED_LIVE`` with the reason it
measures live dynamics on purpose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Model forwards only (tokenizer.encode(text) does not match these).
FORWARD = re.compile(r"\.encode\(\s*(text_tokens|\*\*)|\bblock\(h\b|model\.encode\(")

# script name -> why it measures LIVE dynamics deliberately
ALLOWED_LIVE: dict[str, str] = {}


def test_measurement_scripts_freeze_plasticity():
    offenders = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not FORWARD.search(text):
            continue
        if path.name in ALLOWED_LIVE:
            continue
        if "freeze_plasticity" not in text:
            offenders.append(path.name)
    if offenders:
        pytest.fail(
            "Scripts run model forwards without freeze_plasticity; eval() "
            "does not stop living layers from self-modifying, so these "
            "measure a model that changes under the measurement. Wrap the "
            "forwards in luthi.v2.plasticity.freeze_plasticity(model), or add "
            "the script to ALLOWED_LIVE with the reason it must run live:\n  "
            + "\n  ".join(offenders)
        )
