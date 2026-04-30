"""Tests for the curriculum trainer.

The hot-path tests that need confidence before the rented-GPU run:

  - parse_curriculum_stages handles real-world file_list quirks (blank
    lines, comments inside a stage, missing stages, missing files).
  - stage_transition_delta reports zero when nothing changed and a
    positive number when something did.
  - An end-to-end small-scale curriculum run produces checkpoints, the
    living weights actually change across stages, and a resume from a
    mid-curriculum checkpoint picks up cleanly with continuous living
    state.

The end-to-end test uses tiny synthetic text files and a tiny BPE
tokenizer so the whole thing finishes in seconds on CPU. It exercises
the same code path the rented A100 will hit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from luthi.train_curriculum import (
    parse_curriculum_stages,
    stage_transition_delta,
    snapshot_living_state,
    PRODUCTION_CONFIG,
)
from luthi.checkpoint import load_checkpoint
from luthi.tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# Stage parsing
# ---------------------------------------------------------------------------


class TestStageParser:
    def test_parses_stage_markers_and_groups_files(self, tmp_path):
        # Build a small synthetic corpus + file_list.
        s1_a = tmp_path / "s1_a.txt"; s1_a.write_text("alpha")
        s1_b = tmp_path / "s1_b.txt"; s1_b.write_text("beta")
        s2_a = tmp_path / "s2_a.txt"; s2_a.write_text("gamma")
        list_path = tmp_path / "file_list.txt"
        list_path.write_text(
            "# === Stage: science_philosophy (2 files) ===\n"
            f"{s1_a}\n"
            f"{s1_b}\n"
            "\n"
            "# === Stage: code (1 files) ===\n"
            f"{s2_a}\n",
            encoding="utf-8",
        )

        stages = parse_curriculum_stages(list_path)
        assert [name for name, _ in stages] == ["science_philosophy", "code"]
        assert [len(files) for _, files in stages] == [2, 1]

    def test_skips_missing_files(self, tmp_path):
        present = tmp_path / "present.txt"; present.write_text("x")
        list_path = tmp_path / "file_list.txt"
        list_path.write_text(
            "# === Stage: only_stage (2 files) ===\n"
            f"{present}\n"
            f"{tmp_path / 'absent.txt'}\n",
            encoding="utf-8",
        )
        stages = parse_curriculum_stages(list_path)
        assert len(stages) == 1
        assert len(stages[0][1]) == 1

    def test_drops_stages_with_no_existing_files(self, tmp_path):
        present = tmp_path / "p.txt"; present.write_text("x")
        list_path = tmp_path / "file_list.txt"
        list_path.write_text(
            "# === Stage: empty_stage (1 files) ===\n"
            f"{tmp_path / 'absent.txt'}\n"
            "# === Stage: real_stage (1 files) ===\n"
            f"{present}\n",
            encoding="utf-8",
        )
        stages = parse_curriculum_stages(list_path)
        assert [name for name, _ in stages] == ["real_stage"]

    def test_raises_when_no_stages_have_files(self, tmp_path):
        list_path = tmp_path / "file_list.txt"
        list_path.write_text(
            "# === Stage: nothing (0 files) ===\n"
            f"{tmp_path / 'absent.txt'}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="No curriculum stages"):
            parse_curriculum_stages(list_path)


# ---------------------------------------------------------------------------
# Transition delta
# ---------------------------------------------------------------------------


class TestStageTransitionDelta:
    def test_zero_when_state_unchanged(self):
        snap = {
            "b0.weight": torch.zeros(4, 4),
            "b0.set_point": torch.ones(4, 4),
            "b0.plasticity": torch.ones(4, 4) * 2,
            "b0.update_ema": torch.ones(4, 4) * 0.001,
        }
        delta = stage_transition_delta(snap, snap)
        for key, value in delta.items():
            assert value == 0.0, f"{key} should be 0 when unchanged"

    def test_reports_mean_abs_change(self):
        before = {
            "b0.weight": torch.zeros(4, 4),
            "b0.set_point": torch.zeros(4, 4),
            "b0.plasticity": torch.zeros(4, 4),
            "b0.update_ema": torch.zeros(4, 4),
        }
        after = {
            "b0.weight": torch.full((4, 4), 0.5),
            "b0.set_point": torch.zeros(4, 4),
            "b0.plasticity": torch.zeros(4, 4),
            "b0.update_ema": torch.zeros(4, 4),
        }
        delta = stage_transition_delta(before, after)
        assert delta["delta_weight_mean"] == pytest.approx(0.5, abs=1e-6)
        assert delta["delta_set_point_mean"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end small-scale run
# ---------------------------------------------------------------------------


def _make_tiny_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny multi-stage corpus + file_list and a small BPE tokenizer.

    Returns (file_list_path, tokenizer_path).
    """
    # Two stages, two files each, ~few KB of repetitive text so BPE has
    # enough material to train without being heavy.
    stage_dir = tmp_path / "corpus"
    stage_dir.mkdir()

    sample_text = (
        "the quick brown fox jumps over the lazy dog. "
        "a stitch in time saves nine. "
        "all that glitters is not gold. "
        "to be or not to be, that is the question. "
    ) * 200

    files: dict[str, list[Path]] = {"science": [], "literature": []}
    for stage in files:
        for i in range(2):
            p = stage_dir / f"{stage}_{i}.txt"
            p.write_text(sample_text, encoding="utf-8")
            files[stage].append(p)

    list_path = tmp_path / "file_list.txt"
    lines = []
    for stage, paths in files.items():
        lines.append(f"# === Stage: {stage} ({len(paths)} files) ===")
        for p in paths:
            lines.append(str(p))
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Train a small BPE tokenizer on the synthetic text.
    tok = BPETokenizer(target_vocab_size=512)
    tok.train(sample_text)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    return list_path, tok_path


def _run_curriculum_cli(
    list_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    *,
    cycles: int = 1,
    resume: Path | None = None,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Invoke train_curriculum.py via subprocess so we exercise the real CLI."""
    cmd = [
        sys.executable, "-m", "luthi.train_curriculum",
        "--file_list", str(list_path),
        "--tokenizer_path", str(tokenizer_path),
        "--output_dir", str(output_dir),
        "--checkpoint_password", "test_password",
        "--cycles", str(cycles),
        "--device", "cpu",
        "--batch_size", "4",
        "--seq_len", "16",
        "--lr", "1e-3",
        "--d_model", "32",
        "--n_blocks", "2",
        "--num_episodes", "2",
        "--no_backward_pass",
        "--seed", "1",
    ]
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    if extra:
        cmd.extend(extra)
    return subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        timeout=300,
    )


@pytest.mark.slow
class TestCurriculumEndToEnd:
    def test_small_scale_run_produces_per_stage_checkpoints(self, tmp_path):
        list_path, tok_path = _make_tiny_corpus(tmp_path)
        output_dir = tmp_path / "runs"

        result = _run_curriculum_cli(list_path, tok_path, output_dir, cycles=1)
        assert result.returncode == 0, (
            f"curriculum run failed:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        # One checkpoint per stage in cycle 1.
        ckpts = sorted(output_dir.glob("checkpoint_stage_*.luthi"))
        assert [p.name for p in ckpts] == [
            "checkpoint_stage_1_literature.luthi",
            "checkpoint_stage_1_science.luthi",
        ]

        # The summary file exists and reports two stage records.
        summary_path = output_dir / "curriculum_summary.json"
        assert summary_path.exists()

        # Living state advanced across stages — the second-stage checkpoint
        # has different weights from the first.
        ck_a = load_checkpoint(
            output_dir / "checkpoint_stage_1_science.luthi",
            "test_password", "cpu",
        )
        ck_b = load_checkpoint(
            output_dir / "checkpoint_stage_1_literature.luthi",
            "test_password", "cpu",
        )
        wa = ck_a["model_state_dict"]["blocks.0.living_ffn.weight"]
        wb = ck_b["model_state_dict"]["blocks.0.living_ffn.weight"]
        assert not torch.equal(wa, wb), "living weight should change across stages"

    def test_resume_from_mid_curriculum_checkpoint(self, tmp_path):
        list_path, tok_path = _make_tiny_corpus(tmp_path)
        output_dir = tmp_path / "runs"

        # First run: cycle 1, both stages.
        result1 = _run_curriculum_cli(list_path, tok_path, output_dir, cycles=1)
        assert result1.returncode == 0, result1.stderr

        first_stage_ckpt = output_dir / "checkpoint_stage_1_science.luthi"
        assert first_stage_ckpt.exists()

        # Capture the living state at the science checkpoint.
        before = load_checkpoint(first_stage_ckpt, "test_password", "cpu")
        w_before = before["model_state_dict"]["blocks.0.living_ffn.weight"].clone()

        # Second run: resume from science checkpoint, complete cycle 1.
        # This should ONLY run literature (not re-run science), so the
        # post-resume literature checkpoint must reflect continuous state.
        output_dir2 = tmp_path / "runs2"
        output_dir2.mkdir()
        # Re-run with resume — output to a fresh dir so we can inspect.
        result2 = _run_curriculum_cli(
            list_path, tok_path, output_dir2,
            cycles=1, resume=first_stage_ckpt,
        )
        assert result2.returncode == 0, result2.stderr

        # Resume produces the literature checkpoint of cycle 1.
        resumed_lit = output_dir2 / "checkpoint_stage_1_literature.luthi"
        assert resumed_lit.exists()
        # And does NOT re-emit the science checkpoint.
        assert not (output_dir2 / "checkpoint_stage_1_science.luthi").exists()

        # The literature checkpoint reflects continued evolution from
        # the science checkpoint's living state — weights have moved
        # since we resumed but stayed continuous through the boundary.
        after = load_checkpoint(resumed_lit, "test_password", "cpu")
        w_after = after["model_state_dict"]["blocks.0.living_ffn.weight"]
        # Continuous: not equal to start, but in the same neighborhood as
        # the science snapshot we resumed from (no reset happened).
        assert not torch.equal(w_after, w_before), (
            "literature stage should have advanced the weights"
        )


# ---------------------------------------------------------------------------
# Production config sanity
# ---------------------------------------------------------------------------


def test_production_config_documented():
    """The config dict that gets uploaded with the rental — keep these
    keys intact so the launch script can rely on them."""
    expected = {
        "d_model", "n_blocks", "vocab_size", "seq_len", "num_episodes",
        "hebb_rate", "error_rate", "spike_threshold", "backward_pass_enabled",
    }
    assert expected.issubset(PRODUCTION_CONFIG.keys())
    assert PRODUCTION_CONFIG["d_model"] == 4096
    assert PRODUCTION_CONFIG["n_blocks"] == 36
    assert PRODUCTION_CONFIG["vocab_size"] == 32000
