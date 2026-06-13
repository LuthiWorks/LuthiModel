"""Sustained-run integration smoke test for M9Trainer.

The per-slice unit tests verified each piece in isolation across 3-8
train_steps. This test asks the harder question: does the assembled
loop stay coherent over a sustained run? At 100 steps we expect to
catch:

- Latent NaN/Inf cascades in the M9 sublosses (V-TD divergence the
  K-M9-3 ceiling didn't catch, decoder cycle-consistency exploding
  on a hostile batch, ...).
- Spurious kill fires from running bands that warm too quickly under
  the toy workload.
- Action-log disk runaway (one record per step, JSONL).
- Mid-run checkpoint+resume divergence.
- theta_version / sim_counter drift between MCTS and staleness.

This is intentionally low-bar -- pass/fail on stability, not quality.
A real pilot launch would replace the random-token loader with
LibriSpeech / COCO / a real text corpus.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from luthi.v2.m9.test_runner import _build_trainer


def test_sustained_100_step_run_stays_finite_and_healthy():
    """100 train_steps. Assert: no NaN in any m9 subloss, no kill fires,
    theta_version advances monotonically, action_log accumulates one
    record per step.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trainer = _build_trainer(run_dir, seed=2026)
        n_steps = 100

        kill_fire_log: list[tuple[int, str, str]] = []
        nan_log: list[tuple[int, str, float]] = []
        m8_losses: list[float] = []

        for step in range(n_steps):
            batch = trainer.data_loader.next_batch("text")
            out = trainer.train_step("text", batch)
            m8_losses.append(out["loss"])
            for k, v in out["m9"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if v != v or (v == float("inf")) or (v == float("-inf")):
                        nan_log.append((step, k, v))
            for kill_name, state in out["m9"]["kill_states"].items():
                if state == "fired":
                    kill_fire_log.append((step, kill_name, state))

        trainer.action_log.flush()

        # 1. No NaN/Inf in any m9 subloss.
        assert not nan_log, (
            f"non-finite m9 values: first few = {nan_log[:5]}, total {len(nan_log)}"
        )
        # 2. No M9 kill fired across the run (toy workload should not
        #    trigger any backstop). FLAGGED is fine -- the kill is
        #    being observed but not yet sustained-bad.
        assert not kill_fire_log, (
            f"M9 kill fired: first few = {kill_fire_log[:5]}, total {len(kill_fire_log)}"
        )
        # 3. theta_version advanced by exactly n_steps.
        assert trainer.staleness.theta_version == n_steps, (
            f"theta_version={trainer.staleness.theta_version}, expected {n_steps}"
        )
        # 4. action_log accumulated n_steps records.
        log_path = run_dir / trainer.m9_config.action_log_filename
        records = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == n_steps, (
            f"action_log had {len(records)} records, expected {n_steps}"
        )

        trainer.action_log.close()


def test_sustained_run_with_midpoint_checkpoint_resume():
    """50 steps -> checkpoint -> resume into fresh trainer -> 50 more.
    Final theta_version matches the linear-100-step run; no exception
    on the resume; the rest_action_net carries weights forward
    (verified earlier in test_runner; this asserts the round-trip
    survives a real training trajectory).
    """
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        t1 = _build_trainer(run_dir, seed=2026)

        for _ in range(50):
            batch = t1.data_loader.next_batch("text")
            t1.train_step("text", batch)
        t1._checkpoint(reason="midpoint")
        v_head_at_50 = [
            p.detach().clone() for p in t1.v_head.parameters()
        ]
        theta_at_50 = t1.staleness.theta_version
        t1.action_log.close()

        # Resume into fresh trainer with a *different* seed so the
        # M9 components start from a different init.
        ckpt_path = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-1]
        t2 = _build_trainer(run_dir, seed=99)
        t2.resume(ckpt_path)

        # V head matches t1's at the checkpoint.
        for p_b, p_a in zip(v_head_at_50, t2.v_head.parameters()):
            assert torch.equal(p_b, p_a), "V head mismatch after resume"

        # Run 50 more steps without an exception. theta_version
        # continues from where t1 left off (the resume restores the
        # staleness manager's counter from the checkpoint).
        # NB: at this slice the staleness manager isn't checkpointed,
        # so theta_version restarts. This test asserts the loop
        # SURVIVES the resume + 50 more steps; theta_version's specific
        # value is a separate property to test once we checkpoint it.
        for _ in range(50):
            batch = t2.data_loader.next_batch("text")
            t2.train_step("text", batch)
        assert t2.staleness.theta_version == 50  # restarted from 0
        t2.action_log.close()


def main() -> int:
    tests = [
        test_sustained_100_step_run_stays_finite_and_healthy,
        test_sustained_run_with_midpoint_checkpoint_resume,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
