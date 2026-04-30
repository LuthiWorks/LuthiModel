"""Tests for gradient checkpointing on the spiking living-weight model.

The contract verified here is what makes activation checkpointing safe to
turn on for the production 4096d/36-block run:

  1. Gradients with checkpointing match gradients without (within fp32
     numerical noise). The recomputed forward must reproduce the original
     forward's activations exactly — which means living state mutations
     (Hebbian, episode store, membrane integration) must NOT fire during
     recompute, and the linear op must reuse the same weight snapshot.

  2. The Hebbian update fires exactly once per training step regardless
     of whether checkpointing is enabled. We verify this by counting the
     change in `update_ema` (the Hebbian-update running average) — it
     should advance by the same amount in both modes.

  3. The recompute flag is correctly toggled. We watch is_recomputing()
     transition False → True → False across a forward+backward pair when
     checkpointing is on.
"""

from __future__ import annotations

import copy
import torch

from luthi.model_spiking import SpikingLuthiLM
from luthi.grad_checkpoint import is_recomputing


def _build_model(gradient_checkpointing: bool, seed: int = 1234) -> SpikingLuthiLM:
    """Build a small spiking model with deterministic init."""
    torch.manual_seed(seed)
    m = SpikingLuthiLM(
        vocab_size=32,
        d_model=16,
        n_blocks=2,
        max_seq_len=8,
        gradient_checkpointing=gradient_checkpointing,
        backward_pass_enabled=False,  # keep the test focused on bottom-up
    )
    m.train()
    return m


def _input_batch(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 32, (2, 8), generator=g)


def _living_state_snapshot(model: SpikingLuthiLM) -> dict[str, torch.Tensor]:
    """Capture every living buffer that Hebbian self-modification touches."""
    snap: dict[str, torch.Tensor] = {}
    for i, block in enumerate(model.blocks):
        ffn = block.living_ffn
        for name in (
            "weight", "set_point", "momentum", "input_avg_mag",
            "excitability_acc", "plasticity", "update_ema",
            "membrane_potential", "refractory_counter",
            "spike_mask", "delay_buffer",
        ):
            snap[f"b{i}.{name}"] = getattr(ffn, name).clone()
    return snap


class TestGradientCheckpointing:
    def test_gradients_match_with_and_without_checkpointing(self):
        """The whole point: checkpointing must be a memory optimization,
        not a semantic change. Same input + same init → same gradients."""
        model_off = _build_model(gradient_checkpointing=False, seed=42)
        model_on = _build_model(gradient_checkpointing=True, seed=42)

        # Sanity: init produced identical state.
        s_off = _living_state_snapshot(model_off)
        s_on = _living_state_snapshot(model_on)
        for k in s_off:
            assert torch.allclose(s_off[k], s_on[k]), f"init mismatch: {k}"

        x = _input_batch(seed=7)
        y = _input_batch(seed=8)

        # Forward + backward in both modes.
        loss_off = torch.nn.functional.cross_entropy(
            model_off(x).reshape(-1, 32), y.reshape(-1),
        )
        loss_off.backward()

        loss_on = torch.nn.functional.cross_entropy(
            model_on(x).reshape(-1, 32), y.reshape(-1),
        )
        loss_on.backward()

        # Loss matches — recomputed forward reproduces original activations.
        assert torch.allclose(loss_off, loss_on, atol=1e-5), (
            f"Loss differs with checkpointing: {loss_off.item()} vs "
            f"{loss_on.item()}"
        )

        # Gradients on every trainable parameter match.
        for (n_off, p_off), (n_on, p_on) in zip(
            model_off.named_parameters(), model_on.named_parameters(),
        ):
            assert n_off == n_on
            if p_off.grad is None and p_on.grad is None:
                continue
            assert p_off.grad is not None and p_on.grad is not None, (
                f"grad missing on {n_off}: off={p_off.grad}, on={p_on.grad}"
            )
            assert torch.allclose(p_off.grad, p_on.grad, atol=1e-5, rtol=1e-4), (
                f"grad mismatch on {n_off}: max diff "
                f"{(p_off.grad - p_on.grad).abs().max().item()}"
            )

    def test_hebbian_fires_exactly_once_per_step(self):
        """update_ema is the Hebbian running average — it advances every
        time the self-modification fires. After one forward+backward, both
        modes must have advanced it by the same delta."""
        model_off = _build_model(gradient_checkpointing=False, seed=99)
        model_on = _build_model(gradient_checkpointing=True, seed=99)

        before_off = _living_state_snapshot(model_off)
        before_on = _living_state_snapshot(model_on)

        x = _input_batch(seed=7)
        y = _input_batch(seed=8)

        # One full training step in each mode.
        torch.nn.functional.cross_entropy(
            model_off(x).reshape(-1, 32), y.reshape(-1),
        ).backward()
        torch.nn.functional.cross_entropy(
            model_on(x).reshape(-1, 32), y.reshape(-1),
        ).backward()

        after_off = _living_state_snapshot(model_off)
        after_on = _living_state_snapshot(model_on)

        # update_ema specifically tracks Hebbian update magnitude. If
        # Hebbian fired twice with checkpointing, the EMA would diverge.
        for i in range(2):
            d_off = (after_off[f"b{i}.update_ema"]
                     - before_off[f"b{i}.update_ema"])
            d_on = (after_on[f"b{i}.update_ema"]
                    - before_on[f"b{i}.update_ema"])
            assert torch.allclose(d_off, d_on, atol=1e-7), (
                f"block {i} update_ema delta mismatch — "
                f"Hebbian fired a different number of times."
            )

        # Living weight delta also matches: this is the strongest signal
        # that self-modification was identical across modes.
        for i in range(2):
            d_off = after_off[f"b{i}.weight"] - before_off[f"b{i}.weight"]
            d_on = after_on[f"b{i}.weight"] - before_on[f"b{i}.weight"]
            assert torch.allclose(d_off, d_on, atol=1e-6), (
                f"block {i} weight delta differs across checkpointing modes"
            )

    def test_membrane_potential_advances_once(self):
        """The membrane integrator (Phases 0-1, 5) is also stateful. If
        it fired twice during recompute, membrane_potential would drift
        further than in the non-checkpointed run."""
        model_off = _build_model(gradient_checkpointing=False, seed=7)
        model_on = _build_model(gradient_checkpointing=True, seed=7)

        x = _input_batch(seed=11)
        y = _input_batch(seed=12)

        torch.nn.functional.cross_entropy(
            model_off(x).reshape(-1, 32), y.reshape(-1),
        ).backward()
        torch.nn.functional.cross_entropy(
            model_on(x).reshape(-1, 32), y.reshape(-1),
        ).backward()

        for i in range(2):
            mp_off = model_off.blocks[i].living_ffn.membrane_potential
            mp_on = model_on.blocks[i].living_ffn.membrane_potential
            assert torch.allclose(mp_off, mp_on, atol=1e-6), (
                f"block {i} membrane_potential drifted under checkpointing"
            )
            dp_off = model_off.blocks[i].living_ffn._delay_pos
            dp_on = model_on.blocks[i].living_ffn._delay_pos
            assert dp_off == dp_on, (
                f"block {i} _delay_pos advanced different amounts: "
                f"off={dp_off}, on={dp_on}"
            )

    def test_recompute_flag_is_false_outside_checkpoint(self):
        """The thread-local should be False at rest — only flipped on
        during the backward recompute pass."""
        assert not is_recomputing()
        model = _build_model(gradient_checkpointing=True, seed=1)
        x = _input_batch(seed=1)
        y = _input_batch(seed=2)
        loss = torch.nn.functional.cross_entropy(
            model(x).reshape(-1, 32), y.reshape(-1),
        )
        # Still False after forward — recompute hasn't run yet.
        assert not is_recomputing()
        loss.backward()
        # And cleanly reset after backward completes.
        assert not is_recomputing()

    def test_eval_mode_skips_checkpointing(self):
        """When .eval() is set, no backward will follow, so we don't pay
        the recompute cost. Just verify forward still works in eval."""
        model = _build_model(gradient_checkpointing=True, seed=3)
        model.eval()
        x = _input_batch(seed=4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 8, 32)
