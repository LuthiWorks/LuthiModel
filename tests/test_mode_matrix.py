"""Combinatorial mode-flag sweep for PredictiveCodingLayer.forward (2026-07-15).

The per-feature tests (test_grad_checkpoint, test_frozen_plasticity_reencode,
test_pc_consolidation, test_pc_ops_cpp_sparse) each exercise one mode against
the default configuration. The latent bugs live in the CROSS-products -- a
flag combination nobody ran. This sweep runs the Cartesian product of the
mode axes through short forward+backward passes and asserts the invariants
that must hold in every cell:

  * outputs are finite;
  * living buffers mutate ONLY on the live self-mod path -- never under
    freeze_plasticity, never on a checkpoint recompute, never from
    backward();
  * gradient checkpointing is invisible: outputs, input-grads, and every
    buffer bit-identical to the un-checkpointed control;
  * forbidden cells raise loud (the declared matrix in
    luthi/v2/mode_compat.py), never silently corrupt.

Axes swept:
  T      -- inference_steps_per_forward in {1, 3} (classical PC vs iPC)
  sparse -- off | permanently-fully-gated (threshold=1e9, warmup=0: every
            row gated, so the invariant "gated rows never touch weight"
            becomes the sharp bitwise claim "weight never changes")
  consol -- consolidation tracker off | on (tiny windows; coexistence --
            firing behavior itself is test_pc_consolidation's job)
  gain   -- inverted-U learning gain off | on
  exec   -- plain | gradient-checkpointed | frozen (freeze_plasticity)
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.utils.checkpoint as torch_ckpt

from luthi.grad_checkpoint import luthi_context_fn
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.mode_compat import INCOMPATIBILITIES
from luthi.v2.plasticity import freeze_plasticity


IN, OUT = 8, 8
FLAG_AXES = tuple(itertools.product((1, 3), (False, True), (False, True), (False, True)))
FLAG_IDS = [
    f"T{t}-sparse{int(s)}-consol{int(c)}-gain{int(g)}" for t, s, c, g in FLAG_AXES
]


def _build(T: int, sparse: bool, consol: bool, gain: bool, seed: int = 11) -> PredictiveCodingLayer:
    torch.manual_seed(seed)
    return PredictiveCodingLayer(
        in_features=IN,
        out_features=OUT,
        inference_steps_per_forward=T,
        # Fully-gated sparse cell: threshold no error_acc will ever clear,
        # warmup 0 so the gate is armed from step one. Makes the sparse
        # invariant bitwise-sharp: the weight must NEVER change.
        sparse_threshold=1e9 if sparse else 0.0,
        sparse_warmup_steps=0,
        consolidation_enabled=consol,
        consolidation_window=8,
        consolidation_trigger_window=4,
        learning_gain_enabled=gain,
        num_episodes=4,
        context_dim=8,
        salience_threshold=0.0,  # store on every live forward
    )


def _x(seed: int = 5) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(2, 3, IN, requires_grad=True)


def _snap(layer: PredictiveCodingLayer) -> dict:
    state = {k: v.detach().clone() for k, v in layer.named_buffers()}
    state["__sparse_step_count"] = layer._sparse_step_count
    state["__err_short_count"] = layer._err_short._count
    state["__consolidation_fires"] = layer._consolidation_fire_count
    return state


def _assert_identical(before: dict, after: dict, context: str) -> None:
    assert before.keys() == after.keys()
    for key, prev in before.items():
        cur = after[key]
        if isinstance(prev, torch.Tensor):
            assert torch.equal(prev, cur), (
                f"[{context}] buffer {key!r} mutated -- this path must not "
                f"write living state"
            )
        else:
            assert prev == cur, f"[{context}] counter {key!r} advanced"


class TestFrozenMutatesNothing:
    """freeze_plasticity x every flag combo: grad-capable forward, zero
    living-state writes -- including the counters and the slow traces."""

    @pytest.mark.parametrize("T,sparse,consol,gain", FLAG_AXES, ids=FLAG_IDS)
    def test_frozen_cell(self, T, sparse, consol, gain):
        layer = _build(T, sparse, consol, gain)
        # Prime one live forward so the frozen pass has episodes to recall
        # and "unchanged" is a nontrivial claim.
        layer(_x(seed=3).detach())

        before = _snap(layer)
        x = _x()
        with freeze_plasticity(layer):
            out = layer(x)
        out.sum().backward()

        assert torch.isfinite(out).all(), "frozen forward produced non-finite"
        assert x.grad is not None, "frozen path must stay grad-capable"
        _assert_identical(before, _snap(layer), f"frozen {FLAG_IDS[FLAG_AXES.index((T, sparse, consol, gain))]}")


class TestCheckpointEquivalence:
    """Gradient checkpointing must be invisible at every level -- output,
    input-grad, and every living buffer bit-identical to the plain control
    -- across the legal (T=1) flag combos."""

    LEGAL = [axes for axes in FLAG_AXES if axes[0] == 1]
    LEGAL_IDS = [FLAG_IDS[FLAG_AXES.index(a)] for a in LEGAL]

    @pytest.mark.parametrize("T,sparse,consol,gain", LEGAL, ids=LEGAL_IDS)
    def test_bit_identity(self, T, sparse, consol, gain):
        plain = _build(T, sparse, consol, gain, seed=11)
        ckptd = _build(T, sparse, consol, gain, seed=11)

        x_plain = _x()
        out_plain = plain(x_plain)
        out_plain.sum().backward()

        x_ckpt = _x()
        out_ckpt = torch_ckpt.checkpoint(
            ckptd, x_ckpt, use_reentrant=False, context_fn=luthi_context_fn,
        )
        out_ckpt.sum().backward()

        assert torch.equal(out_plain, out_ckpt), "checkpointing changed the output"
        assert torch.equal(x_plain.grad, x_ckpt.grad), (
            "checkpointing changed the input gradient -- the recompute did "
            "not replay the original forward bit-identically"
        )
        _assert_identical(
            _snap(plain), _snap(ckptd),
            "plain-vs-checkpointed final state",
        )

    @pytest.mark.parametrize("T,sparse,consol,gain", LEGAL, ids=LEGAL_IDS)
    def test_backward_recompute_mutates_nothing(self, T, sparse, consol, gain):
        layer = _build(T, sparse, consol, gain)
        x = _x()
        out = torch_ckpt.checkpoint(
            layer, x, use_reentrant=False, context_fn=luthi_context_fn,
        )
        after_forward = _snap(layer)
        out.sum().backward()  # triggers the recompute replay
        _assert_identical(
            after_forward, _snap(layer),
            "recompute replay (backward)",
        )


class TestLiveMutationDiscipline:
    """Plain execution x every flag combo: self-mod runs exactly once, on
    the forward; backward touches nothing; the sparse gate's bitwise
    contract holds; the gain traces advance only when enabled."""

    @pytest.mark.parametrize("T,sparse,consol,gain", FLAG_AXES, ids=FLAG_IDS)
    def test_live_cell(self, T, sparse, consol, gain):
        layer = _build(T, sparse, consol, gain)
        before = _snap(layer)

        x = _x()
        out = layer(x)
        assert torch.isfinite(out).all()

        after_forward = _snap(layer)
        if sparse:
            # Fully-gated: the weight must be bitwise untouched...
            assert torch.equal(before["weight"], after_forward["weight"]), (
                "sparse gate leaked a weight update"
            )
            # ...while the layer stays alive around it.
            assert not torch.equal(
                before["prediction"], after_forward["prediction"]
            ), "prediction learning must run even when the weight gate is closed"
        else:
            assert not torch.equal(before["weight"], after_forward["weight"]), (
                "live forward did not self-modify"
            )
        assert int(after_forward["episode_count"].item()) == 1, (
            "salience_threshold=0 live forward must store an episode"
        )
        if gain:
            assert layer._err_short._count > 0, (
                "gain enabled but resolution traces never fed"
            )
        else:
            assert layer._err_short._count == 0, (
                "gain disabled but resolution traces were fed (regime f "
                "requires the gain machinery fully inert)"
            )

        out.sum().backward()
        _assert_identical(
            after_forward, _snap(layer), "backward on the plain path",
        )


class TestForbiddenCells:
    """The declared incompatibilities fire loud in their cells -- and the
    sweep knows about every rule in the matrix, so a new rule without a
    pinned cell fails here by name."""

    @pytest.mark.parametrize(
        "sparse,gain", [(False, False), (True, True)],
        ids=["plainflags", "sparse+gain"],
    )
    def test_ipc_x_checkpoint_raises_on_recompute(self, sparse, gain):
        layer = _build(T=3, sparse=sparse, consol=False, gain=gain)
        x = _x()
        out = torch_ckpt.checkpoint(
            layer, x, use_reentrant=False, context_fn=luthi_context_fn,
        )
        with pytest.raises(RuntimeError, match="Incompatible modes"):
            out.sum().backward()

    def test_frozen_original_then_recompute_raises(self):
        """The cell the encoder-level guard protects at the lived path,
        pinned at the LAYER: an original forward under freeze_plasticity
        caches no snapshot, so the checkpoint replay (after the freeze
        exits) must refuse rather than silently reuse stale state."""
        layer = _build(T=1, sparse=False, consol=False, gain=False)
        x = _x()
        with freeze_plasticity(layer):
            out = torch_ckpt.checkpoint(
                layer, x, use_reentrant=False, context_fn=luthi_context_fn,
            )
        with pytest.raises(RuntimeError, match="Incompatible modes"):
            out.sum().backward()

    def test_clear_cache_before_backward_raises(self):
        layer = _build(T=1, sparse=False, consol=False, gain=False)
        x = _x()
        out = torch_ckpt.checkpoint(
            layer, x, use_reentrant=False, context_fn=luthi_context_fn,
        )
        layer.clear_forward_cache()  # the misuse: cache dropped too early
        with pytest.raises(RuntimeError, match="Incompatible modes"):
            out.sum().backward()

    def test_every_declared_rule_has_a_pinned_cell(self):
        """Adding a rule to mode_compat.INCOMPATIBILITIES without adding a
        forbidden-cell test here must fail -- the matrix is only auditable
        if every declared rule is proven to actually fire."""
        pinned_here = {
            "ipc_x_grad_checkpoint",
            "recompute_without_original",
        }
        pinned_elsewhere = {
            # jepa_loss lived-path guard; fires in
            # tests/test_lived_jepa_updates_world_model.py's guard test and
            # is construction-shape (encoder flag), not layer-cell shape.
            "lived_reencode_x_grad_checkpoint",
        }
        declared = {rule.name for rule in INCOMPATIBILITIES}
        assert declared == pinned_here | pinned_elsewhere, (
            "mode_compat.INCOMPATIBILITIES changed: add a forbidden-cell "
            "test for the new rule (or record where it is pinned), then "
            "update this set"
        )
