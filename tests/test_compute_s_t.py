"""Unit tests for ``luthi.v2.m9.s_t.compute_s_t``.

The canonical s_t definition for the training seam (Phase 4a). Pinning
the contract here so neither training nor inference can drift the
behavior under the helper -- changes to compute_s_t are forced through
this test.
"""

from __future__ import annotations

import pytest
import torch

from luthi.v2.m9.s_t import compute_s_t


class TestComputeSTContract:
    def test_pools_3d_to_2d(self):
        latents = torch.randn(2, 8, 4)
        s_t = compute_s_t(latents)
        assert s_t.shape == (2, 4)

    def test_matches_inline_mean(self):
        """The historical inline compute was
        ``raw["online_context_latents"].detach().mean(dim=1)``.
        compute_s_t must produce the same value for the same input."""
        latents = torch.randn(3, 10, 6)
        expected = latents.detach().mean(dim=1)
        actual = compute_s_t(latents)
        assert torch.equal(actual, expected)

    def test_detaches_for_stop_grad(self):
        """The M9 head step requires the s_t consumed by V-head /
        habit-net to be detached so M9 gradients don't flow into the
        JEPA encoder. compute_s_t enforces that here so individual
        call sites don't have to remember."""
        latents = torch.randn(2, 4, 3, requires_grad=True)
        s_t = compute_s_t(latents)
        assert not s_t.requires_grad

    def test_rejects_2d_input(self):
        """A 2D input is almost certainly a bug -- a stale [T, D] from
        somewhere that lost its batch dim. Reject loudly."""
        latents = torch.randn(4, 3)
        with pytest.raises(ValueError, match=r"\[B, T, D\]"):
            compute_s_t(latents)

    def test_rejects_4d_input(self):
        latents = torch.randn(2, 4, 3, 5)
        with pytest.raises(ValueError, match=r"\[B, T, D\]"):
            compute_s_t(latents)

    def test_preserves_batch_dim(self):
        for batch in (1, 2, 7):
            latents = torch.randn(batch, 5, 3)
            assert compute_s_t(latents).shape[0] == batch

    def test_preserves_d_dim(self):
        for d in (1, 16, 64):
            latents = torch.randn(2, 5, d)
            assert compute_s_t(latents).shape[1] == d
