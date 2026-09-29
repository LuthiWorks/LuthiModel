"""The per-block contribution gauge sees what output gauges cannot.

Claim under test: output gauges (effective_rank, chorus_eff_rank) read the
post-residual stream h_out = h_in + c and cannot distinguish a healthy
block from a collapsed block the residual bypasses. The contribution
gauge reads c = h_out - h_in directly.

Why this test exists (A7 carrier-vs-bypass review, 2026-09-29): the A7
verdict "blocks 0-1 were never collapsed" rests on output gauges that
fit a second hypothesis -- the living FFN contributes a near-batch-constant
giant vector (the v5 pathology in miniature) while the residual carries
the rich representation past it. Under that hypothesis every existing
per-block gauge reads healthy. These tests pin the discriminating
behavior synthetically: a bypassed block MUST read dead on the
contribution gauge even while the old gauges read "benign carrier".
"""

from __future__ import annotations

import math

import torch

from luthi.v2.jepa_runner import _block_contribution_metrics, _rank_and_top_share


B, T, D = 8, 16, 32


def _rich_stream(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


class TestContributionGauge:
    def test_soloist_contribution_is_rank1_while_old_gauges_read_carrier(self):
        """The core discrimination test (faithful A7 signature).

        h_in is rich; the block adds a soloist: large variance along ONE
        fixed direction, magnitude varying per sample (this is what
        produces eff~2-4 + high chorus in the real runs -- a strictly
        batch-constant vector would vanish under centering and could
        never make eff low).

        Old gauges on h_out: eff low + chorus HIGH -> the A7 "benign
        carrier" reading; the richness they report is the residual's.
        The contribution gauge must report: the block's own addition is
        rank-1 (chorus NaN), however much variance it carries.
        """
        h_in = _rich_stream(0)
        g = torch.Generator().manual_seed(1)
        soloist = torch.randn(D, generator=g)
        soloist = soloist / soloist.norm()
        g2 = torch.Generator().manual_seed(2)
        amp = torch.randn(B, T, 1, generator=g2) * 10.0
        h_out = h_in + amp * soloist

        eff, tds, chorus = _rank_and_top_share(h_out)
        assert eff < 5.0, f"soloist should crush effective rank, got {eff}"
        assert chorus > D / 2, (
            f"old gauge reads 'benign carrier' (chorus {chorus:.1f}) -- "
            "this is the blind spot under test"
        )

        var_ratio, contrib_chorus = _block_contribution_metrics(h_in, h_out)
        assert var_ratio > 1.0, (
            f"soloist carries real variance, expected var_ratio > 1, "
            f"got {var_ratio}"
        )
        assert math.isnan(contrib_chorus), (
            "rank-1 contribution has no measurable tail; chorus must be "
            "NaN, not the dimensionality of clamp-dust"
        )

    def test_strictly_dead_block(self):
        """Batch-constant contribution: var_ratio ~ 0, chorus NaN."""
        h_in = _rich_stream(9)
        const = torch.full((T, D), 3.0)
        h_out = h_in + const  # broadcast over batch

        var_ratio, contrib_chorus = _block_contribution_metrics(h_in, h_out)
        assert var_ratio < 1e-6, f"dead block must read ~0, got {var_ratio}"
        assert math.isnan(contrib_chorus)

    def test_rich_block_reads_alive(self):
        """A block whose contribution varies with input and spans many
        directions reads alive on both contribution numbers."""
        h_in = _rich_stream(2)
        c = 0.5 * _rich_stream(3)
        h_out = h_in + c

        var_ratio, contrib_chorus = _block_contribution_metrics(h_in, h_out)
        # var(c)/var(h_in) with c = 0.5 * N(0,1): expect ~0.25.
        assert 0.15 < var_ratio < 0.4, f"unexpected var_ratio {var_ratio}"
        assert contrib_chorus > D / 2, (
            f"rich contribution should have high chorus, got {contrib_chorus}"
        )

    def test_low_rank_but_live_block(self):
        """Contribution varies with input but lives in few directions:
        var_ratio alive, chorus low. Distinct from dead."""
        h_in = _rich_stream(4)
        g = torch.Generator().manual_seed(5)
        dirs = torch.randn(2, D, generator=g)
        coords = torch.randn(B, T, 2, generator=g)
        c = coords @ dirs  # rank-2, batch-varying
        h_out = h_in + c

        var_ratio, contrib_chorus = _block_contribution_metrics(h_in, h_out)
        assert var_ratio > 1e-3, f"live block must have var_ratio, got {var_ratio}"
        # Rank-2 contribution: exactly one tail direction survives the
        # noise floor, so chorus == 1.0. Assert loosely for float32 SVD.
        assert contrib_chorus < 2.0, (
            f"rank-2 contribution should have chorus ~1, got {contrib_chorus}"
        )

    def test_dust_contribution_gives_nan_chorus(self):
        """A 1e-9-scale contribution is numerically dead: var_ratio ~ 0
        and chorus NaN, not the dimensionality of float dust."""
        h_in = _rich_stream(6)
        h_out = h_in + 1e-9 * _rich_stream(7)

        var_ratio, contrib_chorus = _block_contribution_metrics(h_in, h_out)
        assert var_ratio < 1e-6
        assert math.isnan(contrib_chorus)

    def test_zero_variance_input_gives_nan_not_zero(self):
        """Undefined ratio must be NaN -- a silent 0.0 would read as
        'dead block' and a silent 1.0 as 'healthy'."""
        h_in = torch.ones(B, T, D)
        h_out = h_in + _rich_stream(8)

        var_ratio, contrib_chorus = _block_contribution_metrics(h_in, h_out)
        assert math.isnan(var_ratio) and math.isnan(contrib_chorus)

    def test_mismatched_shapes_never_raise(self):
        """Diagnostics never kill a run: garbage in, NaN out."""
        var_ratio, contrib_chorus = _block_contribution_metrics(
            torch.zeros(B, T, D), torch.zeros(B, T, D + 1)
        )
        assert math.isnan(var_ratio) and math.isnan(contrib_chorus)


class TestBlockInputsPlumbing:
    def test_encode_emits_aligned_block_inputs(self):
        """collect_block_latents=True must also emit block_inputs, one
        per block, shape-aligned with block_latents, with
        block_latents[i] - block_inputs[i] finite (the contribution)."""
        from luthi.v2.jepa_loss import JEPALoss
        from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

        torch.manual_seed(11)
        model = MultimodalPredictiveCodingLM(
            vocab_size=32, d_model=32, n_blocks=2, n_heads=2,
            ffn_expansion=1, max_seq_len=16,
            max_audio_tokens=16, max_vision_tokens=16,
            backward_pass_enabled=False,
        )
        loss = JEPALoss(online_encoder=model)
        loss.train()
        batch = {"text_tokens": torch.randint(0, 32, (2, 16))}
        result = loss.compute_modality_loss(
            "text", batch, collect_block_latents=True
        )

        bl = result["block_latents"]
        bi = result["block_inputs"]
        assert bl is not None and bi is not None, "plumbing dropped the inputs"
        assert len(bi) == len(bl) == 2, f"expected 2 blocks, got {len(bi)}/{len(bl)}"
        for i, (h_in, h_out) in enumerate(zip(bi, bl)):
            assert h_in.shape == h_out.shape, f"block {i} shape mismatch"
            c = h_out - h_in
            assert torch.isfinite(c).all(), f"block {i} contribution non-finite"
            vr, _ = _block_contribution_metrics(h_in, h_out)
            assert vr == vr, f"block {i} contribution gauge NaN on a live model"
