"""muPC rate balancing (2026-07-30).

`residual_scale` multiplies each block's output into the residual stream, so
every backprop-trained parameter in the block receives a gradient scaled by it.
The living FFN does not: it self-modifies locally inside the forward pass,
before that multiplication, from a LayerNorm'd input whose scale is pinned to
1.0 regardless of depth (measured: PC-input RMS 1.0000 at depths 4/8/12/36).

So muPC attenuates one half of a two-speed system. `mu_pc_balance_rates` scales
the PC rates by the same factor so both halves move together.

See docs/research/2026-07-30_mupc-verdict.md and the block's constructor comment.
"""
import pytest
import torch

from luthi.v2.hybrid_block_pc import PredictiveCodingBlock


def _block(**kw):
    base = dict(d_model=32, n_heads=4, num_episodes=4)
    base.update(kw)
    return PredictiveCodingBlock(**base)


class TestRateFactor:
    def test_off_by_default(self):
        b = _block(mu_pc_enabled=True, mu_pc_exponent=0.25, n_blocks_total=8)
        assert b._mu_pc_rate_factor == 1.0

    def test_negative_power_amplifies(self):
        """power=-1 is the opposite adjustment: amplify the PC rates by 1/s.

        Registered after power=+1 (attenuate) made the collapse worse -- offset
        dominance 0.5657 -> 0.8277. The three-point ordering showed total
        attenuation, not the PC/backprop ratio, tracks the offset, so the
        opposite direction is the one the data points at.
        """
        b = _block(mu_pc_enabled=True, mu_pc_exponent=0.25, n_blocks_total=8,
                   mu_pc_rate_power=-1.0, pc_rate=0.001)
        assert b._mu_pc_rate_factor == pytest.approx(1.0 / b.residual_scale)
        assert b._mu_pc_rate_factor > 1.0
        assert b.living_ffn.pc_rate == pytest.approx(0.001 / b.residual_scale)

    def test_powers_are_symmetric_about_off(self):
        lo = _block(mu_pc_enabled=True, mu_pc_exponent=0.25, n_blocks_total=8,
                    mu_pc_rate_power=1.0)
        hi = _block(mu_pc_enabled=True, mu_pc_exponent=0.25, n_blocks_total=8,
                    mu_pc_rate_power=-1.0)
        assert lo._mu_pc_rate_factor * hi._mu_pc_rate_factor == pytest.approx(1.0)

    def test_factor_matches_residual_scale_when_enabled(self):
        b = _block(mu_pc_enabled=True, mu_pc_exponent=0.25, n_blocks_total=8,
                   mu_pc_rate_power=1.0)
        assert b._mu_pc_rate_factor == pytest.approx(b.residual_scale)
        assert b.residual_scale == pytest.approx(8 ** -0.25)

    def test_no_effect_when_mupc_disabled(self):
        """With muPC off there is no attenuation to balance against."""
        b = _block(mu_pc_enabled=False, n_blocks_total=8,
                   mu_pc_rate_power=1.0)
        assert b.residual_scale == 1.0
        assert b._mu_pc_rate_factor == 1.0

    def test_pc_rates_are_actually_scaled(self):
        base = _block(mu_pc_enabled=True, mu_pc_exponent=0.25,
                      n_blocks_total=8, pc_rate=0.001,
                      pred_learning_rate=0.0001)
        bal = _block(mu_pc_enabled=True, mu_pc_exponent=0.25,
                     n_blocks_total=8, pc_rate=0.001,
                     pred_learning_rate=0.0001, mu_pc_rate_power=1.0)
        s = bal.residual_scale
        assert base.living_ffn.pc_rate == pytest.approx(0.001)
        assert bal.living_ffn.pc_rate == pytest.approx(0.001 * s)
        assert bal.living_ffn.pred_learning_rate == pytest.approx(0.0001 * s)

    def test_deeper_means_more_attenuation(self):
        rates = []
        for L in (4, 8, 36):
            b = _block(mu_pc_enabled=True, mu_pc_exponent=0.25,
                       n_blocks_total=L, mu_pc_rate_power=1.0)
            rates.append(b.living_ffn.pc_rate)
        assert rates[0] > rates[1] > rates[2]


class TestBitIdentity:
    def test_default_path_unchanged(self):
        """A block built without the flag must be bit-identical to one built
        before the flag existed -- i.e. rates untouched."""
        b = _block(mu_pc_enabled=True, mu_pc_exponent=0.25, n_blocks_total=12,
                   pc_rate=0.003, pred_learning_rate=0.0007)
        assert b.living_ffn.pc_rate == pytest.approx(0.003)
        assert b.living_ffn.pred_learning_rate == pytest.approx(0.0007)

    def test_forward_still_runs_with_balancing(self):
        torch.manual_seed(0)
        b = _block(mu_pc_enabled=True, mu_pc_exponent=0.25, n_blocks_total=8,
                   mu_pc_rate_power=1.0)
        out = b(torch.randn(2, 6, 32))
        assert out.shape == (2, 6, 32)
        assert torch.isfinite(out).all()


class TestTheMeasurementThisRestsOn:
    def test_pc_input_scale_is_depth_independent(self):
        """The load-bearing measurement, pinned as a test.

        The 2026-05-16 design doc asserted that muPC's residual scaling means
        "the PC layer's input is dampened, its pred_error is dampened, and its
        self-modification is dampened". That is not what happens: norm2 is a
        LayerNorm, so the PC layer's input is unit-scale at every depth and the
        residual stream's magnitude never reaches it. If this test ever fails,
        the balancing rationale needs re-deriving.
        """
        seen = {}
        for L in (4, 36):
            torch.manual_seed(0)
            b = _block(mu_pc_enabled=True, mu_pc_exponent=0.25,
                       n_blocks_total=L)
            rec = []
            h = b.living_ffn.register_forward_hook(
                lambda m, i, o: rec.append(
                    float(i[0].float().pow(2).mean().sqrt())
                )
            )
            torch.manual_seed(1)
            b(torch.randn(2, 6, 32))
            h.remove()
            seen[L] = rec[0]
        assert seen[4] == pytest.approx(seen[36], rel=0.05), seen
