"""Depth-8 remedy probes (2026-08-07): TC-SIGReg, interior Weak-SIGReg,
orthogonal penalty. Unit tests for the three helpers plus the fail-loud
contract on inert configuration."""
import math

import pytest
import torch

from luthi.v2.jepa_loss import (
    orthogonality_penalty,
    sketched_isotropy_penalty,
    temporal_center,
)


class TestTemporalCenter:
    def test_shape_preserved(self):
        z = torch.randn(2, 32, 16)
        assert temporal_center(z, 9).shape == z.shape

    def test_constant_sequence_gives_zero_residual(self):
        z = torch.ones(2, 32, 16) * 3.7
        assert temporal_center(z, 9).abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_shared_offset_removed(self):
        """The batch/temporal-shared component -- our measured pathology --
        must vanish from SIGReg's view under temporal centering."""
        z = torch.randn(2, 32, 16)
        offset = torch.randn(1, 1, 16) * 10
        r_plain = temporal_center(z, 9)
        r_offset = temporal_center(z + offset, 9)
        assert torch.allclose(r_plain, r_offset, atol=1e-5)

    def test_even_window_refused(self):
        with pytest.raises(ValueError):
            temporal_center(torch.randn(1, 8, 4), 8)

    def test_zero_window_is_identity(self):
        z = torch.randn(2, 8, 4)
        assert temporal_center(z, 0) is z


class TestSketchedIsotropy:
    def _sketch(self, d=64, k=16):
        g = torch.Generator().manual_seed(0)
        return torch.randn(d, k, generator=g) / math.sqrt(d)

    def test_isotropic_is_small_rank1_is_large(self):
        d, k = 64, 16
        sk = self._sketch(d, k)
        iso = torch.randn(4000, d)
        u = torch.randn(d)
        rank1 = torch.randn(4000, 1) * u * 3.0
        p_iso = sketched_isotropy_penalty(iso, sk).item()
        p_r1 = sketched_isotropy_penalty(rank1, sk).item()
        assert p_r1 > 3 * p_iso

    def test_collapse_toward_zero_variance_is_penalized(self):
        """Shrinking latents must RAISE the penalty (variance floor) --
        the anti-collapse property the interior blocks currently lack."""
        sk = self._sketch()
        z = torch.randn(4000, 64)
        assert (sketched_isotropy_penalty(z * 0.01, sk)
                > sketched_isotropy_penalty(z, sk))


class TestOrthogonalityPenalty:
    def test_orthogonal_matrix_is_near_zero_any_scale(self):
        q, _ = torch.linalg.qr(torch.randn(64, 64))
        assert orthogonality_penalty(q * 7.3).item() == pytest.approx(0.0, abs=1e-6)

    def test_rank1_is_large(self):
        u = torch.randn(64, 1); v = torch.randn(1, 64)
        assert orthogonality_penalty(u @ v).item() > 10

    def test_scale_free(self):
        w = torch.randn(64, 64)
        a = orthogonality_penalty(w).item()
        b = orthogonality_penalty(w * 100).item()
        assert a == pytest.approx(b, rel=1e-4)


class TestMuPCSchedule:
    """Scheduled muPC (2026-08-07, Brian's design): anneal residual_scale
    1.0 -> muPC target across a ramp; loud contracts on misconfiguration."""

    def _guard(self, **over):
        from luthi.v2.jepa_runner import JEPATrainer, RunnerConfig, SamplerConfig

        class _G:
            _apply_mu_pc_schedule = JEPATrainer._apply_mu_pc_schedule
        g = _G()
        cfg = RunnerConfig(sampler=SamplerConfig(corpus_sizes_tokens={"text": 1000}))
        for k, v in over.items():
            setattr(cfg, k, v)
        g.config = cfg

        class _B:
            def __init__(self):
                self.residual_scale = 1.0
                self.mu_pc_rate_power = 0.0

        class _Enc:
            blocks = [_B() for _ in range(8)]

        class _LM:
            online_encoder = _Enc()
        g.loss_module = _LM()
        g.global_step = 0
        return g

    def test_disabled_is_noop(self):
        g = self._guard(mu_pc_schedule_start=0)
        g.loss_module.online_encoder.blocks[0].residual_scale = 1.0
        g._apply_mu_pc_schedule()
        assert g.loss_module.online_encoder.blocks[0].residual_scale == 1.0

    def test_anneal_reaches_mupc_target(self):
        g = self._guard(mu_pc_schedule_start=3000, mu_pc_schedule_ramp=1000)
        target = 1.0 / (8 ** 0.25)
        g.global_step = 2999; g._apply_mu_pc_schedule()
        assert g.loss_module.online_encoder.blocks[0].residual_scale == pytest.approx(1.0)
        g.global_step = 3500; g._apply_mu_pc_schedule()
        mid = g.loss_module.online_encoder.blocks[0].residual_scale
        assert 1.0 > mid > target
        g.global_step = 4000; g._apply_mu_pc_schedule()
        assert g.loss_module.online_encoder.blocks[0].residual_scale == pytest.approx(target)
        g.global_step = 9999; g._apply_mu_pc_schedule()
        assert g.loss_module.online_encoder.blocks[0].residual_scale == pytest.approx(target)

    def test_refuses_built_in_mupc(self):
        g = self._guard(mu_pc_schedule_start=3000)
        g.loss_module.online_encoder.blocks[3].residual_scale = 0.5946
        with pytest.raises(RuntimeError, match="double-attenuate"):
            g._apply_mu_pc_schedule()

    def test_refuses_rate_power_arms(self):
        g = self._guard(mu_pc_schedule_start=3000)
        g.loss_module.online_encoder.blocks[2].mu_pc_rate_power = -4.0
        with pytest.raises(RuntimeError, match="rate_power"):
            g._apply_mu_pc_schedule()
