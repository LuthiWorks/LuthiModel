"""SIGReg's dedicated RNG generator (2026-08-01).

The defect: the projection matrix was drawn with a bare `torch.randn`, so every
SIGReg forward advanced the GLOBAL RNG stream -- the same stream feeding data
sampling, dropout, and every other stochastic component. And the amount consumed
scales with `num_proj`, so changing `num_proj` shifted the entire downstream
random sequence including data order. `num_proj` was never a clean single
variable, and nothing reported it.

Matches the reference (`rbalestr-lab/lejepa`, `SlicingUnivariateTest`), which
seeds a dedicated generator from a `global_step` counter.
"""
import pytest
import torch

from luthi.v2.sigreg import SIGReg


def _x():
    torch.manual_seed(0)
    return torch.randn(1, 64, 128)


class TestDefaultIsFixed:
    def test_on_by_default(self):
        """A confirmed defect ships fixed, per the 2026-07-28 principle."""
        assert SIGReg().use_generator is True

    def test_global_stream_untouched(self):
        s, x = SIGReg(num_proj=64), _x()
        torch.manual_seed(11)
        without = torch.randn(4)
        torch.manual_seed(11)
        s(x)
        after = torch.randn(4)
        assert torch.equal(without, after)

    def test_num_proj_no_longer_shifts_the_global_stream(self):
        """The specific confound: num_proj must not perturb downstream RNG.

        This is what made num_proj un-testable -- two runs differing only in it
        would also differ in data order.
        """
        draws = []
        for n in (64, 512):
            s = SIGReg(num_proj=n)
            torch.manual_seed(3)
            s(_x())
            draws.append(torch.randn(2))
        assert torch.equal(draws[0], draws[1])


class TestDeterminism:
    def test_same_step_same_result_regardless_of_global_seed(self):
        a = SIGReg(num_proj=64)
        b = SIGReg(num_proj=64)
        x = _x()
        torch.manual_seed(1)
        va = a(x).item()
        torch.manual_seed(99999)
        vb = b(x).item()
        assert va == pytest.approx(vb, rel=1e-9)

    def test_step_advances_so_projections_differ(self):
        s, x = SIGReg(num_proj=64), _x()
        s.train()
        first = s(x).item()
        second = s(x).item()
        assert first != pytest.approx(second, rel=1e-9)
        assert int(s.global_step.item()) == 2

    def test_step_counter_checkpoints(self):
        """global_step is a persistent buffer, so a resumed run does not replay
        the same projections it already used."""
        s = SIGReg(num_proj=64)
        s.train()
        for _ in range(3):
            s(_x())
        restored = SIGReg(num_proj=64)
        restored.load_state_dict(s.state_dict())
        assert int(restored.global_step.item()) == 3
        assert restored(_x()).item() == pytest.approx(s(_x()).item(), rel=1e-9)


class TestLegacyPathPreserved:
    def test_legacy_still_consumes_global_stream(self):
        """use_generator=False must reproduce pre-2026-08-01 behaviour exactly,
        so a prior run can be re-created."""
        s, x = SIGReg(num_proj=64, use_generator=False), _x()
        torch.manual_seed(11)
        without = torch.randn(4)
        torch.manual_seed(11)
        s(x)
        after = torch.randn(4)
        assert not torch.equal(without, after)

    def test_legacy_is_seed_dependent(self):
        x = _x()
        torch.manual_seed(5)
        a = SIGReg(num_proj=64, use_generator=False)(x).item()
        torch.manual_seed(6)
        b = SIGReg(num_proj=64, use_generator=False)(x).item()
        assert a != pytest.approx(b, rel=1e-9)


class TestStatisticUnchanged:
    def test_generator_does_not_change_what_sigreg_measures(self):
        """Both paths must agree on the distributions SIGReg is meant to catch.

        The generator changes WHICH directions are sampled, not the statistic.
        With enough projections both paths must rank these identically.
        """
        torch.manual_seed(0)
        good = torch.randn(1, 512, 64)
        shrunk = good * 0.01
        offset = good + 3.0
        for use_gen in (True, False):
            torch.manual_seed(4)
            s = SIGReg(num_proj=512, use_generator=use_gen)
            g, sh, off = s(good).item(), s(shrunk).item(), s(offset).item()
            assert sh > g * 10, (use_gen, g, sh)
            assert off > g * 10, (use_gen, g, off)


class TestEvalPurity:
    """A persistent counter that advances during eval makes evaluation mutate
    the model. Caught by tests/test_heldout_eval.py the moment the generator
    landed; pinned here at the unit level too."""

    def test_eval_does_not_advance_the_counter(self):
        s, x = SIGReg(num_proj=64), _x()
        s.eval()
        before = int(s.global_step.item())
        for _ in range(5):
            s(x)
        assert int(s.global_step.item()) == before

    def test_eval_batches_see_identical_projections(self):
        """Second benefit of holding the step: the eval metric is not jittered
        by resampling directions between batches."""
        s, x = SIGReg(num_proj=64), _x()
        s.eval()
        assert s(x).item() == pytest.approx(s(x).item(), rel=1e-12)

    def test_training_mode_still_advances(self):
        s, x = SIGReg(num_proj=64), _x()
        s.train()
        s(x)
        assert int(s.global_step.item()) == 1
