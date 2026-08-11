"""Interior Weak-SIGReg (2026-08-07): the one depth-8 remedy that survived.

The TC-SIGReg, orthogonality-penalty and scheduled-muPC tests that lived here
were removed on 2026-08-10 with their mechanisms (pruning brief
docs/reviews/2026-08-10_pruning-and-visreg-brief-for-opus.md). Their results
are preserved in docs/, and git history keeps the code -- what closed by
verdict does not stay in the live tree.

wsig is kept because its arrest result stands: interior covariance pressure at
alpha=10 is the only intervention in the depth arc that ever prevented the
collapse rather than delaying it.
"""
import math

import torch

from luthi.v2.jepa_loss import sketched_isotropy_penalty


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

    def test_penalty_is_scale_sensitive(self):
        """Pins the 08-08 family verdict: the scale-fight is load-bearing.

        The trace-normalized variant (shape-only pressure) was built, run as
        VBG Term B, and lost 0-for-6; raw pressure completed and recovered
        breadth. This test fails if a future refactor quietly normalizes the
        surviving penalty and re-introduces the losing behaviour.
        """
        sk = self._sketch()
        z = torch.randn(4000, 64)
        assert (sketched_isotropy_penalty(z * 100.0, sk)
                > 10.0 * sketched_isotropy_penalty(z, sk))
