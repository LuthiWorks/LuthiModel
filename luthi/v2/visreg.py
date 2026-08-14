"""VISReg: Variance-Invariance-Sketching Regularization.

Implementation of arXiv 2606.02572 (Wu, Balestriero, Levine) as the
replacement for SIGReg, per the external protocol
(docs/research/refs/2026-08-10_suggested-solutions-external.md, step 3)
and the design rulings in
docs/reviews/2026-08-10_pruning-and-visreg-brief-for-opus.md.

Why replace SIGReg
------------------
The paper's stated motivation is our measured disease: the Epps-Pulley
statistic's gradient DIMINISHES as the embedding collapses and
eventually vanishes -- exactly the "depth breaks the rescue" failure
the depth-8 record demonstrates (transit universal, rescue absent).
VISReg's sorted-quantile objective keeps a non-vanishing gradient in
near-collapsed states; test_collapse_gradient_nonvanishing pins that
property in code, because it is the property we are buying.

The three terms (formulas verified against the arXiv HTML 2026-08-11,
not transcribed from summaries)
-------------------------------------------------------------------
    L_scale  = (1/D) sum_j (1 - sigma_j(Z))^2          Eq. 1
    Z~       = Z / (sg(sigma) + eps)                   Eq. 2  (sg = stop-grad;
                                                       NO centering -- the
                                                       center term owns the mean)
    L_shape  = (1/K) sum_k || sort(Z~ w_k) - q_N ||^2  Eq. 5  (sum over N,
                                                       mean over K slices)
    L_center = || mu ||^2                              Eq. 6  (NOT divided by D)
    L_Reg    = ls*L_scale + lsh*L_shape + lc*L_center  Eq. 7  (defaults all 1.0)

Quantiles q_N are Normal(0,1).icdf(i/(N+1)) for i = 1..N -- the paper's
plotting positions, computed via erfinv on CPU and cached per N.

The top-level combination is the caller's job (jepa_loss.py):
    L_VISReg = (1 - lambda) * L_pred + lambda * L_Reg   Eq. 9  (CONVEX --
structurally different from the additive `l_pred + 0.2*l_sigreg` form
every prior run used; 0.6 is the paper's small-dataset recommendation).

Deliberate divergences from the paper's Algorithm 1, both recorded:
- Projections come from a dedicated CPU generator seeded by a
  ``global_step`` buffer, not bare ``torch.randn``. The paper's bare
  draw would advance the global RNG stream, the exact confound the
  2026-08-01 SIGReg fix removed; we do not reintroduce it.
- The generator lives on CPU and the draw is transferred, so a DML run
  and a CPU run see identical directions (DirectML has no device
  generators anyway).

DirectML status: torch.sort forward AND backward verified finite and
non-zero on privateuseone:0 by the build-seat feasibility read
(2026-08-10, return note (c)) -- the sliced-Wasserstein path is clear.

Input contract: ``forward(z)`` with z of shape (N, D) -- the flattened
per-position sample set, the same set SIGReg received. N must be >= 2
(per-dimension std needs it); fail loud otherwise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class VISReg(nn.Module):
    """Variance-Invariance-Sketching Regularizer (single-GPU).

    Args:
        num_proj: number of random slices K. Paper guidance: K = C*D with
            C > 1; at D=512 the registered value is 1024 (C=2).
        lambda_scale / lambda_shape / lambda_center: Eq. 7 component
            weights; paper default 1.0 each.
        eps: denominator guard in the Eq. 2 standardization.
    """

    def __init__(
        self,
        num_proj: int = 1024,
        lambda_scale: float = 1.0,
        lambda_shape: float = 1.0,
        lambda_center: float = 1.0,
        eps: float = 1e-4,
        shape_normalize: bool = False,
    ):
        super().__init__()
        self.num_proj = int(num_proj)
        self.lambda_scale = float(lambda_scale)
        self.lambda_shape = float(lambda_shape)
        self.lambda_center = float(lambda_center)
        self.eps = float(eps)
        # 2026-08-14 audit, item B1. Eq. 5 SUMS over N while L_scale means
        # over D, so l_shape scales with the sample count. At N = 32x128
        # that made the regularizer four to six orders of magnitude larger
        # than l_pred, and the convex Eq. 9 mix at the paper's lambda=0.6
        # put VISReg at 98.6-99.99% of the objective for an entire 54,000-
        # step run -- l_pred never exceeded 1.4%. Within the regularizer
        # itself the nominal 1.0/1.0/1.0 weighting resolved to ~95% shape,
        # ~5% scale, ~0.55% center, and center is the anti-offset term
        # while the observed disease was a soloist.
        #
        # The implementation is faithful to the paper; what was never
        # checked is what the paper's lambda MEANS at our N. This flag
        # makes l_shape a mean over N as well as over K, which makes
        # lambda a scale-free mixing weight and incidentally removes the
        # batch-size dose distortion the 2026-08-11 smoke measured
        # directly (l_shape 1,461,016 at batch 32 vs 693,472 at batch 16
        # -- the predicted ~2x, i.e. pure N-scaling).
        #
        # OPT-IN, default False, per this ladder's standing discipline:
        # every completed family's configuration must keep its meaning.
        # Turning it on is a registered change; see docs/DECISIONS.md
        # (2026-08-14) and it MUST be paired with a re-derived lambda,
        # because the same lambda over a differently-normalized term is a
        # different dose, not a correction.
        self.shape_normalize = bool(shape_normalize)
        # Same dedicated-generator discipline as SIGReg (2026-08-01):
        # per-step reproducible projections, global RNG stream untouched.
        self._generator: torch.Generator | None = None
        self.register_buffer(
            "global_step", torch.zeros((), dtype=torch.long), persistent=True
        )
        # Quantile cache: {n: cpu float32 tensor}. Not buffers -- N is a
        # data-shape property, not model state, and seq_len is fixed per
        # run so this holds one entry in practice.
        self._quantile_cache: dict[int, torch.Tensor] = {}

    def _projection(self, d: int, device, dtype) -> torch.Tensor:
        """(D, K) column-unit-normalized slice directions, per-step seeded."""
        if self._generator is None:
            self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(self.global_step.item()))
        a = torch.randn(
            d, self.num_proj, generator=self._generator,
            device="cpu", dtype=torch.float32,
        )
        # Advance only while training, mirroring SIGReg: eval must not
        # mutate model state (tests/test_heldout_eval.py's contract), and
        # a fixed eval step means every eval batch sees the same slices.
        if self.training:
            self.global_step.add_(1)
        a = a.to(device=device, dtype=dtype)
        return a / a.norm(p=2, dim=0)

    def _quantiles(self, n: int, device, dtype) -> torch.Tensor:
        """q_N: Normal(0,1).icdf(i/(N+1)), i = 1..N. CPU-computed, cached."""
        q = self._quantile_cache.get(n)
        if q is None:
            u = torch.arange(1, n + 1, dtype=torch.float64) / (n + 1)
            # icdf via erfinv on CPU: DirectML's erfinv support is
            # unverified and this is a one-time constant per N anyway.
            q = (math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)).to(torch.float32)
            self._quantile_cache[n] = q
        return q.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> dict:
        """z: (N, D) sample set. Returns component dict, all grad-connected.

        Keys: ``l_reg`` (the Eq. 7 weighted sum -- add this to the loss),
        ``l_scale``, ``l_shape``, ``l_center`` (unweighted components,
        for the per-step record; a term you cannot read you cannot dose).
        """
        if z.dim() != 2:
            raise ValueError(
                f"VISReg expects (N, D); got shape {tuple(z.shape)}. Flatten "
                "positions into the sample dim before calling."
            )
        n, d = z.shape
        if n < 2:
            raise ValueError(
                f"VISReg needs N >= 2 samples for per-dimension std; got {n}."
            )

        mu = z.mean(dim=0)                       # (D,)
        sigma = z.std(dim=0)                     # (D,) centered, unbiased

        # Eq. 1 -- scale: press every dimension's std toward 1. Applied to
        # TRUNK latents whose native band is 0.25-0.35; the pressure toward
        # unit is deliberate and evidence-backed (the wsig alpha=10 arrest
        # worked partly BY pressing scale up; trace-normalized lost 0-for-6).
        l_scale = (1.0 - sigma).square().mean()

        # Eq. 6 -- center: the batch mean at the origin. The direct
        # instrument against offset dominance, the measured "first act" of
        # every collapse in the record.
        l_center = mu.square().sum()

        # Eq. 2 + 5 -- shape: per-dim scale divided out under stop-grad
        # (decoupling: scale errors are L_scale's job), NO centering, then
        # K 1-D marginals each matched to N(0,1) order statistics. Sorting
        # keeps gradient signal in near-collapsed states where the
        # Epps-Pulley CF statistic's gradient vanishes -- the property this
        # module exists for.
        z_tilde = z / (sigma.detach() + self.eps)
        a = self._projection(d, z.device, z.dtype)   # (D, K)
        proj = z_tilde @ a                           # (N, K)
        sorted_proj, _ = proj.sort(dim=0)
        q = self._quantiles(n, z.device, z.dtype).unsqueeze(1)  # (N, 1)
        # Eq. 5 sums over N and means over K. `shape_normalize` means over
        # N too -- see the ctor for why the sum makes lambda depend on
        # batch size and buries l_pred.
        if self.shape_normalize:
            l_shape = (sorted_proj - q).square().mean(dim=0).mean()
        else:
            l_shape = (sorted_proj - q).square().sum(dim=0).mean()

        l_reg = (
            self.lambda_scale * l_scale
            + self.lambda_shape * l_shape
            + self.lambda_center * l_center
        )
        return {
            "l_reg": l_reg,
            "l_scale": l_scale,
            "l_shape": l_shape,
            "l_center": l_center,
        }
