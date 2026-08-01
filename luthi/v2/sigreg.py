"""SIGReg: Sketch Isotropic Gaussian Regularizer.

Port from le-wm (Maes / LeCun; LeJEPA family, Balestriero & LeCun).
Replaces the EMA target + VICReg apparatus 4.8 approved for refactor
2026-06-09. Reference: ``le-wm/module.py`` and ``le-wm/train.py``
(defaults lambda=0.1, knots=17, num_proj=1024).

What it does
------------
Project the batch onto ``num_proj`` random unit directions; on each
1-D projection, compute the Epps-Pulley statistic -- squared diff
between the empirical characteristic function (mean of cos/sin over
the batch dim) and the standard-Gaussian CF (real ``exp(-t^2/2)``,
imag 0), integrated over t in [0, 3] (trapezoidal with Gaussian
window).

By Cramer-Wold: all 1-D marginals Gaussian <=> joint isotropic
Gaussian. Constant input -> can't be Gaussian (no complete collapse);
isotropic prior forces unit-variance and zero-covariance (no
dimensional collapse).

Critical input contract -- CORRECTED 2026-07-31
-----------------------------------------------
The paragraph that stood here was **backwards**, and it survived three
days past its own refutation because the 2026-07-28 correction was
applied to ``jepa_loss.py`` and not to this file. It read:

    "SIGReg targets N(0,1), so its input must be ~standardized. That is
     the projection head's job (Linear -> BatchNorm) ... SIGReg must run
     on a separate BN-projected head, NOT on the LayerNorm'd trunk
     output. The paper warns about this; we honor it in jepa_loss.py."

Wrong on both counts, and **the attribution to the paper should not be
trusted** -- that was our inference, presented as the paper's warning.

**Why it is wrong.** BatchNorm subtracts the batch mean and divides by
the batch std -- precisely the two quantities SIGReg exists to
constrain. Pre-standardizing its input hands it a solved problem and the
anti-collapse term stops binding on the encoder. Measured with THIS
module, not a reimplementation: under a 100x uniform shrink, SIGReg on
raw latents moves 0.86 -> 706 (it fights); after BatchNorm 0.566 ->
0.545, a 3.7% move (it is blind). Under an offset sweep 0 -> 0.995, raw
goes 1.0 -> 2111, post-BN 0.563 -> 0.567.

SIGReg does NOT require standardized input. It requires the input whose
distribution you actually want to constrain. Default is now
``sigreg_projection="linear"``; ``"linear_bn"`` retains the old
behaviour for A/B only.

**Also wrong:** the trunk's ``final_norm`` is applied only in the
LM-style ``forward()``, never in ``encode()``, so the JEPA objective
never sees it. The concern about it "washing out the distribution" was
about a layer that is not on this path.

Known remaining degree of freedom (measured 2026-07-31): the Linear head
still absorbs **scale** -- singular values run 0.42-0.55, so a trunk at
std ~3.0 presents to SIGReg at ~1.27. That is why ``std_p5`` (trunk) and
``L_sigreg`` (projected) can disagree. Not currently treated as a
defect; recorded so it is not rediscovered.

CROSS-CHECKED against the reference 2026-08-01 (`rbalestr-lab/lejepa`;
note `pip install lejepa` fails -- never published to PyPI -- but the
repo clones and installs from source). **Bit-identical**: buffers ``t``,
``phi`` and ``weights`` match to 0.0, and the statistic matches to
relative difference 0.00e+00 on isotropic N(0,1), a 100x shrink, a +3.0
offset, and a rank-2 degenerate input -- the last being the regime our
depth-8 runs actually occupy.

The reference also settles the BatchNorm question against our old code:
its projector is ``MLP(512, [2048, 2048, proj_dim],
norm_layer=BatchNorm1d)``, which places BatchNorm only in the HIDDEN
layers -- SIGReg sees a bare final Linear. Our current default matches;
the old one did not.

One real divergence remains, unexplored: their projector is a 3-layer
MLP with internal nonlinearity, ours is a single ``nn.Linear``. Whether
SIGReg behaves differently fed a wide nonlinear projection is untested.
See docs/research/2026-08-01_sigreg-verified-against-reference.md.

See docs/research/2026-07-30_mupc-verdict.md and the corrected block in
jepa_loss.py.

Shape: forward expects ``(T, B, D)`` where T = temporal/position
group, B = sample dim (mean is taken over this), D = embedding dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).

    Args:
        knots: number of trapezoidal integration nodes over t in [0, 3].
        num_proj: number of random unit-vector projections per call.
    """

    def __init__(
        self,
        knots: int = 17,
        num_proj: int = 1024,
        use_generator: bool = True,
    ):
        super().__init__()
        self.num_proj = num_proj
        # Dedicated-generator mode (2026-08-01), matching the reference's
        # `SlicingUnivariateTest`. **Default True.**
        #
        # THE DEFECT IT FIXES, measured: our projection draw uses bare
        # `torch.randn`, so every SIGReg forward advances the GLOBAL RNG
        # stream -- the same stream that feeds data sampling, dropout and
        # every other stochastic component. Confirmed: the next global draw
        # after a forward differs from the draw without one, and it differs
        # AGAIN between num_proj=256 and num_proj=1024.
        #
        # The consequence is not that completed runs are wrong -- they all used
        # one num_proj and the seed44 byte-identical rerun did reproduce. It is
        # that **num_proj has never been a clean single variable**: changing it
        # changes how much RNG SIGReg consumes, which shifts the entire
        # downstream random sequence including data order. Any experiment on
        # SIGReg's own parameters would have been confounded and nothing would
        # have reported it.
        #
        # The reference seeds a dedicated generator from a `global_step`
        # counter, so projections are reproducible per step and the global
        # stream is untouched. Same design here.
        #
        # ON WHY THIS DEFAULTS TO TRUE. It was written default-False to
        # preserve bit-identity with prior runs. Brian caught that this
        # contradicts the principle set on 2026-07-28, when the JEPA objective
        # fix shipped default-ON with the reasoning: "leaving a verified defect
        # as the default is worse than breaking comparability with runs
        # produced under it."
        #
        # That reasoning applies here with more force, not less. The objective
        # defect at least showed up in the numbers eventually. RNG-stream
        # coupling is SILENT -- it would surface only as an unattributable
        # difference in some future experiment, which is the invisible-default
        # failure argued against at length in the relative_trust
        # recommendation.
        #
        # Enabling it changes the random stream, so runs from here are not
        # bit-comparable with earlier ones. That boundary already exists at the
        # 07-28 objective fix, and every depth-8 run postdates it, so this is
        # the cheapest possible moment to take the break.
        #
        # `use_generator=False` is retained to reproduce pre-2026-08-01
        # behaviour exactly.
        self.use_generator = bool(use_generator)
        self._generator = None
        self._generator_device = None
        self.register_buffer(
            "global_step", torch.zeros((), dtype=torch.long), persistent=True
        )
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        # Trapezoidal weights: 2*dt for interior nodes, dt for endpoints.
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        # Standard-Gaussian CF real part: exp(-t^2/2). Imag part is 0.
        # Also serves as a Gaussian window down-weighting the tail.
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # CF target (real part)
        self.register_buffer("weights", weights * window)

    def _projection(self, d: int, device, dtype) -> torch.Tensor:
        """Draw the (D, num_proj) projection matrix.

        ``use_generator=False`` reproduces the legacy path exactly: a bare
        ``torch.randn`` on the global stream.

        ``use_generator=True`` draws from a dedicated generator seeded by
        ``global_step``, so the global stream is untouched and the projections
        for a given step are reproducible.

        The generator is created on CPU and the result transferred, rather than
        on ``device``. DirectML does not support device generators, and a CPU
        generator has the additional property that the projections are
        identical across backends -- a DML run and a CPU run see the same
        directions, which the device-generator version could not offer.
        """
        if not self.use_generator:
            return torch.randn(d, self.num_proj, device=device, dtype=dtype)

        if self._generator is None:
            self._generator = torch.Generator(device="cpu")
            self._generator_device = "cpu"
        self._generator.manual_seed(int(self.global_step.item()))
        a = torch.randn(
            d, self.num_proj, generator=self._generator,
            device="cpu", dtype=torch.float32,
        )
        # Advance ONLY while training. `global_step` is a persistent buffer, so
        # incrementing it during evaluation would make held-out eval mutate
        # model state -- caught immediately by
        # tests/test_heldout_eval.py::TestEvalMutatesNothing, which exists
        # because a read-only eval that silently writes is exactly the
        # silent-success failure this project keeps finding.
        #
        # Holding the step fixed in eval has a second benefit: every eval batch
        # sees the SAME projection directions, so the metric is not jittered by
        # resampling between batches. The reference increments unconditionally;
        # it does not carry our eval-purity constraint.
        if self.training:
            self.global_step.add_(1)
        return a.to(device=device, dtype=dtype)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: ``(T, B, D)`` standardized embeddings.

        Returns: scalar tensor -- mean SIGReg statistic over T and
        ``num_proj`` directions. Lower is closer to isotropic N(0,1).
        Add to total loss with a weight (LeWM default lambda=0.1).
        """
        # Random unit-vector projection matrix (D, num_proj).
        A = self._projection(proj.size(-1), proj.device, proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        # Project (T, B, D) @ (D, num_proj) -> (T, B, num_proj), then
        # multiply by t-grid -> (T, B, num_proj, knots).
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Empirical CF: mean over B (dim -3 of x_t). Shape (T, num_proj, knots).
        # Squared error against standard-Gaussian CF (real phi, imag 0).
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # Trapezoidal integrate over knots: (T, num_proj, knots) @ (knots,)
        # -> (T, num_proj). Multiply by B (chi-square-like scaling).
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Unit test: prove SIGReg fires correctly on synthetic distributions before
# wiring it into the substrate. Per 4.8 build-order step 1: this is the
# first thing to look at, before any jepa_loss refactor.
# ---------------------------------------------------------------------------


def _unit_test() -> int:
    torch.manual_seed(42)

    # Reduced num_proj for a faster unit test; default 1024 is for training.
    sigreg = SIGReg(knots=17, num_proj=256)

    B = 1024  # large enough for the empirical CF to estimate stably
    D = 64
    T = 1     # one position-group for the test

    print("=" * 64)
    print("SIGReg unit test")
    print("=" * 64)

    cases: list[tuple[str, torch.Tensor]] = []

    # 1. Isotropic standard Gaussian -- should be near zero.
    cases.append((
        "Gaussian N(0,1):           ",
        torch.randn(T, B, D),
    ))

    # 2. Constant input -- catastrophic collapse case.
    cases.append((
        "Constant (all zeros):      ",
        torch.zeros(T, B, D),
    ))

    # 3. Rank-1 input -- complete dimensional collapse.
    direction = torch.randn(D)
    direction = direction / direction.norm()
    coeffs = torch.randn(T, B, 1)
    cases.append((
        "Rank-1 (single direction): ",
        coeffs * direction.view(1, 1, D),
    ))

    # 4. Scaled Gaussian (wrong variance).
    cases.append((
        "N(0, 9) (wrong variance):  ",
        3.0 * torch.randn(T, B, D),
    ))

    # 5. Shifted Gaussian (wrong mean).
    cases.append((
        "N(1, 1) (wrong mean):      ",
        torch.randn(T, B, D) + 1.0,
    ))

    # 6. Partial dimensional collapse -- rank D/2.
    proj_mat = torch.randn(D // 2, D)
    proj_mat = proj_mat / proj_mat.norm(dim=1, keepdim=True)
    z = torch.randn(T, B, D // 2)
    cases.append((
        "Rank D/2 (partial collapse):",
        z @ proj_mat,
    ))

    results: dict[str, float] = {}
    for label, x in cases:
        val = float(sigreg(x).item())
        results[label] = val
        print(f"  {label} SIGReg = {val:.6f}")

    print("=" * 64)
    s_gauss = results["Gaussian N(0,1):           "]
    print(f"  Ratio constant / Gaussian        : "
          f"{results['Constant (all zeros):      '] / s_gauss:>8.2f}x")
    print(f"  Ratio rank-1 / Gaussian          : "
          f"{results['Rank-1 (single direction): '] / s_gauss:>8.2f}x")
    print(f"  Ratio N(0,9) / Gaussian          : "
          f"{results['N(0, 9) (wrong variance):  '] / s_gauss:>8.2f}x")
    print(f"  Ratio N(1,1) / Gaussian          : "
          f"{results['N(1, 1) (wrong mean):      '] / s_gauss:>8.2f}x")
    print(f"  Ratio rank D/2 / Gaussian        : "
          f"{results['Rank D/2 (partial collapse):'] / s_gauss:>8.2f}x")

    # Assertions: SIGReg should fire on every degeneracy. Conservative
    # bounds because the constant/rank-1 cases produce extreme ratios and
    # the partial-collapse case a more modest one.
    fail = False
    if not results["Constant (all zeros):      "] > s_gauss * 10:
        print(
            f"FAIL: constant SIGReg "
            f"({results['Constant (all zeros):      ']:.4f}) "
            f"not >> Gaussian ({s_gauss:.4f})"
        )
        fail = True
    if not results["Rank-1 (single direction): "] > s_gauss * 10:
        print(f"FAIL: rank-1 SIGReg not >> Gaussian")
        fail = True
    if not results["N(0, 9) (wrong variance):  "] > s_gauss * 5:
        print(f"FAIL: N(0,9) SIGReg not >> Gaussian")
        fail = True
    if not results["N(1, 1) (wrong mean):      "] > s_gauss * 5:
        print(f"FAIL: N(1,1) SIGReg not >> Gaussian")
        fail = True
    if not results["Rank D/2 (partial collapse):"] > s_gauss * 3:
        print(f"FAIL: rank-D/2 SIGReg not >> Gaussian")
        fail = True

    if fail:
        print("UNIT TEST FAILED.")
        return 1
    print("UNIT TEST PASSED.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_unit_test())
