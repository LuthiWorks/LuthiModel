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

Critical input contract
-----------------------
SIGReg targets ``N(0, 1)``, so its input must be ~standardized. That
is the projection head's job (Linear -> BatchNorm). The encoder's
final LayerNorm in our trunk would wash out the distribution SIGReg
shapes -- SIGReg must run on a separate BN-projected head, NOT on
the LayerNorm'd trunk output. The paper warns about this; we honor
it in jepa_loss.py.

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

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
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

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: ``(T, B, D)`` standardized embeddings.

        Returns: scalar tensor -- mean SIGReg statistic over T and
        ``num_proj`` directions. Lower is closer to isotropic N(0,1).
        Add to total loss with a weight (LeWM default lambda=0.1).
        """
        # Random unit-vector projection matrix (D, num_proj).
        A = torch.randn(
            proj.size(-1), self.num_proj,
            device=proj.device, dtype=proj.dtype,
        )
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
