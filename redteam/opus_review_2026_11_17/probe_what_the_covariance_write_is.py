"""What the covariance write is, in closed form, checked against the code.

Opus 5.5 review of 2026-11-17 (Fable's relevant-updates design, stages 1 and 2).

The living layer's output is linear in its input, y = W x, and its error is
e = x - y P (``pc_ops``: ``err_tok = x_flat - output @ prediction``). So the
error is (I - P^T W) x: a function of the input alone, through slow states.
The covariance rule's write (``write_rule="covariance"``, before its norm
cap, with the per-element clamp not binding) is then

    write = W  S  (I - W^T P)  T,     S = the token covariance of the input,
                                      T = diag(trust) (precision, unless relative)

So what was lived enters only through S. A direction of the input that the
layer already reconstructs through P gets nothing, and an output direction no
token drove gets nothing.

This checks the identity on the real update (``_pc_self_modify_python``), with
everything but the write held still, and shows that two different token sets
with the same covariance write the same thing.

Run from the repo root:  python redteam/opus_review_2026_11_17/probe_what_the_covariance_write_is.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from luthi.v2.pc_ops import _pc_self_modify_python

torch.manual_seed(0)
D, N = 16, 64
W0 = torch.randn(D, D, dtype=torch.float64) * 0.3
P = torch.randn(D, D, dtype=torch.float64) * 0.3
trust = torch.rand(D, dtype=torch.float64) + 0.5


def code_write(x: torch.Tensor) -> torch.Tensor:
    """The weight change the real update makes, with everything else held still:
    no pull, no home following, no prediction learning, and an update average so
    large that the metaplasticity factor is exactly 1."""
    W = W0.clone()
    out = x @ W.T                                   # the layer's output, y = W x
    _pc_self_modify_python(
        weight=W, prediction=P.clone(), set_point=W0.clone(),
        momentum=torch.zeros(D, D, dtype=torch.float64),
        update_ema=torch.full((D, D), 1e9, dtype=torch.float64),
        precision=trust.clone(), error_acc=torch.zeros(D, dtype=torch.float64),
        plasticity=torch.ones(D, dtype=torch.float64),
        x_flat=x, output=out, pc_rate=1.0, pred_learning_rate=0.0,
        homeostatic_decay=0.0, set_point_adapt_rate=0.0, momentum_decay=0.9,
        update_ema_decay=0.99, precision_ema_decay=0.99, precision_min=1e-6,
        precision_max=1e6, prediction_clamp=1e6, write_rule="covariance",
    )
    return W - W0


def closed_form(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(dim=0, keepdim=True)
    S = xc.T @ xc / x.shape[0]
    return W0 @ S @ (torch.eye(D, dtype=torch.float64) - W0.T @ P) @ torch.diag(trust)


def capped(x: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
    """The rule's own cap: no larger than the mean-field write of the same batch."""
    out = x @ W0.T
    err = x.mean(0) - out.mean(0) @ P
    mf = torch.outer(out.mean(0), (err * trust).clamp(-1, 1)).norm()
    return write * (mf / write.norm()).clamp(max=1.0)


x = torch.randn(N, D, dtype=torch.float64) * 0.05 + torch.randn(D, dtype=torch.float64) * 0.05
w_code, w_form = code_write(x), capped(x, closed_form(x))
print(f"1. the real update against W S (I - W^T P) T, capped: relative difference "
      f"{float((w_code - w_form).norm() / w_code.norm()):.1e}")

# Different tokens, the same covariance: an orthogonal mix of the centred tokens
# that keeps their mean (Q fixes the all-ones vector).
xc = x - x.mean(0, keepdim=True)
ones = torch.ones(N, 1, dtype=torch.float64) / N ** 0.5
A = torch.randn(N, N, dtype=torch.float64)
A = A - ones @ (ones.T @ A)                         # orthogonal to the all-ones vector
Q, _ = torch.linalg.qr(torch.cat([ones, A[:, : N - 1]], dim=1))
Q = Q @ torch.diag(torch.sign(torch.diag(Q.T @ torch.cat([ones, A[:, : N - 1]], dim=1))))
mix = Q[:, 1:] @ torch.linalg.qr(torch.randn(N - 1, N - 1, dtype=torch.float64))[0] @ Q[:, 1:].T
x2 = x.mean(0, keepdim=True) + mix @ xc
print(f"2. a different set of tokens: {float((x2 - x).norm() / x.norm()):.0%} of the tokens' size changed; "
      f"same covariance to {float(((x2 - x2.mean(0)).T @ (x2 - x2.mean(0)) - xc.T @ xc).norm() / (xc.T @ xc).norm()):.0e}; "
      f"the writes differ by {float((code_write(x2) - w_code).norm() / w_code.norm()):.1e} of their size")
