# SIGReg cross-checked against the LeJEPA reference implementation

**Date:** 2026-08-01
**Reference:** `rbalestr-lab/lejepa` (Balestriero & LeCun), cloned from GitHub.
**Ours:** `luthi/v2/sigreg.py`, ported from le-wm 2026-06-09, never previously
validated against anything but itself.

Note: `pip install lejepa` fails -- the package is not on PyPI under that name.
The repo clones fine and `MINIMAL.md` carries the entire implementation in ~20
lines, so no install and no environment risk were needed.

## Verdict: our port is faithful. Verbatim, including variable names.

The reference's minimal SIGReg (`MINIMAL.md`):

```python
t = torch.linspace(0, 3, knots, dtype=torch.float32)
dt = 3 / (knots - 1)
weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
weights[[0, -1]] = dt
window = torch.exp(-t.square() / 2.0)
...
x_t = (proj @ A).unsqueeze(-1) * self.t
err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
statistic = (err @ self.weights) * proj.size(-2)
return statistic.mean()
```

Point by point against `luthi/v2/sigreg.py`:

| element | reference | ours | |
|---|---|---|---|
| integration grid | `linspace(0, 3, 17)` | same | match |
| `dt` | `3 / (knots - 1)` | same | match |
| trapezoid weights | `2*dt`, endpoints `dt` | same | match |
| `phi` (CF target) | `exp(-t^2/2)` | same | match |
| integration weights | `weights * window` -- the SAME `exp(-t^2/2)` used twice | same | match |
| projection | `randn(D, K)`, `div_(norm(p=2, dim=0))` | same | match |
| error form | `(cos_mean - phi)^2 + sin_mean^2` | same | match |
| scaling | `* proj.size(-2)` (sample dim) | same | match |
| `num_proj` | 256 in the minimal example | 1024 default | documented knob |

The package version (`lejepa/univariate/epps_pulley.py` +
`multivariate/slicing.py`) matches too, with DDP `all_reduce` and a
`world_size` factor we do not need single-GPU.

**SIGReg itself was never the problem.**

## The BatchNorm question is settled, and against our old code

Our `sigreg.py` docstring asserted until 2026-07-31 that SIGReg's input "must be
~standardized", that this is "the projection head's job (Linear -> BatchNorm)",
and that "the paper warns about this."

The reference does the opposite. Its projector:

```python
self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
```

Expanded (verified by instantiating torchvision's MLP):

```
Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Dropout
                                                                          ^
                                                        SIGReg sees THIS: a bare Linear
```

**BatchNorm appears only in the projector's HIDDEN layers. There is none between
the final Linear and SIGReg.** And `Slicing.forward` applies no standardization
either -- it projects and calls the test directly.

So:

* our old default (`Linear -> BatchNorm1d -> SIGReg`) put a normalization layer
  exactly where the reference deliberately has none;
* our current default since 2026-07-28 (`Linear -> SIGReg`) **matches the
  reference architecture**;
* the claim "the paper warns about this" was backwards. The reference warns
  about it by not doing it.

The 07-28 fix was justified on measurement at the time (SIGReg blind to a 100x
shrink post-BN: 0.566 -> 0.545). It is now independently confirmed as restoring
upstream behaviour rather than merely improving ours.

## What this does and does not tell us about the depth problem

**Does:** SIGReg is not implicated. The statistic is upstream-correct and the
projection is upstream-correct. Any remaining collapse is not a broken
regularizer.

**Does not:** anything about the depth-8 rank collapse. That remains open, and
the reference offers no direct guidance -- LeJEPA is trained on images with a
ViT backbone, not on text with a self-modifying PC trunk.

One structural difference worth noting rather than acting on: the reference
projector is a 3-layer MLP (512 -> 2048 -> 2048 -> 128) with internal BatchNorm
and nonlinearity. Ours is a single `nn.Linear(d_model, d_model)`. Whether SIGReg
behaves differently when fed a wide nonlinear projection versus a bare linear one
is untested here and is a real difference from upstream -- just not the one we
went looking for.
