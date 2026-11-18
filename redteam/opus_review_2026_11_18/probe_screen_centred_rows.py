"""How much does screening on centred rows tighten the screened search?

Opus 5.5, review of build brief 10 (2026-11-18). Chat Fable's sketch
(``redteam/fable_2026_11_18/probe_screened_search.py``) takes its 4-byte
product on raw rows, so its error bound grows with a raw row's length. With
summaries alike at 0.999, the part that decides the likeness, a row's distance
from the mean, is a small fraction of that length. This takes the product on a
4-byte copy of each row centred on the reference mean m0 (made in 8-byte
arithmetic, then rounded), adds back (mean - m0) . cue in 8-byte arithmetic,
and bounds the error by the centred copy's length. Both screens use the
corrected branch: the upper bound's formula is chosen by the sign of
num + err and the lower bound's by num - err (the sketch uses num). Same
stores, cues and moved mean as the sketch; the answers are checked against the
exact search.

No model. Made-up summaries at 768 wide.

Run: python redteam/opus_review_2026_11_18/probe_screen_centred_rows.py [rows]
"""
import sys

import torch

torch.manual_seed(1)
torch.set_num_threads(4)
d = 768
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
U = 2.0 ** -24


def store(kind):
    base = torch.randn(d, dtype=torch.float64)
    if kind == "walk":
        walk = (torch.randn(n, d, dtype=torch.float64) * 2e-4).cumsum(0)
        walk = walk / walk.norm(dim=1).max() * 0.045
        return (base / base.norm() + walk).to(torch.float32)
    if kind == "tight":
        return (base / base.norm() + torch.randn(n, d, dtype=torch.float64) * 1.6e-4).to(torch.float32)
    return torch.randn(n, d, dtype=torch.float64).to(torch.float32)


def exact(M, mean, cue):
    c = cue.to(torch.float64) - mean
    R = M.to(torch.float64) - mean
    nr = R.norm(dim=1) * c.norm()
    return torch.where(nr > 0, (R @ c) / nr, torch.zeros_like(nr))


def screen(S32, s_norm, n0, m0, mean, cue, shift):
    """S32: the 4-byte rows the pass runs on; s_norm: their lengths in 8-byte.
    shift: the 8-byte vector whose product with the cue turns S32's product
    into (r - mean) . c. Returns the rows to compute exactly."""
    c = cue.to(torch.float64) - mean
    cn = float(c.norm())
    c32 = c.to(torch.float32)
    dot = (S32 @ c32).to(torch.float64)
    err = (d + 2) * U * s_norm * float(c32.to(torch.float64).norm()) * 1.01
    num = dot - float(shift @ c)
    drift = float((mean - m0).norm())
    lo = (n0 - drift).clamp_min(0.0)
    hi = n0 + drift
    up = torch.where(num + err >= 0, (num + err) / (lo * cn).clamp_min(1e-300), (num + err) / (hi * cn)).clamp(max=1.0)
    low = torch.where(num - err >= 0, (num - err) / (hi * cn), (num - err) / (lo * cn).clamp_min(1e-300)).clamp(min=-1.0)
    return (up >= low.max()).nonzero().flatten()


for kind in ("walk", "tight", "random"):
    M = store(kind)
    M64 = M.to(torch.float64)
    m0 = M64.mean(0)
    rn = M64.norm(dim=1)
    n0 = (M64 - m0).norm(dim=1)
    C32 = (M64 - m0).to(torch.float32)             # one 4-byte copy, centred on m0
    c_norm = C32.to(torch.float64).norm(dim=1)
    extra = M[torch.randint(n, (n // 50,))].to(torch.float64)
    mean = (m0 * n + extra.sum(0)) / (n + n // 50)  # the mean has moved since m0, as in the sketch
    raw_k, cen_k, missed = [], [], 0
    for _ in range(30):
        i = int(torch.randint(n, (1,)))
        noise = {"walk": 1e-5, "tight": 1e-4, "random": 0.5}[kind]
        cue = (M64[i] + torch.randn(d, dtype=torch.float64) * noise).to(torch.float32)
        best = int(exact(M, mean, cue).argmax())
        a = screen(M, rn, n0, m0, mean, cue, mean)              # raw rows
        b = screen(C32, c_norm, n0, m0, mean, cue, mean - m0)   # centred rows
        missed += (best not in set(a.tolist())) + (best not in set(b.tolist()))
        raw_k.append(a.numel())
        cen_k.append(b.numel())
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"{kind:6s} n={n}: rows computed exactly, raw screen median {med(raw_k)} most {max(raw_k)} | "
          f"centred screen median {med(cen_k)} most {max(cen_k)} | best row missed {missed} of 60")
