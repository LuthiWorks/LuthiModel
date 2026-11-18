"""Does the screened search keep the best row it may return, when the nearest rows are excluded?

Opus 5.5, review of build brief 10 (2026-11-18). The store's ``nearest``
(Sanctuary ``sanctuary/life/memory.py``) leaves out every memory still the
present, within ``recent`` moments, and takes the best of the rest. Chat
Fable's sketch (``redteam/fable_2026_11_18/probe_screened_search.py``) keeps a
row for exact computation when its upper bound reaches the best lower bound
over *every* row. After the mind has moved, the newest memories are the most
like the cue, and they are the ones left out. This makes that case: the
newest 100 summaries in a new region, the cue among them, the newest 100 rows
counted as recent. It compares the bar taken over every row with the bar taken
over the rows the search may return.

No model. Made-up summaries at 768 wide, as in the sketch.

Run: python redteam/opus_review_2026_11_18/probe_screen_bar_over_eligible_rows.py
"""
import torch

torch.manual_seed(2)
torch.set_num_threads(4)
d, n, U = 768, 50000, 2.0 ** -24

base = torch.randn(d, dtype=torch.float64)
base /= base.norm()
walk = (torch.randn(n, d, dtype=torch.float64) * 2e-4).cumsum(0)
walk = walk / walk.norm(dim=1).max() * 0.045
jump = torch.randn(d, dtype=torch.float64)
walk[-100:] += jump / jump.norm() * 0.03          # the newest memories, after a move
M = (base + walk).to(torch.float32)
M64 = M.to(torch.float64)
m0 = M64.mean(0)
mean = m0                                         # no drift here: the case does not need it
rn = M64.norm(dim=1)
n0 = (M64 - m0).norm(dim=1)
recent = torch.zeros(n, dtype=torch.bool)
recent[-100:] = True
eligible = ~recent

cue = (M64[-1] + torch.randn(d, dtype=torch.float64) * 1e-5).to(torch.float32)   # the present is like the newest
c = cue.to(torch.float64) - mean
cn = float(c.norm())
c32 = c.to(torch.float32)
R = M64 - mean
exact = (R @ c) / (R.norm(dim=1) * cn)

# The sketch's screen: a 4-byte product and the classical bound (no drift, so one branch).
dot = (M @ c32).to(torch.float64)
err = (d + 2) * U * rn * float(c32.to(torch.float64).norm()) * 1.01
num = dot - float(mean @ c)
up = ((num + err) / (n0 * cn)).clamp(max=1.0)
low = ((num - err) / (n0 * cn)).clamp(min=-1.0)

best = int(torch.where(eligible, exact, torch.full_like(exact, -2.0)).argmax())
print(f"likeness: best recent row {float(exact[recent].max()):.3f}, best row the search may return {float(exact[best]):.3f}")
for name, bar in (("bar over every row (the sketch)", low.max()), ("bar over the rows it may return", low[eligible].max())):
    candidates = ((up >= bar) & eligible).nonzero().flatten()
    print(f"{name:32s}: candidates it may return {candidates.numel():5d}; "
          f"the best among them: {best in set(candidates.tolist())}")
