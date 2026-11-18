"""Is a row's 8-byte dot product the same, bit for bit, in any batch?

Opus 5.5, review of build brief 10 (2026-11-18). The screened search computes
its candidates in a small batch; the store's full search computes chunks of
1,024 rows (``_likeness`` in Sanctuary ``sanctuary/life/memory.py``). "The
same likeness, bit for bit" needs a row's dot product not to change with the
rows around it. Some BLAS builds take other code paths for the last rows of a
block. This compares each row's value in its 1,024-row chunk with its value in
random subsets of odd sizes, for the matrix product and for a multiply and
sum, at 1, 4 and 16 threads. A pass here says this build behaves; the test
that matters is the same one on the machine the life runs on.

No model.

Run: python redteam/opus_review_2026_11_18/probe_dot_same_in_any_batch.py
"""
import torch

torch.manual_seed(0)
d = 768
M = torch.randn(4096, d, dtype=torch.float64) * 1e-3 + 1.0
c = torch.randn(d, dtype=torch.float64) * 1e-3
for threads in (1, 4, 16):
    torch.set_num_threads(threads)
    for name, f in (("matrix product", lambda R: R @ c), ("multiply and sum", lambda R: (R * c).sum(dim=1))):
        full = torch.cat([f(M[a:a + 1024]) for a in range(0, 4096, 1024)])
        differ = tried = 0
        g = torch.Generator().manual_seed(1)
        for k in (1, 2, 3, 5, 7, 13, 64, 487):
            for _ in range(20):
                idx = torch.randperm(4096, generator=g)[:k]
                differ += int((f(M[idx]) != full[idx]).sum())
                tried += k
        print(f"threads {threads:2d}, {name:16s}: rows whose value in a subset differs from its chunk: {differ} of {tried}")
print(f"torch {torch.__version__}")
