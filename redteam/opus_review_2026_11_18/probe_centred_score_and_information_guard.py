"""The centred score's invariances, and which information guard a moved frame cannot fail.

Opus 5.5, for Amendment 6's second draft (2026-11-18). Made-up tensors, no
model. Two checks:

1. The centred score: per window, remove the window's mean from the
   prediction and from the targets separately, then the mean squared
   difference over the targets' own within-window variance. It should not
   move under a shift of either side or both, or a common rescaling or
   rotation. It should fall when the prediction learns the content, and
   also (wrongly, for E1) when each target is pulled toward its own
   prediction.
2. The information guard, for that last case. A linear probe from a target
   to its token's input embedding, pooled over documents, against
   representational similarity within each window (the correlation of the
   targets' pairwise distances with the tokens' pairwise distances). In E1
   each document is read by its own model, so each has its own frame. The
   check moves each document's frame (rotation, scale, shift) and,
   separately, pulls each target 30% of the way toward its prediction, which
   is made from context and blind to the token.

Run: python redteam/opus_review_2026_11_18/probe_centred_score_and_information_guard.py
"""
import torch

torch.manual_seed(0)


def centred(P, T):
    """P, T: [windows, predicted positions, width]."""
    Pc = P - P.mean(dim=1, keepdim=True)
    Tc = T - T.mean(dim=1, keepdim=True)
    return float((Pc - Tc).pow(2).mean() / Tc.pow(2).mean())


# 1. The centred score.
W, L, d = 8, 32, 64
T = torch.randn(W, L, d) + 3.0
P = T + 0.5 * torch.randn(W, L, d) + 0.7
s = torch.randn(W, 1, d)                        # a different shift for each window
Q, _ = torch.linalg.qr(torch.randn(d, d))
print("1. The centred score")
for name, value in (
    ("as made", centred(P, T)),
    ("targets shifted", centred(P, T + s)),
    ("prediction shifted", centred(P + 2 * s, T)),
    ("both shifted", centred(P + s, T + s)),
    ("both scaled by 3", centred(3 * P, 3 * T)),
    ("both rotated", centred(P @ Q, T @ Q)),
    ("prediction 20% of the way to each target (content learned)", centred(P + 0.2 * (T - P), T)),
    ("each target 20% of the way to its prediction (bent)", centred(P, T + 0.2 * (P - T))),
):
    print(f"   {name:60s} {value:.6f}")

# 2. The information guard.
d, vocab, docs, W, Lt = 64, 500, 24, 8, 32
emb = torch.randn(vocab, d)                     # A's input embeddings: what arrived
mix = torch.randn(d, d) / d ** 0.5


def make_doc():
    tok = torch.randint(vocab, (W, Lt))
    ctx = torch.randn(W, 1, d).expand(W, Lt, d) + 0.3 * torch.randn(W, Lt, d)
    return tok, emb[tok] @ mix + ctx, ctx + 0.2 * torch.randn(W, Lt, d)   # tokens, targets, predictions


def probe_error(targets, toks, folds=8):
    """Ridge from a target to its token's input embedding, fitted on the other folds' documents."""
    errs = []
    for i in range(len(targets)):
        fit = [j for j in range(len(targets)) if j % folds != i % folds]
        X = torch.cat([targets[j].reshape(-1, d) for j in fit])
        Y = torch.cat([emb[toks[j]].reshape(-1, d) for j in fit])
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        lam = 0.01 * float((Xc * Xc).sum()) / d
        Wp = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(d), Xc.T @ Yc)
        pred = (targets[i].reshape(-1, d) - mx) @ Wp + my
        errs.append(float((pred - emb[toks[i]].reshape(-1, d)).pow(2).mean() / emb.var()))
    return sum(errs) / len(errs)


def similarity(targets, toks):
    """Per window: Pearson correlation of the targets' pairwise distances with the
    tokens' pairwise input-embedding distances; the mean over windows and documents."""
    iu = torch.triu_indices(Lt, Lt, 1)
    out = []
    for T_, tok in zip(targets, toks):
        for w in range(W):
            a = torch.cdist(T_[w], T_[w])[iu[0], iu[1]]
            b = torch.cdist(emb[tok[w]], emb[tok[w]])[iu[0], iu[1]]
            out.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    return sum(out) / len(out)


data = [make_doc() for _ in range(docs)]
toks = [t for t, _, _ in data]
base = [T_ for _, T_, _ in data]


def moved(T_):
    Q_, _ = torch.linalg.qr(torch.randn(d, d))
    return 1.3 * (T_ @ Q_) + torch.randn(d) * 2.0


print("2. The information guard (probe error: lower holds more of what arrived; similarity: higher does)")
for name, ts in (
    ("A's targets", base),
    ("each document's frame moved", [moved(T_) for T_ in base]),
    ("each target 30% of the way to its prediction", [0.7 * T_ + 0.3 * P_ for _, T_, P_ in data]),
):
    print(f"   {name:46s} probe error {probe_error(ts, toks):.3f}   similarity {similarity(ts, toks):.3f}")
