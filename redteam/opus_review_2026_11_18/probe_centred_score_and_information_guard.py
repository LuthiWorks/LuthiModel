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
3. Which guard sees a small pull (added for Amendment 7's draft). Made
   pulls at lam 0.3 and 0.01 under four kinds of prediction, for the
   similarity guard and for the targets' side of the centred score (A's
   prediction on the moved targets, less A's own score); then moved frames
   and real learning, which the targets' side should not take for a pull.

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



# 3. Added for Amendment 7's draft: which guard sees a small pull, and which a
# moved frame fools. Four kinds of prediction: a blurred copy of the targets
# (0.6 of their within-window spread) with a little or a lot of noise; an
# over-spread copy (1.4); and one made from position alone, blind to the token.
# The made pull of Amendment 6 (e'), T + lam (P - T), at lam 0.3 and 0.01.
# "Targets' side": A's prediction scored on the moved targets, less A's own
# centred score; negative when the targets moved so as to fit A's prediction.
import math

torch.manual_seed(0)
docs3 = 32
pos = torch.randn(Lt, d)


def tstat(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) or 1e-300
    return m / (s / math.sqrt(len(v)))


kinds = {
    "blurred copy, little noise": lambda T_, Tm: Tm + 0.6 * (T_ - Tm) + 0.25 * torch.randn_like(T_),
    "blurred copy, much noise": lambda T_, Tm: Tm + 0.6 * (T_ - Tm) + 1.0 * torch.randn_like(T_),
    "over-spread copy": lambda T_, Tm: Tm + 1.4 * (T_ - Tm) + 0.25 * torch.randn_like(T_),
    "from position alone": lambda T_, Tm: Tm + 0.8 * pos + 0.1 * torch.randn_like(T_),
}
made = []
for _ in range(docs3):
    tok = torch.randint(vocab, (W, Lt))
    ctx = torch.randn(W, 1, d) + 0.3 * torch.randn(W, Lt, d)
    T_ = emb[tok] @ mix + ctx + 0.5 * pos
    made.append((tok, T_, T_.mean(dim=1, keepdim=True)))
print("3. Made pulls: t over documents of each guard's change (a fall, or a gain on the targets' side, is negative)")
for name, make in kinds.items():
    Ps = [make(T_, Tm) for _, T_, Tm in made]
    for lam in (0.3, 0.01):
        sim_change, side = [], []
        for (tok, T_, _), P_ in zip(made, Ps):
            Tp = T_ + lam * (P_ - T_)
            sim_change.append(similarity([Tp], [tok]) - similarity([T_], [tok]))
            side.append(centred(P_, Tp) - centred(P_, T_))
        print(f"   {name:28s} lam {lam:<4}: similarity t {tstat(sim_change):+7.1f} | targets' side t {tstat(side):+9.1f}")
print("   Not pulls (blurred copy, little noise): targets' side, and the whole change of the centred score")
P0 = [kinds["blurred copy, little noise"](T_, Tm) for _, T_, Tm in made]
rows = {}
for (_, T_, _), P_ in zip(made, P0):
    Qr, _ = torch.linalg.qr(torch.eye(d) + 0.05 * torch.randn(d, d))
    for case, Tn, Pn in (("frame shifted, prediction with it", T_ + 1.0, P_ + 1.0),
                         ("frame rotated, prediction with it", T_ @ Qr, P_ @ Qr),
                         ("frame shrunk 1%, prediction with it", 0.99 * T_, 0.99 * P_),
                         ("the prediction learns, targets still", T_, P_ + 0.1 * (T_ - P_))):
        rows.setdefault(case, []).append((centred(P_, Tn) - centred(P_, T_), centred(Pn, Tn) - centred(P_, T_)))
for case, v in rows.items():
    side = [a for a, _ in v]
    whole = [b for _, b in v]
    print(f"   {case:38s} targets' side {sum(side) / len(side):+.5f} | whole {sum(whole) / len(whole):+.5f}")
