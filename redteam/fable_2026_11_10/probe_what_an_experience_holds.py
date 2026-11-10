"""Capacity check (Fable 5.2, 2026-11-10): what should one experience hold?

For the two proposals in docs/reviews/2026-11-10_follow-up-from-opus.md.
Run from the LuthiModel root, with Sanctuary importable (about a minute on CPU):

    PYTHONPATH=<Sanctuary>:<LuthiModel> LUTHI_PATH=<LuthiModel> \\
        python redteam/fable_2026_11_10/probe_what_an_experience_holds.py
    D=64 SEED=7 ... (the second configuration below)

A two-block living model trained for 300 batches lives 300 moments on
Sanctuary's small path. Someone speaks every third moment, alternating two
different sentences (A and B). For each living layer and each lived moment,
three candidate vectors for "the experience":

  whole   mean of the layer's input over every token in view    (stored today)
  new     mean over the tokens that joined the view this moment (proposal 1a)
  change  whole(this moment) - whole(last moment)               (proposal 1b)

Columns: median cosine between pairs of moments of each kind; leave-one-out
nearest-centroid accuracy at telling a spoken moment from a quiet one and
sentence A from sentence B; and the median size of a spoken moment's vector
over a quiet one's.

Printed on 2026-11-10 (32 wide, seed 42; view 32 tokens; 13.5 new when
spoken, 1 when quiet):

    blocks.0   all pairs  quiet~quiet   A~A    A~B   spoken~quiet  spoken/quiet  A/B   size
    whole        0.989      0.989      0.993  0.990     0.989          76%       97%   1.0
    new          0.900      0.999      0.994  0.888     0.397         100%      100%   0.4
    change      -0.048     -0.083      0.995  0.069    -0.271         100%      100%   0.9
    blocks.1
    whole        0.992      0.992      0.993  0.992     0.992          80%       72%   1.0
    new          0.923      0.985      0.993  0.963     0.514         100%      100%   0.6
    change      -0.139      0.152      0.961  0.283    -0.419          83%      100%   1.1

and (64 wide, seed 7):

    blocks.0
    whole        0.931      0.928      0.930  0.933     0.932          68%       77%   1.0
    new          0.711      0.994      0.945  0.709     0.225          98%       98%   0.3
    change      -0.034      0.010      0.911  0.058    -0.137          96%       95%   1.0
    blocks.1
    whole        0.903      0.902      0.900  0.904     0.904          50%       33%   1.0
    new          0.716      0.940      0.891  0.832     0.574          98%       94%   0.5
    change      -0.071      0.013      0.806  0.187    -0.186          95%       96%   1.0

Reading: see docs/reviews/2026-11-10_on-gauging-a-moment-from-fable.md.
"""
import os, random, sys, tempfile, types, statistics
from argparse import Namespace
from pathlib import Path
os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
ROOT = Path(tempfile.mkdtemp(prefix="gauge_"))
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")
import torch
LUTHI = Path(os.environ["LUTHI_PATH"]); sys.path.insert(0, str(LUTHI))
import scripts.jepa_pilot_driver as driver
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from sanctuary.life import LifeLoop, LuthiMind, TextWorld

D = int(os.environ.get("D", "32")); BATCHES = int(os.environ.get("BATCHES", "300")); SEED = int(os.environ.get("SEED", "42"))
def train():
    corpus = ROOT / "corpus"; corpus.mkdir()
    rng = random.Random(0)
    words = ("river stone light memory window garden quiet music signal field morning "
             "shadow pattern voice letter bridge winter colour thread").split()
    (corpus / "text.txt").write_text(" ".join(rng.choice(words) for _ in range(40000)), encoding="utf-8")
    driver.OUTPUT_ROOT = ROOT / "runs"
    driver._run_one("living", D, SEED, Namespace(
        data_dir=str(corpus), tokenizer=str(LUTHI / "corpus_build" / "tokenizer_32k.json"),
        n_blocks=2, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
        holdout_fraction=0.2, heldout_batches=2, probe_batches=2, kill_warmup=10**9,
        checkpoint_interval=900, checkpoint_slots=3, max_batches_per_epoch=BATCHES))
    return ROOT / "runs" / f"living_{D}d_seed{SEED}"

torch.manual_seed(0)
mind = LuthiMind(train(), ROOT / "life"); world = TextWorld(); loop = LifeLoop(mind, world, hz=0.0)
layers = {p: m for p, m in mind.model.named_modules() if isinstance(m, PredictiveCodingLayer)}
state = {"on": False}; caught = {p: [] for p in layers}
for p, m in layers.items():
    m.register_forward_pre_hook(lambda mod, args, p=p: caught[p].append(args[0].detach().reshape(-1, args[0].shape[-1]).clone()) if state["on"] else None)
lu = mind._luthi
proxy = types.SimpleNamespace(**{n: getattr(lu, n) for n in dir(lu) if not n.startswith("__")})
def perceive(model, **kw):
    state["on"] = True
    try: return lu.perceive(model, **kw)
    finally: state["on"] = False
proxy.perceive = perceive; mind._luthi = proxy
news, kinds = [], []
real_take = loop.line.take
def take():
    v = real_take(); news.append(v.new); return v
loop.line.take = take
A, B = "a zebra walked past the gate", "the garden is quiet this morning"
for i in range(300):
    kind = "quiet"
    if i % 3 == 0:
        kind = "A" if (i // 3) % 2 else "B"
        world.say(A if kind == "A" else B, source="brian")
    kinds.append(kind); loop.step()
print(f"view tokens per moment: {caught[next(iter(layers))][-1].shape[0]}; new tokens: spoken {statistics.median(n for n, k in zip(news, kinds) if k != 'quiet')}, quiet {statistics.median(n for n, k in zip(news, kinds) if k == 'quiet')}")
print(f"moments: {kinds.count('A')} sentence A, {kinds.count('B')} sentence B, {kinds.count('quiet')} quiet\n")

def unit(x): return x / x.norm(dim=1, keepdim=True).clamp_min(1e-30)
def cos_between(u, ia, ib):
    c = u[ia] @ u[ib].T
    if ia is ib or (len(ia) == len(ib) and bool((torch.tensor(ia) == torch.tensor(ib)).all())):
        c = c[~torch.eye(len(ia), dtype=bool)]
    return float(c.median())
def loo(u, labels):
    """Leave-one-out nearest-centroid accuracy."""
    right = 0; names = sorted(set(labels))
    for i in range(len(labels)):
        best, bc = None, -9
        for n in names:
            idx = [j for j, l in enumerate(labels) if l == n and j != i]
            c = float(u[i] @ unit(u[idx].mean(0, keepdim=True))[0])
            if c > bc: best, bc = n, c
        right += best == labels[i]
    return right / len(labels)

skip = 12   # let the view fill
for p in layers:
    X = caught[p]
    whole = torch.stack([x.mean(0) for x in X]).double()
    new = torch.stack([x[-n:].mean(0) for x, n in zip(X, news)]).double()
    change = torch.cat([torch.zeros(1, whole.shape[1], dtype=torch.double), whole[1:] - whole[:-1]])
    idx = list(range(skip, 300)); k = [kinds[i] for i in idx]
    iA = [j for j, x in enumerate(k) if x == "A"]; iB = [j for j, x in enumerate(k) if x == "B"]; iQ = [j for j, x in enumerate(k) if x == "quiet"]
    spoke = ["spoken" if x != "quiet" else "quiet" for x in k]
    print(p)
    print(f"  {'':8s} {'all pairs':>10s} {'quiet~quiet':>12s} {'A~A':>7s} {'A~B':>7s} {'spoken~quiet':>13s} {'tell spoken/quiet':>18s} {'tell A/B':>9s} {'size: spoken/quiet':>19s}")
    for name, V in (("whole", whole), ("new", new), ("change", change)):
        v = V[idx]; u = unit(v); n = v.norm(dim=1)
        allp = float((u @ u.T)[~torch.eye(len(idx), dtype=bool)].median())
        ab = [j for j in range(len(k)) if k[j] != "quiet"]
        print(f"  {name:8s} {allp:10.3f} {cos_between(u, iQ, iQ):12.3f} {cos_between(u, iA, iA):7.3f} {cos_between(u, iA, iB):7.3f} {cos_between(u, iA + iB, iQ):13.3f} {loo(u, spoke):18.0%} {loo(u[ab], [k[j] for j in ab]):9.0%} {float(n[iA + iB].median() / n[iQ].median().clamp_min(1e-30)):19.1f}")
    print()
