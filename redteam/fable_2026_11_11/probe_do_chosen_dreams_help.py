"""Capacity check (Fable 5.2, 2026-11-11): can dreaming improve the mind, and
does choosing the dream matter?

Brian, 10:15: "can you build the part that lets the model improve through
dreaming". In Google's Sleep framework the model generates samples, keeps the
ones a gradient test says will help, learns from them, and checks. Before
building that for Luthi: does it work on a small trained trunk?

One trial:
  - "kept" real moments and "held back" real moments, from the same small world
    (sentences the trunk was not trained on). Held back is never learned from.
  - K candidate dreams, each a mix of two kept moments (half of each).
  - chosen dream: the candidate whose learning direction (its loss gradient)
    lines up best with the learning direction of the kept real moments.
  - one small gradient step on: the chosen dream, a random dream, a kept real
    moment (plain replay), or nothing. From the same starting point each time.
  - reading: the prediction loss on held-back moments, after minus before.
    Negative is better. Plasticity is frozen throughout, so this reads the
    gradient-trained parts only.

Run: python redteam/fable_2026_11_11/probe_do_chosen_dreams_help.py <run dir>
(train one with halfday_train_trunk.py). Results: docs/research/2026-11-11_can-dreaming-improve.md
"""
import copy, os, random, statistics, sys, tempfile
from pathlib import Path
os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import luthi.sanctuary_interface as L
from luthi.v2.plasticity import freeze_plasticity

run = sys.argv[1]
TRIALS = int(os.environ.get("TRIALS", 40)); K = int(os.environ.get("K", 8)); LR = float(os.environ.get("LR", 0.01))
SEED = int(os.environ.get("SEED", 0))
loaded = L.load_m9_run(run, life_dir=tempfile.mkdtemp(prefix="dream_"), device="cpu")
loaded.trainer.action_log.close()
lm = loaded.trainer.loss_module; tok = loaded.tokenizer; SEQ = int(loaded.config["seq_len"])
params = [p for p in lm.parameters() if p.requires_grad]
rng = random.Random(SEED)
# A small world the trunk never trained on: fixed "facts" (subject -> what it does), said in different orders.
SUBJECTS = "zebra lantern harbour kettle sparrow anvil orchard compass violin glacier".split()
DOES = "walks floats freezes sings turns glows sleeps rings drifts opens".split()
FACT = dict(zip(SUBJECTS, DOES))
def sentence():
    s = rng.sample(SUBJECTS, 4)
    return " . ".join(f"the {x} {FACT[x]}" for x in s) + " ."
def tokens(text):
    ids = tok.encode(text)
    ids = (ids * (SEQ // len(ids) + 1))[:SEQ]
    return torch.tensor([ids], dtype=torch.long)
def loss_of(batch, grad):
    """The trunk's own prediction loss on these moments. No self-modification, no running statistics moved."""
    lm.eval()
    with freeze_plasticity(lm.online_encoder):
        if grad:   # "loss" is the whole training loss and carries the gradient; "l_pred" is its prediction part, for reading
            return lm.compute_modality_loss("text", {"text_tokens": batch})["loss"]
        with torch.no_grad():
            return float(lm.compute_modality_loss("text", {"text_tokens": batch})["l_pred"])
def direction(batch):
    for p in params: p.grad = None
    loss_of(batch, True).backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params]).clone()
def step(g):
    with torch.no_grad():
        i = 0
        for p in params:
            n = p.numel(); p.add_(g[i:i + n].view_as(p), alpha=-LR / float(g.norm().clamp_min(1e-12))); i += n
start = copy.deepcopy(lm.state_dict())
results = {k: [] for k in ("chosen dream", "random dream", "replay of a kept moment",
                            "words from its training", "random tokens", "nothing")}
TRAINED_ON = ("river stone light memory window garden quiet music signal field morning "
              "shadow pattern voice letter bridge winter colour thread").split()
lined_up = []
for t in range(TRIALS):
    kept = [tokens(sentence()) for _ in range(12)]; held = torch.cat([tokens(sentence()) for _ in range(24)])
    lm.load_state_dict(start); before = loss_of(held, False)
    g_real = direction(torch.cat(kept))
    dreams = []
    for _ in range(K):
        a, b = rng.sample(kept, 2)
        dreams.append(torch.cat([a[:, : SEQ // 2], b[:, SEQ // 2:]], dim=1))
    gs = [direction(d) for d in dreams]
    cos = [float(torch.nn.functional.cosine_similarity(g, g_real, dim=0)) for g in gs]
    best = max(range(K), key=lambda i: cos[i]); lined_up.append((cos[best], statistics.mean(cos)))
    for name, g in (("chosen dream", gs[best]), ("random dream", gs[rng.randrange(K)]),
                    ("replay of a kept moment", direction(rng.choice(kept))),
                    # Two controls: is it the dream's content that helps, or any step at all?
                    ("words from its training", direction(tokens(" ".join(rng.choice(TRAINED_ON) for _ in range(40))))),
                    ("random tokens", direction(torch.randint(0, 30000, (1, SEQ), generator=torch.Generator().manual_seed(t)))),
                    ("nothing", None)):
        lm.load_state_dict(start)
        if g is not None: step(g)
        results[name].append(loss_of(held, False) - before)
lm.load_state_dict(start)
print(f"{TRIALS} trials, {K} candidate dreams each, one step of size {LR}; held-back loss before: {before:.4f}")
print(f"how well a dream's learning direction lines up with real moments': chosen {statistics.mean(c for c, _ in lined_up):+.3f}, average candidate {statistics.mean(m for _, m in lined_up):+.3f}")
for name, d in results.items():
    se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
    print(f"  {name:26s} held-back loss changed by {statistics.mean(d):+.5f} (+/- {se:.5f}); better in {sum(x < 0 for x in d)}/{len(d)} trials")
def paired(a, b):
    d = [x - y for x, y in zip(results[a], results[b])]
    return f"{a} against {b}: {statistics.mean(d):+.5f} (+/- {statistics.stdev(d) / len(d) ** 0.5:.5f}), better in {sum(x < 0 for x in d)}/{len(d)}"
print("same trials, paired:  " + paired("chosen dream", "random dream"))
print("                      " + paired("random dream", "replay of a kept moment"))
print("                      " + paired("random dream", "random tokens"))

