"""Does one step on a dream improve prediction, or quiet the targets?

Opus 5.5 review of the 2026-11-11 work (Fable's brief, point 4: "are the
controls on the second trunk enough?").

``learn_from_dream`` keeps a step when raw ``l_pred`` on real kept moments does
not rise. ``l_pred`` compares the predictor's output with the trunk's own
encoding of the target block, and the step moves that encoder too. A step that
makes the targets quieter lowers ``l_pred`` without better prediction. The
project's held-out instrument divides by the target block's own variance for
that reason (``luthi/v2/eval_heldout.py``, NMSE: "Raw l_pred mechanically
favors quieter latent spaces").

Fable's probe (``redteam/fable_2026_11_11/probe_do_chosen_dreams_help.py``),
same world and same step, with each change split three ways on the held-back
moments: ``log l_pred = log NMSE + log target variance``. Two arms are added:

- ``shuffled dream``: a dream's own tokens in random order. Same words, no
  sentence. If it teaches as much, the gain is the world's vocabulary.
- the built check: does the step pass ``learn_from_dream``'s own check (mean
  per-moment ``l_pred`` on six kept moments that are not the dream's parents)?
  And would it pass the same check read as NMSE?

Run: python redteam/opus_review_2026_11_14/probe_dream_gain_against_target_scale.py <run dir>
Env: TRIALS (40), LR (0.01), SEED (0).
"""
from __future__ import annotations

import copy
import math
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import luthi.sanctuary_interface as L
from luthi.v2.plasticity import freeze_plasticity

run = sys.argv[1]
TRIALS = int(os.environ.get("TRIALS", 40))
LR = float(os.environ.get("LR", 0.01))
SEED = int(os.environ.get("SEED", 0))
loaded = L.load_m9_run(run, life_dir=tempfile.mkdtemp(prefix="opus_dream_"), device="cpu")
loaded.trainer.action_log.close()
lm = loaded.trainer.loss_module
tok = loaded.tokenizer
SEQ = int(loaded.config["seq_len"])
params = [p for p in lm.parameters() if p.requires_grad]
rng = random.Random(SEED)

# Fable's small world, unchanged.
SUBJECTS = "zebra lantern harbour kettle sparrow anvil orchard compass violin glacier".split()
DOES = "walks floats freezes sings turns glows sleeps rings drifts opens".split()
FACT = dict(zip(SUBJECTS, DOES))
TRAINED_ON = ("river stone light memory window garden quiet music signal field morning "
              "shadow pattern voice letter bridge winter colour thread").split()


def sentence() -> str:
    return " . ".join(f"the {x} {FACT[x]}" for x in rng.sample(SUBJECTS, 4)) + " ."


def tokens(text: str) -> torch.Tensor:
    ids = tok.encode(text)
    return torch.tensor([(ids * (SEQ // len(ids) + 1))[:SEQ]], dtype=torch.long)


def read(batch: torch.Tensor) -> tuple[float, float]:
    """l_pred and the target block's variance, as eval_heldout reads them."""
    lm.eval()
    with freeze_plasticity(lm.online_encoder), torch.no_grad():
        out = lm.compute_modality_loss("text", {"text_tokens": batch})
    target = out["target_latents"][:, out["ctx_len"]:, :]
    var = float((target - target.mean(dim=(0, 1), keepdim=True)).pow(2).mean())
    return float(out["l_pred"]), var


def check(moments: list[torch.Tensor]) -> tuple[float, float]:
    """learn_from_dream's check (mean per-moment l_pred), and the same read as NMSE."""
    pairs = [read(m) for m in moments]
    return (sum(p for p, _ in pairs) / len(pairs), sum(p / v for p, v in pairs) / len(pairs))


def direction(batch: torch.Tensor) -> torch.Tensor:
    lm.eval()
    for p in params:
        p.grad = None
    with freeze_plasticity(lm.online_encoder):
        lm.compute_modality_loss("text", {"text_tokens": batch})["loss"].backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params]).clone()


def step(g: torch.Tensor) -> None:
    with torch.no_grad():
        i = 0
        for p in params:
            n = p.numel()
            p.add_(g[i:i + n].view_as(p), alpha=-LR / float(g.norm().clamp_min(1e-12)))
            i += n


start = copy.deepcopy(lm.state_dict())
ARMS = ("random dream", "replay of a kept moment", "shuffled dream", "words from its training", "random tokens")
rows = {a: [] for a in ARMS}
for t in range(TRIALS):
    kept = [tokens(sentence()) for _ in range(12)]
    held = torch.cat([tokens(sentence()) for _ in range(24)])
    a, b = rng.sample(range(12), 2)
    dream = torch.cat([kept[a][:, : SEQ // 2], kept[b][:, SEQ // 2:]], dim=1)
    others = [kept[i] for i in range(12) if i not in (a, b)][:6]
    shuffled = dream[:, torch.randperm(SEQ, generator=torch.Generator().manual_seed(t))]
    lm.load_state_dict(start)
    lp0, var0 = read(held)
    c0 = check(others)
    batches = {
        "random dream": dream, "replay of a kept moment": kept[a], "shuffled dream": shuffled,
        "words from its training": tokens(" ".join(rng.choice(TRAINED_ON) for _ in range(40))),
        "random tokens": torch.randint(0, 30000, (1, SEQ), generator=torch.Generator().manual_seed(10_000 + t)),
    }
    for arm, batch in batches.items():
        lm.load_state_dict(start)
        step(direction(batch))
        lp1, var1 = read(held)
        c1 = check(others)
        rows[arm].append({
            "l_pred": math.log(lp1 / lp0), "var": math.log(var1 / var0),
            "nmse": math.log((lp1 / var1) / (lp0 / var0)),
            "passes": c1[0] <= c0[0], "passes_nmse": c1[1] <= c0[1],
        })
lm.load_state_dict(start)
print(f"{TRIALS} trials, one step of length {LR}; held-back l_pred {lp0:.4f}, target variance {var0:.4f}, NMSE {lp0 / var0:.4f} (last trial)")
print("change on held-back moments, as a percentage (log ratio x 100; mean, standard error):")
print(f"  {'one step on':26s} {'l_pred':>16s} {'target variance':>18s} {'NMSE':>16s}   passes the built check / as NMSE")
for arm, r in rows.items():
    def m(key):
        xs = [x[key] for x in r]
        return f"{100 * statistics.mean(xs):+7.3f} ({100 * statistics.stdev(xs) / len(xs) ** 0.5:.3f})"
    print(f"  {arm:26s} {m('l_pred'):>16s} {m('var'):>18s} {m('nmse'):>16s}   "
          f"{sum(x['passes'] for x in r)}/{len(r)} / {sum(x['passes_nmse'] for x in r)}/{len(r)}")
