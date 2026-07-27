"""Refined: per-seed servings of EXTREME Greek-density windows (>=50% Greek
pieces), the tier that plausibly triggers trust events. Anchor: seed44 @ 58650."""
import sys, re
sys.path.insert(0, r"C:\Dev\LuthiModel")
import torch
from luthi.tokenizer import BPETokenizer

CACHE = r"C:\Dev\LuthiModel\corpus_build\cache\b623e9aae3db1895040c6f1077b0bad5126745e84329fe4c4f95993ce8fef8f5.pt"
SEQ_LEN, STRIDE, BS, HOLDOUT = 128, 64, 32, 0.02
TOTAL_STEPS = 72042
GOLD = 0x9E3779B97F4A7C15

tokens = torch.load(CACHE, weights_only=True)
n = tokens.numel()
tok = BPETokenizer.load(r"C:\Dev\LuthiModel\corpus_build\tokenizer_32k.json")
greek_re = re.compile("[\u0370-\u03ff\u1f00-\u1fff]")
gids = [i for i, b in tok.vocab.items() if greek_re.search(b.decode("utf-8", errors="ignore"))]
mask = torch.zeros(tok.vocab_size, dtype=torch.bool)
mask[torch.tensor(gids)] = True
stream = mask[tokens]
cs = torch.cat([torch.zeros(1, dtype=torch.int32), stream.to(torch.int32).cumsum(0)])

holdout_tokens = int(n * HOLDOUT)
train_max_start = (n - holdout_tokens - SEQ_LEN) - SEQ_LEN
n_seq = (train_max_start // STRIDE) + 1

# density of the ACTUAL served window at each valid start (starts are k*stride)
k = torch.arange(n_seq)
starts_all = k * STRIDE
wdens = cs[starts_all + SEQ_LEN] - cs[starts_all]     # per servable sequence

for tier in (64, 80):
    hot_seq = (wdens >= tier).nonzero().flatten()      # sequence indices (start/stride)
    print(f"\n=== tier >= {tier}/128 Greek: {len(hot_seq)} servable sequences ===")
    lo = int(starts_all[hot_seq].min()); hi = int(starts_all[hot_seq].max()) + SEQ_LEN
    print(f"token span of tier: [{lo:,} .. {hi:,}]")
    hotset = torch.zeros(n_seq, dtype=torch.bool); hotset[hot_seq] = True
    for seed in (42, 43, 44, 45, 46):
        steps = []
        for e in range(4):
            se = seed ^ ((e * GOLD) & 0xFFFFFFFFFFFFFFFF)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(se & 0x7FFFFFFFFFFFFFFF))
            perm = torch.randperm(n_seq, generator=gen)
            m = hotset[perm]
            for p in m.nonzero().flatten().tolist():
                step = (e * n_seq + p) // BS + 1
                if step <= TOTAL_STEPS:
                    steps.append(step)
        steps.sort()
        print(f"  seed {seed}: {steps}")
        if seed == 44 and 58650 in steps:
            print(f"    VALIDATION: 58650 present at this tier")
