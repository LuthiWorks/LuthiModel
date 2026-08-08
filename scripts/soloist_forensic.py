"""Identify the persistent dominant-variance direction (the 'soloist') in
depth-8 trunks.

Brian's 2026-08-07 observation: stable_rank stays ~1-5 through every
depth-8 run including both recoveries, while healthy d4 sits at 31-47.
The dominant direction is NOT the batch-mean offset (stable_rank is
computed on centered covariance). This script names it: load a
checkpoint, encode a real batch, take the top principal direction of the
final-block latents, and test what it tracks — position, token identity,
token log frequency, or the positional-embedding subspace.

Read-only. CPU. Record: docs/research/2026-08-07_depth-remedy-probes-
hypothesis.md (SOLOIST section).
"""
import json
import statistics
import sys
from collections import Counter
from math import log
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def analyze(run_name: str) -> None:
    run = REPO / "runs" / "jepa_pilot" / run_name
    pr = json.loads((run / "pilot_result.json").read_text())
    mk = dict(pr["config"].get("model_kwargs") or {})
    if not mk:
        # pre-2026-08-05 runs lack persisted kwargs; reconstruct minimally
        mk = dict(vocab_size=32000, d_model=512,
                  n_blocks=pr["config"]["n_blocks"], n_heads=4,
                  ffn_expansion=1, max_seq_len=128,
                  backward_pass_enabled=True, consolidation_enabled=True,
                  learning_gain_enabled=True, relative_trust=True,
                  episode_recall_threshold=0.7, n_episodes=None)
        mk = {k: v for k, v in mk.items() if v is not None}
        mk.update(mu_pc_enabled=True, mu_pc_exponent=0.25)
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    from luthi.v2.multimodal_data import TextDataset, TextDatasetConfig

    cks = sorted((run / "checkpoints").glob("ckpt_*.pt"))
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    model = MultimodalPredictiveCodingLM(**mk)
    # strict=False: pre-2026-07-27 checkpoints predate the episode/drive
    # diagnostic buffers today's model registers; those buffers are
    # read-only telemetry and their init values are correct for replay.
    model.load_state_dict(ck["online_state_dict"], strict=False)
    model.eval()

    ds = TextDataset(TextDatasetConfig(
        source_paths=[str(REPO / "corpus_build" / "gutenberg_100")],
        tokenizer_path=REPO / "corpus_build" / "tokenizer_32k.json",
        batch_size=32, seq_len=mk.get("max_seq_len", 128), stride=64,
        base_seed=7, holdout_fraction=0.02,
    ))
    batch = ds.next_batch()
    tokens = batch["text_tokens"]

    with torch.no_grad():
        out = model.encode(text_tokens=tokens, collect_block_latents=True)
    z = out["block_latents"][-1].float()
    B, S, D = z.shape
    flat = z.reshape(-1, D)
    flat = flat - flat.mean(0, keepdim=True)
    _, Sv, Vh = torch.linalg.svd(flat, full_matrices=False)
    var = Sv.pow(2) / (flat.shape[0] - 1)
    share = var / var.sum()
    u1 = Vh[0]
    proj = flat @ u1

    print(f"== {run_name}  (final block, {B}x{S}x{D})")
    print(f"   top-dir variance share: {float(share[0]):.3f} (2nd {float(share[1]):.3f}, 3rd {float(share[2]):.3f}); latent stable rank {float(var.sum()/var.max()):.2f}")

    pp = proj.reshape(B, S)
    pos_mean = pp.mean(0)
    pos_r = torch.corrcoef(torch.stack([pos_mean, torch.arange(S, dtype=torch.float32)]))[0, 1]
    print(f"   position: linear r={float(pos_r):+.3f}; positional-profile variance / soloist variance = {float(pos_mean.var()/pp.var()):.3f}")

    tok = tokens.reshape(-1)
    by_tok: dict[int, list] = {}
    for t, v in zip(tok.tolist(), proj.tolist()):
        by_tok.setdefault(t, []).append(v)
    groups = {t: vs for t, vs in by_tok.items() if len(vs) >= 5}
    if groups:
        within = sum(statistics.pvariance(vs) * len(vs) for vs in groups.values())
        within /= sum(len(vs) for vs in groups.values())
        overall = statistics.pvariance([v for vs in groups.values() for v in vs])
        print(f"   token identity: within-token/overall variance = {within/overall:.3f} (0 = soloist IS token identity) [{len(groups)} tokens n>=5]")
        freq = Counter(tok.tolist())
        ts = list(groups)
        xs = torch.tensor([log(freq[t]) for t in ts])
        ys = torch.tensor([sum(by_tok[t]) / len(by_tok[t]) for t in ts])
        r = torch.corrcoef(torch.stack([xs, ys]))[0, 1]
        print(f"   token log-frequency vs mean coordinate: r={float(r):+.3f}")

    P = model.pos_embedding.weight.detach().float()
    Q, _ = torch.linalg.qr(P.t())
    print(f"   fraction of soloist inside positional-embedding subspace: {float((Q.t() @ u1).pow(2).sum()):.3f}")
    E = model.embedding.weight.detach().float()
    _, _, EV = torch.linalg.svd(E - E.mean(0, keepdim=True), full_matrices=False)
    print(f"   fraction inside token-embedding top-128 subspace: {float((EV[:128] @ u1).pow(2).sum()):.3f}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        analyze(name)
