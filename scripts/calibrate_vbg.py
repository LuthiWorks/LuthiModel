"""Offline dose calibration for the variance-budget governor (VBG).

Spec: docs/reviews/2026-08-07_variance-budget-governor-spec-for-opus.md §2
("size both weights against measured magnitudes, not paper defaults ...
show the arithmetic in a comment").

Answers three questions the spec's doses depend on, all read-only on CPU
against an existing checkpoint:

  1. raw vs trace-normalized sketched penalty on the SAME latents -> the
     conversion factor for w_share, since alpha=10 on the raw scale is the
     only dose ever measured to arrest the depth-8 collapse.
  2. top-direction share in SKETCH space (what the governor computes) vs
     FULL space (what soloist_forensic.py reports and what the spec's
     cap=0.05 was read off). These are not the same number.
  3. the cap term's magnitude at the measured share, for w_cap.

Usage:
    python scripts/calibrate_vbg.py probe_d8_wsig10_512d_seed46
    python scripts/calibrate_vbg.py probe_v5_d8_dk5000_512d_seed46
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GOVERNED_BLOCKS = (0, 3, 6)
SKETCH_K = 64


def load_model(run_name: str):
    run = REPO / "runs" / "jepa_pilot" / run_name
    pr = json.loads((run / "pilot_result.json").read_text())
    mk = dict(pr["config"].get("model_kwargs") or {})
    if not mk:
        mk = dict(vocab_size=32000, d_model=512,
                  n_blocks=pr["config"]["n_blocks"], n_heads=4,
                  ffn_expansion=1, max_seq_len=128,
                  backward_pass_enabled=True, consolidation_enabled=True,
                  learning_gain_enabled=True, relative_trust=True,
                  episode_recall_threshold=0.7)
        mk.update(mu_pc_enabled=True, mu_pc_exponent=0.25)
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    cks = sorted((run / "checkpoints").glob("ckpt_*.pt"))
    if not cks:
        raise SystemExit(f"no checkpoints in {run}")
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    model = MultimodalPredictiveCodingLM(**mk)
    # strict=False, same reason as soloist_forensic.py: older checkpoints
    # predate diagnostic buffers today's model registers.
    model.load_state_dict(ck["online_state_dict"], strict=False)
    model.eval()
    return model, mk, cks[-1].name


def batch_tokens(mk: dict) -> torch.Tensor:
    from luthi.v2.multimodal_data import TextDataset, TextDatasetConfig
    ds = TextDataset(TextDatasetConfig(
        source_paths=[str(REPO / "corpus_build" / "gutenberg_100")],
        tokenizer_path=REPO / "corpus_build" / "tokenizer_32k.json",
        batch_size=32, seq_len=mk.get("max_seq_len", 128), stride=64,
        base_seed=7, holdout_fraction=0.02,
    ))
    return ds.next_batch()["text_tokens"]


def full_space_top_share(z: torch.Tensor) -> float:
    flat = z.reshape(-1, z.shape[-1]).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat.t() @ flat) / max(flat.shape[0] - 1, 1)
    sv = torch.linalg.svdvals(cov).clamp(min=1e-12)
    return float((sv.max() / sv.sum()).item())


def main() -> None:
    from luthi.v2.jepa_loss import (
        sketched_isotropy_penalty, top_direction_share, soloist_cap_penalty,
    )

    run_name = sys.argv[1] if len(sys.argv) > 1 else "probe_d8_wsig10_512d_seed46"
    model, mk, ck_name = load_model(run_name)
    print(f"=== {run_name}  ({ck_name}) ===")

    g = torch.Generator().manual_seed(20260807)
    sketch = torch.randn(model.d_model, SKETCH_K, generator=g) / (model.d_model ** 0.5)

    toks = batch_tokens(mk)
    with torch.no_grad():
        out = model.encode(text_tokens=toks, causal=False, collect_block_latents=True)
    blocks = out.get("block_latents") or []
    print(f"blocks collected: {len(blocks)}")

    gp = torch.Generator().manual_seed(20260808)
    v0 = torch.randn(SKETCH_K, generator=gp)
    v0 = v0 / v0.norm()

    raws, norms, sk_shares, fu_shares = [], [], [], []
    for bi in GOVERNED_BLOCKS:
        if bi >= len(blocks):
            continue
        z = blocks[bi].float()
        raw = sketched_isotropy_penalty(z, sketch, trace_normalized=False)
        nrm = sketched_isotropy_penalty(z, sketch, trace_normalized=True)
        share_sk, _ = top_direction_share(z, sketch, v0.clone(), n_iter=25)
        share_fu = full_space_top_share(z)
        raws.append(float(raw)); norms.append(float(nrm))
        sk_shares.append(float(share_sk)); fu_shares.append(share_fu)
        print(f"  block {bi}:  raw={float(raw):8.3f}  trace_norm={float(nrm):8.3f}"
              f"   share_sketch={float(share_sk):.4f}  share_full={share_fu:.4f}")

    if not raws:
        raise SystemExit("no governed blocks collected")

    raw_m = sum(raws) / len(raws)
    nrm_m = sum(norms) / len(norms)
    sk_m = sum(sk_shares) / len(sk_shares)
    fu_m = sum(fu_shares) / len(fu_shares)

    print("\n--- means ---")
    print(f"  raw penalty          {raw_m:.4f}")
    print(f"  trace-normalized     {nrm_m:.4f}")
    print(f"  ratio raw/norm       {raw_m / max(nrm_m, 1e-12):.3f}")
    print(f"  share (sketch K=64)  {sk_m:.4f}")
    print(f"  share (full D)       {fu_m:.4f}")
    print(f"  sketch/full inflation {sk_m / max(fu_m, 1e-12):.2f}x")

    # --- w_share: match the alpha=10 raw-scale contribution ---
    # Measured on probe_d8_wsig10_512d_seed46 by the loss identity
    # (l_wsig is not logged; see the runner deviation note):
    #   step 100: loss 451.11 = l_pred 3.738 + 0.2*1887.56 + 10*l_wsig
    #             => 10*l_wsig = 69.86, l_wsig = 6.986  (15.5% of loss)
    # The arrest happened in the first few hundred steps, so the early
    # contribution is the dose that mattered.
    ARREST_CONTRIB = 69.86
    w_share = ARREST_CONTRIB / max(nrm_m, 1e-12)
    print(f"\n  w_share to match arrest contribution {ARREST_CONTRIB}: "
          f"{w_share:.2f}")

    # --- w_cap: O(10) contribution at the floor share ---
    for probe_share in (0.05, 0.08, 0.12, sk_m):
        pen = float(soloist_cap_penalty(torch.tensor(probe_share), 0.05))
        w = 10.0 / pen if pen > 0 else float("inf")
        print(f"  at share {probe_share:.4f}: cap_penalty={pen:.6f}"
              f"   w_cap for O(10) = {w:,.0f}")


if __name__ == "__main__":
    main()
