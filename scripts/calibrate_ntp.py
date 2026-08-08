"""Dose calibration for the LLM-JEPA next-token term.

Spec: docs/reviews/2026-08-08_llm-jepa-integration-spec-for-opus.md §2
("compute on a real batch before choosing, show the arithmetic").

Measures, on a freshly-initialized model with the stage-50 config (depth 8,
muPC OFF) and a real gutenberg_100 batch:

  1. L_NTP and the JEPA-side total at init -> the loss-share the spec
     targets (NTP at 30-50% of total).
  2. GRADIENT-norm contribution of each term to the shared trunk. Loss share
     and gradient share are different quantities and can disagree badly; the
     thing that decides what training optimizes is the gradient. Reported
     side by side so the design seat can rule on which to dose against.

Read-only. CPU. No checkpoint needed -- init is the state the spec targets.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def grad_norm(loss, params) -> float:
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    tot = 0.0
    for g in gs:
        if g is not None:
            tot += float(g.detach().pow(2).sum())
    return tot ** 0.5


def main() -> None:
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
    from luthi.v2.jepa_loss import JEPALoss
    from luthi.v2.multimodal_data import TextDataset, TextDatasetConfig

    mk = dict(
        vocab_size=32000, d_model=512, n_blocks=8, n_heads=4, ffn_expansion=1,
        max_seq_len=128, backward_pass_enabled=True, consolidation_enabled=True,
        learning_gain_enabled=True, relative_trust=True,
        episode_recall_threshold=0.7,
        mu_pc_enabled=False,          # stage 50: muPC OFF
    )
    torch.manual_seed(46)
    model = MultimodalPredictiveCodingLM(**mk)
    loss_mod = JEPALoss(online_encoder=model, sigreg_lambd=0.2, w_ntp=1.0)

    ds = TextDataset(TextDatasetConfig(
        source_paths=[str(REPO / "corpus_build" / "gutenberg_100")],
        tokenizer_path=REPO / "corpus_build" / "tokenizer_32k.json",
        batch_size=32, seq_len=128, stride=64, base_seed=7,
        holdout_fraction=0.02,
    ))
    toks = ds.next_batch()["text_tokens"]

    out = loss_mod.compute_modality_loss("text", {"text_tokens": toks})
    l_ntp = float(out["l_ntp"])
    l_pred = float(out["l_pred"])
    l_sigreg = float(out["l_sigreg"])
    jepa_side = l_pred + 0.2 * l_sigreg

    print("=== magnitudes at init (d8, muPC off, real batch) ===")
    print(f"  L_NTP            {l_ntp:10.4f}   (ln(32000) = 10.373 expected)")
    print(f"  l_pred           {l_pred:10.4f}")
    print(f"  l_sigreg         {l_sigreg:10.4f}   (x0.2 = {0.2*l_sigreg:.4f})")
    print(f"  JEPA-side total  {jepa_side:10.4f}")

    print("\n=== loss-share dosing (the spec's target: NTP 30-50%) ===")
    for target in (0.30, 0.40, 0.50):
        # w*ntp / (w*ntp + J) = target  =>  w = target*J / ((1-target)*ntp)
        w = target * jepa_side / ((1 - target) * l_ntp)
        print(f"  NTP at {target:.0%} of total  ->  w_ntp = {w:8.2f}")

    # --- gradient share, the quantity that actually steers training ---
    # Measured on the SHARED trunk only (LM head excluded): that is where the
    # two objectives actually compete. Each term's graph is built separately
    # so the norms are attributable.
    trunk = [p for n, p in model.named_parameters()
             if p.requires_grad and "output_proj" not in n]

    ntp_only = loss_mod._ntp_loss(toks)
    g_ntp = grad_norm(ntp_only, trunk)

    saved = loss_mod.w_ntp
    loss_mod.w_ntp = 0.0                      # JEPA-only forward
    out_j = loss_mod.compute_modality_loss("text", {"text_tokens": toks})
    g_jepa = grad_norm(out_j["loss"], trunk)
    loss_mod.w_ntp = saved

    print("\n=== gradient-norm share on the shared trunk ===")
    print(f"  ||dL_NTP/dtheta||    {g_ntp:12.6f}")
    print(f"  ||dL_JEPA/dtheta||   {g_jepa:12.6f}")
    print(f"  ratio JEPA/NTP       {g_jepa / max(g_ntp, 1e-12):10.2f}")
    for target in (0.30, 0.50):
        w = target * g_jepa / ((1 - target) * max(g_ntp, 1e-12))
        print(f"  NTP at {target:.0%} of GRADIENT -> w_ntp = {w:8.2f}")


if __name__ == "__main__":
    main()
