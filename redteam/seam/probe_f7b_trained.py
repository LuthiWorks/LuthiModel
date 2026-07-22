"""F7b — encoder-half probe against the REAL trained M8 checkpoint (#2).

Run:  python -m redteam.seam.probe_f7b_trained

The earlier `probe_seam_review.py::probe_f7b` ran on a tiny *untrained* model
and could only speak to an untrained encoder/V-head. This runs the load-bearing
half of F7b on the actual trained multimodal-PC encoder produced by #1
(E:/runs/m8_multimodal_smoke/model_final.luthi):

  Does the TRAINED encoder map the two real seam input regimes to
  distributionally different pooled s_t?
    - training s_t  = compute_s_t( encode(context-fraction slice, causal=False) )
                      (mirrors runner.py:469  compute_s_t(raw["online_context_latents"]))
    - actor s_t     = compute_s_t( encode(full sequence, causal=False) )
                      (mirrors sanctuary_interface.encode_state -> compute_s_t)

Both go through the REAL compute_s_t helper and the REAL model.encode, on REAL
in-distribution text (random token ids would be OOD for the encoder and a
confound). The V-head-amplification half is deferred (needs trained M9 heads).

Secondary: the causal axis — does the generation-path captured encode
(model(x, return_encode_result=True), the `generate_text(return_state=True)`
path) agree with encode_state's causal=False encode? If they disagree, the
seam has two actor s_t definitions and that's its own correctness issue.

Convention: confirmed=True means the concern is REAL (a material gap exists).
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import torch

from luthi.generate import load_model_from_checkpoint
from luthi.v2.m9.s_t import compute_s_t

from redteam.m9_step1._common import Verdict, report

CKPT = "E:/runs/m8_multimodal_smoke/model_final.luthi"
CONTEXT_FRACTION = 0.8  # JEPALoss default; the training-time context slice.
WINDOW = 64             # tokens per sample window (<= seq_len)
STRIDE = 24
MAX_WINDOWS = 80
REL_GAP_FLAG = 0.10     # input-space rel gap above this = concern is real

# Fallback in-distribution-ish English if no corpus file is reachable. Real
# prose (not random ids) so the trained encoder operates in-distribution.
_FALLBACK = (
    "The morning came slowly over the hills, and the light moved across the "
    "valley the way water finds its level, without hurry and without doubt. "
    "She had waited a long time for news, and when it came it was smaller than "
    "she had feared and larger than she had hoped, which is the usual shape of "
    "news. The old man set down his cup and looked at the rain against the "
    "glass. There are questions that answer themselves if you are patient, he "
    "said, and questions that only grow louder the longer you stare at them. "
    "The child ran ahead along the path, stopping to gather stones, naming each "
    "one after a person she loved, then leaving them in a careful row by the "
    "gate. Far off a bell rang twice and was silent. The work of the day began "
    "as it always did, with small things done in order, and the larger meaning "
    "left to gather on its own, the way moss gathers, without being asked."
) * 6


def _load_text() -> str:
    for cand in (
        "corpus_build/gutenberg_100",
        "corpus_build/gutenberg_sample",
    ):
        p = Path(cand)
        if p.is_dir():
            chunks = []
            for f in sorted(p.glob("*"))[:6]:
                try:
                    chunks.append(f.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
            text = "\n".join(chunks).strip()
            if len(text) > 2000:
                return text
    return _FALLBACK


def _windows(token_ids, n_max):
    out = []
    i = 0
    while len(out) < n_max and i + WINDOW <= len(token_ids):
        out.append(token_ids[i:i + WINDOW])
        i += STRIDE
    return out


def _encode_pool(model, tokens, *, causal):
    """compute_s_t over a single text window's encode. tokens: 1D list."""
    x = torch.tensor([tokens], dtype=torch.long)
    res = model.encode(
        text_tokens=x,
        audio_waveform=None, audio_tokens=None,
        image=None, vision_tokens=None,
        causal=causal,
    )
    return compute_s_t(res["latents"])  # [1, D]


def main() -> int:
    verdicts: list[Verdict] = []
    src = "?"
    try:
        if not os.environ.get("LUTHI_CHECKPOINT_KEY"):
            print("LUTHI_CHECKPOINT_KEY not set — cannot decrypt the checkpoint.")
            return 0
        model, tok, cfg, ep = load_model_from_checkpoint(CKPT)
        model.eval()

        text = _load_text()
        src = "corpus" if text is not _FALLBACK else "fallback-passage"
        ids = tok.encode(text)
        wins = _windows(ids, MAX_WINDOWS)
        if len(wins) < 8:
            print(f"Too few windows ({len(wins)}); aborting.")
            return 0

        train_pool, actor_pool, paired = [], [], []
        causal_gap = []
        with torch.no_grad():
            for w in wins:
                ctx_len = max(1, int(round(len(w) * CONTEXT_FRACTION)))
                s_train = _encode_pool(model, w[:ctx_len], causal=False)
                s_actor = _encode_pool(model, w, causal=False)
                train_pool.append(s_train)
                actor_pool.append(s_actor)
                paired.append(float((s_train - s_actor).norm().item()))
                # secondary: causal axis on the full window
                try:
                    s_actor_causal = _encode_pool(model, w, causal=True)
                    causal_gap.append(
                        float((s_actor - s_actor_causal).norm().item())
                    )
                except Exception:
                    pass

        train = torch.cat(train_pool, 0)
        actor = torch.cat(actor_pool, 0)
        train_norm = float(train.norm(dim=1).mean().item())
        actor_norm = float(actor.norm(dim=1).mean().item())
        mean_pair = sum(paired) / len(paired)
        rel_gap = mean_pair / (0.5 * (train_norm + actor_norm) + 1e-9)
        mean_shift = float((train.mean(0) - actor.mean(0)).norm().item())

        confirmed = bool(rel_gap > REL_GAP_FLAG)
        detail = (
            f"[trained encoder | text={src} | n={len(wins)} windows, "
            f"W={WINDOW}, ctx_frac={CONTEXT_FRACTION}] "
            f"paired ||s_train(context) - s_actor(full)|| mean = {mean_pair:.4f} "
            f"(|s_train|={train_norm:.4f}, |s_actor|={actor_norm:.4f}, "
            f"rel gap = {rel_gap:.3f}); per-dim mean shift = {mean_shift:.4f}. "
            + (
                "TRAINED encoder maps context-slice vs full to materially "
                "different s_t -> the heads would see an inference-time "
                "distribution shift; the separate-full-encode training-side fix "
                "is warranted (verify magnitude at larger scale)."
                if confirmed else
                "context-slice vs full s_t are close on the trained encoder -> "
                "the inference distribution shift is small (re-verify at larger "
                "scale before calling F7b closed)."
            )
        )
        verdicts.append(Verdict(
            "F7b (encoder-half): actor s_t (full) is OOD vs training s_t "
            "(context-fraction) on the TRAINED encoder",
            confirmed, detail,
        ))

        if causal_gap:
            cmean = sum(causal_gap) / len(causal_gap)
            crel = cmean / (actor_norm + 1e-9)
            cconf = bool(crel > REL_GAP_FLAG)
            verdicts.append(Verdict(
                "F7b (causal axis): generation-path causal encode disagrees "
                "with encode_state's causal=False encode",
                cconf,
                f"full-window s_t, causal=False vs causal=True: mean "
                f"||delta|| = {cmean:.4f}, rel = {crel:.3f}. "
                + (
                    "The two actor s_t definitions (encode_state vs the "
                    "generation-forward capture) diverge — the seam must pin "
                    "ONE causal convention or the V-head sees a third "
                    "distribution."
                    if cconf else
                    "The two actor encode paths agree closely on this encoder."
                ),
            ))
    except Exception:
        verdicts.append(Verdict(
            "F7b-trained probe raised", False,
            "ERROR:\n" + traceback.format_exc(),
        ))

    report("F7b ON THE TRAINED M8 CHECKPOINT (#2, encoder-half)", verdicts)
    n_conf = sum(1 for v in verdicts if v.confirmed)
    print("\n" + "=" * 62)
    print(f"F7b-trained: {n_conf}/{len(verdicts)} concerns CONFIRMED")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
