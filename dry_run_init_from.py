"""Dry-run check: does a width-expanded seed load into the M7 model?

Builds the M7 model with the same shape-determining args the real run uses,
validates the checkpoint config the way m5_runner's --init-from does, and
attempts a strict load. Loads no data, runs no training — it just confirms the
seed is loadable before committing GPU time. Requires LUTHI_CHECKPOINT_KEY.

Defaults match run_m7_1024d.bat. Override --n_heads etc. if the M7 config changes.

    python dry_run_init_from.py --init-from runs/m7_seed/expanded_from_m6.luthi
"""
import argparse

from luthi.tokenizer import BPETokenizer
from luthi.checkpoint import load_checkpoint
from luthi.v2.model_pc import PredictiveCodingLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", dest="init_from", required=True)
    ap.add_argument("--load_tokenizer", default="corpus_build/tokenizer_32k.json")
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--n_blocks", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=4)  # M7 decision 2026-05-30: 4 (matches M6 seed)
    ap.add_argument("--ffn_expansion", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--mu_pc_exponent", type=float, default=0.25)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.load_tokenizer)
    print(f"[tokenizer] {args.load_tokenizer}  vocab={tok.vocab_size}")

    # Only shape-determining args affect the buffer key set. PC hyperparams,
    # consolidation, sparse/iPC knobs set values or non-buffer attributes, so
    # their defaults are fine for a pure load test.
    model = PredictiveCodingLM(
        vocab_size=tok.vocab_size,
        d_model=args.d_model, n_blocks=args.n_blocks, n_heads=args.n_heads,
        ffn_expansion=args.ffn_expansion, max_seq_len=args.seq_len,
        mu_pc_enabled=True, mu_pc_exponent=args.mu_pc_exponent,
    )
    print(f"[model] d_model={args.d_model} n_heads={args.n_heads} "
          f"n_blocks={args.n_blocks} buffers={sum(b.numel() for b in model.buffers()):,}")

    print(f"[init-from] loading {args.init_from}")
    ckpt = load_checkpoint(args.init_from, trusted=True)
    cfg = ckpt.get("config", {})

    # Mirror m5_runner's --init-from config validation.
    ok = True
    for k, runv in (("d_model", args.d_model), ("n_heads", args.n_heads),
                    ("n_blocks", args.n_blocks), ("ffn_expansion", args.ffn_expansion)):
        cv = cfg.get(k)
        if cv != runv:
            print(f"[FAIL] config {k}: checkpoint={cv!r}  run={runv!r}")
            ok = False
    if not ok:
        print("[RESULT] config mismatch — the real run's --init-from will reject this "
              "seed before training. Resolve before launch.")
        return

    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except Exception as e:
        print(f"[FAIL] strict load: {type(e).__name__}: {str(e)[:400]}")
        print("[RESULT] seed is NOT loadable into the M7 model.")
        return

    print("[OK] strict load succeeded.")
    print("[RESULT] seed is compatible — safe to launch M7 with this --init-from.")


if __name__ == "__main__":
    main()
