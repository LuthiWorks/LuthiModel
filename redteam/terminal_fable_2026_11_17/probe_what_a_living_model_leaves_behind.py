"""The smallest case of a test that fails after a living model was built.

``tests/test_m9_training_path.py::test_m9_rides_along_without_touching_the_trunk``
trains the same tiny run twice in one process, without M9 and with it, and
asserts the trunks are equal. It passes alone. It fails when any test that
builds a living model ran before it in the same process (2026-11-17).

This trains the two runs as that test's fixture does, in a fresh process,
optionally after doing ``--first`` something:

    nothing        (the test alone)
    build          build a small living model and drop it
    build-dead     build the same model with a dead FFN
    forward        build a living model and run one forward
    seed           only ``torch.manual_seed(0)``
    draw           draw one number from torch's global stream

and prints a fingerprint of each run's trunk, so it can be seen which of the
two runs moved, and what moves it.

    LUTHI_ALLOW_CPU=1 python redteam/terminal_fable_2026_11_17/probe_what_a_living_model_leaves_behind.py --first build
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

WORDS = "river stone light memory window garden quiet music signal".split()


def fingerprint(state: dict, skip: str = "output_proj.") -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        if not k.startswith(skip):
            h.update(k.encode())
            h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first", default="nothing",
                    choices=("nothing", "build", "build-dead", "forward", "seed", "draw"))
    args = ap.parse_args()

    if args.first in ("build", "build-dead", "forward"):
        from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
        torch.manual_seed(0)
        m = MultimodalPredictiveCodingLM(vocab_size=64, d_model=32, n_blocks=2, n_heads=4,
                                         max_seq_len=16, dead_ffn=args.first == "build-dead")
        if args.first == "forward":
            m.encode(text_tokens=torch.randint(0, 64, (1, 16)), causal=False)
        del m
    elif args.first == "seed":
        torch.manual_seed(0)
    elif args.first == "draw":
        torch.rand(1)

    import scripts.jepa_pilot_driver as driver

    root = Path(tempfile.mkdtemp(prefix="left_behind_"))
    os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(root / "cache")
    corpus = root / "corpus"
    corpus.mkdir()
    rng = random.Random(0)
    (corpus / "text.txt").write_text(" ".join(rng.choice(WORDS) for _ in range(12000)),
                                     encoding="utf-8")
    prints = {}
    for name, switches in (("no_m9", {"no_m9": True}), ("m9", {})):
        driver.OUTPUT_ROOT = root / name
        driver._run_one("living", 32, 42, Namespace(
            data_dir=str(corpus), tokenizer=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"),
            n_blocks=1, epochs=1, batch_size=4, seq_len=32, stride=32, lr=3e-4,
            holdout_fraction=0.2, heldout_batches=2, probe_batches=2,
            kill_warmup=10**9, checkpoint_interval=900, checkpoint_slots=3,
            max_batches_per_epoch=6, **switches))
        run_dir = root / name / driver._run_name("living", 32, 42)
        [path] = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-1:]
        state = torch.load(path, map_location="cpu", weights_only=False)
        prints[name] = fingerprint(state["online_state_dict"])
    print(f"RESULT first={args.first:<10} no_m9 {prints['no_m9']}  m9 {prints['m9']}  "
          f"{'equal' if prints['no_m9'] == prints['m9'] else 'DIFFERENT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
