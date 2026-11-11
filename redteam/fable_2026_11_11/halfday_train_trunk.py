"""Train the small trunk once and leave it at a fixed place for the half-day probes."""
import os, random, sys
from argparse import Namespace
from pathlib import Path
os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
ROOT = Path(sys.argv[1]); ROOT.mkdir(parents=True, exist_ok=True)
os.environ["LUTHI_CORPUS_CACHE_DIR"] = str(ROOT / "cache")
LUTHI = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(LUTHI))
import scripts.jepa_pilot_driver as driver
D = int(os.environ.get("D", "32")); SEED = int(os.environ.get("SEED", "42")); BATCHES = int(os.environ.get("BATCHES", "300"))
corpus = ROOT / "corpus"; corpus.mkdir(exist_ok=True)
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
print("RUN", ROOT / "runs" / f"living_{D}d_seed{SEED}")
