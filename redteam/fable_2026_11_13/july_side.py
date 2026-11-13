"""Runs under the code of 2026-07-26 (commit 0230cc9f). Trains a small run
with that day's driver, then reloads it the way that day's verdict script did
and records what a few forwards compute."""
import argparse, json, sys, types
from pathlib import Path
import torch

out_root, arm, filelist = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
torch.set_num_threads(2)
import scripts.jepa_pilot_driver as drv
import luthi.v2.pc_ops as pc_ops
pc_ops.is_cpp_available = lambda: True      # the driver's guard; the reference maths runs
drv.OUTPUT_ROOT = out_root
drv.ARM_FILELIST[arm] = filelist
drv._device = lambda: torch.device("cpu")
args = types.SimpleNamespace(
    data_dir="unused", tokenizer=str(drv.REPO_ROOT / "corpus_build" / "tokenizer_32k.json"),
    n_blocks=2, epochs=1, batch_size=8, seq_len=32, stride=16, lr=3e-4,
    holdout_fraction=0.1, heldout_batches=2, probe_batches=2, kill_warmup=10**6,
    max_batches_per_epoch=60, seeds=None, n_seeds=1,
)
rp = drv._result_path(arm, 64, 42)
result = json.loads(rp.read_text()) if rp.exists() else drv._run_one(arm, 64, 42, args)
print("outcome", result["outcome"], "config keys", sorted(result["config"]))

# Reload as scripts/pilot_verdict.py did that day.
from luthi.living_extra_state import apply_living_extra_state
from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.multimodal_data import TextDataset, TextDatasetConfig
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
run_dir = out_root / drv._run_name(arm, 64, 42)
cfg = result["config"]
src = [l.strip() for l in Path(cfg["file_list"]).read_text().splitlines() if l.strip()]
ds = TextDataset(TextDatasetConfig(source_paths=src, tokenizer_path=Path(args.tokenizer),
    batch_size=cfg["batch_size"], seq_len=cfg["seq_len"], stride=cfg["stride"],
    base_seed=42, holdout_fraction=cfg["holdout_fraction"]))
kw = dict(vocab_size=ds.vocab_size(), d_model=64, n_blocks=cfg["n_blocks"], n_heads=4,
          ffn_expansion=1, max_seq_len=cfg["seq_len"], backward_pass_enabled=False)
kw.update(drv.ARM_CONFIGS[arm])
model = MultimodalPredictiveCodingLM(**kw)
loss = JEPALoss(online_encoder=model, sigreg_lambd=cfg.get("sigreg_lambd", 0.1))
ck = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-1]
state = torch.load(ck, map_location="cpu", weights_only=False)
model.load_state_dict(state["online_state_dict"])
apply_living_extra_state(model, state.get("living_extra_state"), source="july")
loss.predictor.load_state_dict(state["predictor_state_dict"])
loss.projection_heads.load_state_dict(state["projection_heads_state_dict"])

batches = [b["text_tokens"].clone() for b in ds.holdout_batches(8)][:6]
windows = torch.cat(batches)
assert len(batches) == 6
rec = {"batches": batches, "eval": [], "train": []}
loss.eval()
with torch.no_grad():
    for i, b in enumerate(batches[:3]):
        torch.manual_seed(1000 + i)
        r = loss.compute_modality_loss("text", {"text_tokens": b})
        rec["eval"].append({k: torch.as_tensor(v).detach().clone() for k, v in r.items() if isinstance(v, (torch.Tensor, float, int))})
rec["after_eval"] = {k: v.clone() for k, v in model.state_dict().items()}
loss.train()
with torch.no_grad():
    for i, b in enumerate(batches[3:]):
        torch.manual_seed(2000 + i)
        r = loss.compute_modality_loss("text", {"text_tokens": b})
        rec["train"].append({k: torch.as_tensor(v).detach().clone() for k, v in r.items() if isinstance(v, (torch.Tensor, float, int))})
rec["after_train"] = {k: v.clone() for k, v in model.state_dict().items()}
rec["keys"] = sorted(state["online_state_dict"])
torch.save(rec, run_dir / "july_reference.pt")
print("checkpoint", ck.name, "online keys", len(rec["keys"]), "windows", tuple(windows.shape))
print("keys", sorted(rec["eval"][0]))
