"""Runs under today's code. Rebuilds the July-made run and repeats the
forwards the July code recorded."""
import sys, torch
from pathlib import Path
from luthi.v2.run_rebuild import rebuild_run
torch.set_num_threads(2)
run_dir = Path(sys.argv[1])
ref = torch.load(run_dir / "july_reference.pt", weights_only=False)
r = rebuild_run(run_dir, device=torch.device("cpu"))
cfg = r.result["config"]
print("marks:", {k: (v if not isinstance(v, list) else len(v)) for k, v in cfg.items() if k.endswith(("_source", "_training")) or k in ("taper", "sigreg_projection")})
loss, model = r.loss_module, r.model
import os
if os.environ.get("OLD_SIGREG_STREAM"): loss.sigreg.use_generator = False; print("SIGReg drawing from the global stream, as in July")
def same(a, b): return bool(torch.equal(torch.as_tensor(a), torch.as_tensor(b)))
loss.eval(); ok = True
with torch.no_grad():
    for i, b in enumerate(ref["batches"][:3]):
        torch.manual_seed(1000 + i)
        out = loss.compute_modality_loss("text", {"text_tokens": b})
        d = {k: (float(torch.as_tensor(out[k]).float().mean()), float(v.float().mean())) for k, v in ref["eval"][i].items() if k in out and not same(out[k], v)}
        ok &= "l_pred" not in d
        print("eval ", i, "l_pred same" if "l_pred" not in d else "l_pred DIFFERENT", "| differ:", d)
now = model.state_dict()
diff = [k for k, v in ref["after_eval"].items() if not torch.equal(v, now[k])]
print("after eval: entries compared", len(ref["after_eval"]), "different", len(diff), diff[:6]); ok &= not diff
loss.train()
with torch.no_grad():
    for i, b in enumerate(ref["batches"][3:]):
        torch.manual_seed(2000 + i)
        out = loss.compute_modality_loss("text", {"text_tokens": b})
        d = {k: (float(torch.as_tensor(out[k]).float().mean()), float(v.float().mean())) for k, v in ref["train"][i].items() if k in out and not same(out[k], v)}
        ok &= "l_pred" not in d
        print("train", i, "l_pred same" if "l_pred" not in d else "l_pred DIFFERENT", "| differ:", d)
now = model.state_dict()
diff = [k for k, v in ref["after_train"].items() if not torch.equal(v, now[k])]
print("after train: different", len(diff), diff[:6]); ok &= not diff
moved = [k for k in ref["after_train"] if "living_ffn" in k and not torch.equal(ref["after_train"][k], torch.load(r.checkpoint, weights_only=False)["online_state_dict"][k])]
print("living entries the forwards changed (so the comparison is of a layer that was writing):", len(moved))
print("VERDICT:", "SAME MODEL" if ok else "NOT THE SAME")
