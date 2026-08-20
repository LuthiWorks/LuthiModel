"""Time a real JEPA training step at the ruled 768x8 config, on whichever
backend this interpreter resolves. Progress-printed so a hang is visible."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM
from luthi.v2.jepa_loss import JEPALoss
from luthi.v2 import pc_ops
if os.environ.get("BENCH_NO_CPP") == "1":
    pc_ops._use_cpp = False

def get_dev():
    try:
        import torch_directml
        return torch_directml.device(), "directml"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), ("rocm" if getattr(torch.version,"hip",None) else "cuda")
    return torch.device("cpu"), "cpu"

def sync(tag):
    if tag in ("rocm","cuda"):
        torch.cuda.synchronize()

dev, tag = get_dev()
print(f"backend={tag} torch={torch.__version__} cpp_ops={pc_ops._use_cpp}", flush=True)
D = int(os.environ.get("BENCH_D", 768)); NB = int(os.environ.get("BENCH_NB", 8))
B = int(os.environ.get("BENCH_B", 32));  L = int(os.environ.get("BENCH_L", 128))
N = int(os.environ.get("BENCH_N", 10))
print(f"building {D}d x {NB} blocks, batch {B}, seq {L} ...", flush=True)
t = time.perf_counter()
torch.manual_seed(0)
model = MultimodalPredictiveCodingLM(
    vocab_size=32000, d_model=D, n_blocks=NB, n_heads=8,
    ffn_expansion=1, max_seq_len=L, backward_pass_enabled=True).to(dev)
loss_mod = JEPALoss(online_encoder=model, sigreg_lambd=0.0, visreg_lambda=0.6,
                    visreg_num_proj=2*D, sigreg_projection="none").to(dev)
opt = torch.optim.AdamW([p for p in loss_mod.parameters() if p.requires_grad], lr=3e-4)
tok = torch.randint(0, 32000, (B, L), device=dev)
sync(tag); print(f"  build+move {time.perf_counter()-t:.1f}s", flush=True)

def step():
    opt.zero_grad(set_to_none=True)
    out = loss_mod.compute_modality_loss("text", {"text_tokens": tok})
    out["loss"].backward(); opt.step()

t = time.perf_counter(); step(); sync(tag)
print(f"  first step (incl kernel warmup) {time.perf_counter()-t:.1f}s", flush=True)
for i in range(2):
    t = time.perf_counter(); step(); sync(tag)
    print(f"  warm step {i+1}: {time.perf_counter()-t:.2f}s", flush=True)
t0 = time.perf_counter()
for _ in range(N):
    step()
sync(tag)
dt = (time.perf_counter()-t0)/N
print(f"RESULT backend={tag} {dt*1000:.0f} ms/step ({dt:.3f} s/step) "
      f"at {D}d x {NB}, batch {B}", flush=True)
