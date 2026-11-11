"""What can a rest put in home from an episode early in a (scaled) half-day?

Twin lives. X hears sentence A in the episode window, Y hears B; everything
else is identical. One living forward a moment. At the end of the half-day each
candidate takes a step in home of the same size (BETA of home's norm, unless
marked 'as built'), poured over R resting moments. Then a washout, then:

  share   how much of the episode's own change (W_X - W_Y at the end of the
          episode) is in home_X - home_Y, as a percentage of that change
  cos     how well home_X - home_Y lines up with it
  apart   how different the twins' steps were: |step_X - step_Y| / |step_X|
  own     each life's own layer error on its own episode's view against the
          other's (own W, own P; positive = better on its own)

Results of 2026-11-11 and how to read them:
docs/research/2026-11-11_what-a-rest-can-carry.md. Train the trunk first with
halfday_train_trunk.py <dir>, then pass <dir>/runs/living_32d_seed42 here.
Needs SANCTUARY_PATH. Use OMP_NUM_THREADS=1 when running two at once.
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from halfday_lib import *
from luthi.v2.day_record import enable_day_record, day_records
from luthi.v2.rest import rest as rest_pass
from luthi.v2.plasticity import freeze_plasticity

run = sys.argv[1]
T0 = int(os.environ.get("T0", 6000)); LEN = int(os.environ.get("LEN", 2500)); N = int(os.environ.get("N", 100000))
R = int(os.environ.get("R", 300)); WASH = int(os.environ.get("WASH", 5000)); BETA = float(os.environ.get("BETA", 0.05))
BG_SEED = int(os.environ.get("BG_SEED", 0))
t_start = time.time()

class Tracked(Life):
    """A life that also keeps the running sums the candidates need."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        enable_day_record(self.model)
        z = lambda: {n: torch.zeros_like(m.weight) for n, m in self.layers.items()}
        self.sum_writes, self.sum_surprise_writes = z(), z()
        self.before = {}
        self.records = {n: 0 for n in self.layers}
        self.P0 = {n: m.prediction.detach().clone() for n, m in self.layers.items()}
        for n, m in self.layers.items():
            m.register_forward_pre_hook(lambda mod, args, n=n: self.before.__setitem__(n, mod.weight.detach().clone()))
            m.register_forward_hook(self._writes(n))
    def _writes(self, n):
        def hook(mod, args, out):
            # what this forward wrote, before homeostasis pulled some of it back (to first order)
            write = (mod.weight - self.before[n]) + mod.homeostatic_decay * (mod.weight - mod.set_point)
            self.sum_writes[n] += write.detach()
            now = len(mod._day_record)
            if now > self.records[n]:
                self.sum_surprise_writes[n] += write.detach()
            self.records[n] = now
        return hook
    def fork(self):
        other = super().fork()
        other.sum_writes = {n: v.clone() for n, v in self.sum_writes.items()}
        other.sum_surprise_writes = {n: v.clone() for n, v in self.sum_surprise_writes.items()}
        other.before = {}; other.records = dict(self.records)
        for n, m in other.layers.items():
            m.register_forward_pre_hook(lambda mod, args, n=n, o=other: o.before.__setitem__(n, mod.weight.detach().clone()))
            m.register_forward_hook(other._writes(n))
        return other

X, Y = Tracked(run, A, T0, LEN, seed=BG_SEED), Tracked(run, B, T0, LEN, seed=BG_SEED)
names = list(X.layers)

def views(life):
    out = {}; tok = life.line._encode
    for label, sentence in (("A", A), ("B", B)):
        line = Line(tok, life.window)
        for i in range(12):
            if i % 3 == 0:
                line.add(Percept(modality="language", content=sentence, source="brian"))
            tokens = line.take().tokens
        caught = {}
        hooks = [m.register_forward_pre_hook(lambda mod, args, n=n: caught.__setitem__(n, args[0].detach().reshape(-1, args[0].shape[-1]).mean(0)))
                 for n, m in life.layers.items()]
        saved = ({n: v.clone() for n, v in life.sum_disp.items()}, {n: v.clone() for n, v in life.sum_writes.items()},
                 {n: v.clone() for n, v in life.sum_surprise_writes.items()}, life.forwards)
        with freeze_plasticity(life.model), torch.no_grad():
            life.model.encode(text_tokens=torch.tensor([tokens], dtype=torch.long), causal=False)
        for h in hooks: h.remove()
        life.sum_disp, life.sum_writes, life.sum_surprise_writes, life.forwards = saved
        out[label] = caught
    return out

def own(x, y):
    vx, vy = views(x), views(y); out = {}
    for n in names:
        e = lambda v, m: float((v - (v @ m.weight.T) @ m.prediction).pow(2).sum())
        xa, xb, ya, yb = e(vx["A"][n], x.layers[n]), e(vx["B"][n], x.layers[n]), e(vy["A"][n], y.layers[n]), e(vy["B"][n], y.layers[n])
        out[n] = (xb + ya - xa - yb) / (xa + xb + ya + yb)
    return out

def line_for(tag, x, y, D, steps=None):
    sx, sy = x.snap(), y.snap(); o = own(x, y); parts = []
    for n in names:
        sh, c = share(sx[n]["home"], sy[n]["home"], D[n])
        shW, _ = share(sx[n]["W"], sy[n]["W"], D[n])
        apart = ""
        if steps is not None:
            apart = f" apart {float((steps[0][n]-steps[1][n]).norm()/steps[0][n].norm().clamp_min(1e-30)):5.2f}"
        parts.append(f"{n[7]}: home share {100*sh:7.3f}% cos {c:+.2f}{apart} | W share {100*shW:7.2f}% | own {100*o[n]:+6.2f}%")
    print(f"  {tag:26s} " + "   ".join(parts), flush=True)

X.live(T0); Y.live(T0)
assert all(torch.equal(X.layers[n].weight, Y.layers[n].weight) for n in names)
X.live(LEN); Y.live(LEN)
D = {n: (X.layers[n].weight - Y.layers[n].weight).detach().clone() for n in names}
for life in (X, Y):   # where the weight stood, from home, at the end of this life's own episode
    life.aimed = {n: (m.weight - m.set_point).detach().clone() for n, m in life.layers.items()}
print(f"episode {T0}..{T0+LEN} ({100*LEN/N:.1f}% of the half-day of {N}); step size {BETA} of |home|; rest {R}; washout {WASH}; background seed {BG_SEED}")
for n in names:
    m = X.layers[n]
    print(f"  {n}: |home| {float(m.set_point.norm()):.2f}  standing gap {float((m.weight-m.set_point).norm()):.2f}  the episode's own change {float(D[n].norm()):.2f}")
line_for("end of episode", X, Y, D)
X.live(N - X.t); Y.live(N - Y.t)
line_for("end of half-day", X, Y, D)
for n in names:
    print(f"  {n}: twins' weights now differ by {float((X.layers[n].weight - Y.layers[n].weight).norm()):.2f}; moments the layer found surprising: {X.records[n]} of {X.forwards}")

def direction(life, cand):
    if cand.startswith("last"):    return {n: (m.weight - m.set_point).detach().clone() for n, m in life.layers.items()}
    if cand.startswith("trace"):   return {n: life.sum_disp[n] / life.forwards for n in names}
    if cand.startswith("writes"):  return {n: life.sum_writes[n].clone() for n in names}
    if cand.startswith("surprise"):return {n: life.sum_surprise_writes[n].clone() for n in names}
    if cand.startswith("aimed"):   return {n: life.aimed[n].clone() for n in names}
    if cand.startswith("pledger"): return {n: (life.layers[n].prediction - life.P0[n]).detach().clone() for n in names}
    raise ValueError(cand)

for cand in os.environ.get("CANDS", "none,last as built,last,trace,surprise,pledger,replay as built,aimed,aimed as built").split(","):
    x, y = X.fork(), Y.fork(); steps = []
    for life in (x, y):
        step = None
        if cand == "replay as built":
            home0 = {n: m.set_point.detach().clone() for n, m in life.layers.items()}
            rest_pass(life.model, keep=[], toss=[], unjudged="keep")
            steps.append({n: life.layers[n].set_point.detach() - home0[n] for n in names})
        elif cand != "none":
            d = direction(life, cand)
            if cand.endswith("as built"):
                step = {n: d[n].clone() for n in names}
            else:
                step = {n: d[n] * (BETA * life.layers[n].set_point.norm() / d[n].norm().clamp_min(1e-30)) for n in names}
            steps.append({n: step[n].clone() for n in names})
            if life is x:
                print(f"    [{cand}] step is " + ", ".join(f"{float(step[n].norm()/life.layers[n].set_point.norm()):.3f}" for n in names) + " of |home|", flush=True)
        for r in range(R):
            life.moment()
            if step is not None:
                with torch.no_grad():
                    for n in names:
                        life.layers[n].set_point.add_(step[n], alpha=0.01); step[n] = step[n] * 0.99
        life.live(WASH)
    line_for(f"[{cand}]", x, y, D, steps if steps else None)
print(f"({time.time()-t_start:.0f}s)")
