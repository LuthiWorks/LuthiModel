"""Shared pieces for the half-day probes: twin lives that differ in one early
episode, lived by perception alone (one living forward a moment)."""
import copy, os, random, sys, tempfile
from pathlib import Path
os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
LUTHI = str(Path(__file__).resolve().parents[2]); SANCT = os.environ["SANCTUARY_PATH"]
os.environ["LUTHI_PATH"] = LUTHI
sys.path.insert(0, LUTHI); sys.path.insert(0, SANCT)
import torch
import luthi.sanctuary_interface as L
from luthi.v2.living_layer_pc import PredictiveCodingLayer
from sanctuary.core.schema import Percept
from sanctuary.life.senses import Line

WORDS = ("river stone light memory window garden quiet music signal field morning "
         "shadow pattern voice letter bridge winter colour thread").split()
A = "a zebra walked slowly past the old iron gate"
B = "seven lanterns floated over the frozen harbour"

def load(run):
    loaded = L.load_m9_run(run, life_dir=tempfile.mkdtemp(prefix="life_"), device="cpu")
    loaded.trainer.action_log.close()
    model = loaded.model
    model.eval()
    return model, loaded.tokenizer, int(loaded.config["seq_len"])

def layers_of(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, PredictiveCodingLayer)}

class Life:
    """One mind living by perception. `episode` is the sentence it hears in
    the episode window; everything else is the same schedule for every life."""
    def __init__(self, run, episode, t0, length, seed=0):
        self.model, tok, self.window = load(run)
        self.line = Line(tok.encode, self.window)
        self.layers = layers_of(self.model)
        self.episode, self.t0, self.length = episode, t0, length
        self.t = 0
        self.rng = random.Random(seed)          # the background, the same in every life
        self.sum_disp = {n: torch.zeros_like(m.weight) for n, m in self.layers.items()}
        self.forwards = 0
        for n, m in self.layers.items():
            m.register_forward_hook(self._after(n))
    def _after(self, name):
        def hook(mod, args, out):
            self.sum_disp[name] += (mod.weight - mod.set_point).detach()
            if name == next(iter(self.layers)):
                self.forwards += 1
        return hook
    def moment(self):
        background = " ".join(self.rng.choice(WORDS) for _ in range(7))   # drawn every moment, so the stream never depends on the episode
        if self.t % 3 == 0:
            said = self.episode if self.t0 <= self.t < self.t0 + self.length else background
            self.line.add(Percept(modality="language", content=said, source="brian"))
        tokens = self.line.take().tokens
        L.perceive(self.model, text_tokens=torch.tensor([tokens], dtype=torch.long))
        self.t += 1
    def live(self, moments):
        for _ in range(moments):
            self.moment()
    def snap(self):
        return {n: {"W": m.weight.detach().clone(), "home": m.set_point.detach().clone(),
                    "P": m.prediction.detach().clone()} for n, m in self.layers.items()}
    def fork(self):
        """A copy that lives on separately from here. Hooks are re-made on the copy."""
        other = copy.copy(self)
        other.model = copy.deepcopy(self.model)
        for m in other.model.modules():
            m._forward_hooks.clear()
        other.layers = layers_of(other.model)
        other.line = copy.deepcopy(self.line)
        other.rng = random.Random(); other.rng.setstate(self.rng.getstate())
        other.sum_disp = {n: v.clone() for n, v in self.sum_disp.items()}
        for n, m in other.layers.items():
            m.register_forward_hook(other._after(n))
        return other

def share(x, y, d):
    """How much of direction d (what the episode wrote) is in x - y."""
    diff = (x - y).double().flatten(); d = d.double().flatten()
    return float(diff @ d / (d @ d)), float(diff @ d / (diff.norm() * d.norm()).clamp_min(1e-300))
