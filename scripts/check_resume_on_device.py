"""Does a run that loses power and resumes end where it would have ended?

Run this on the training machine, on the training backend, before a long
family. It trains one short run of an arm three times:

  straight_a   start to finish
  straight_b   start to finish again (the same run, a second time)
  cut          killed without warning partway, then started again, so it
               resumes from its newest checkpoint the way resume_queue.py
               would after a power cut

Each run saves one checkpoint, a few steps before the cut, and one at the
end. The final checkpoints are compared. On CPU all three are identical
(tests/test_resume_matches_uninterrupted.py). A GPU backend may not repeat
itself bit for bit, so the question here is whether ``cut`` differs from
``straight_a`` by more than ``straight_b`` does. ``straight_b`` is the noise
of the backend. A resume that loses state shows up as a difference well
above it.

    python scripts/check_resume_on_device.py
    python scripts/check_resume_on_device.py --arm lc768_living --d-model 768 --steps 60 --cut-at 25

Everything is written under --out (default: a fresh temp directory), never
under runs/. Nothing here reads or writes an existing run. At 768x8 a
checkpoint is about 1.4 GB and this keeps six, so give --out a drive with
10 GB free. The directory can be deleted afterwards.

The default reads --data-dir, not the arm's registered file list, so it
starts in seconds. Pass --registered-corpus to load the arm's real corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Differ between any two runs by construction, and are not the run's state.
NOT_STATE = ("wall_clock_seconds", "reason", "rng_state")
KILLED = 137
LAG = 5     # steps between the last save and the cut


def _child(ns: argparse.Namespace) -> int:
    """One run of the driver. With --die-at, the process ends at that step
    with no cleanup and no parting checkpoint."""
    import scripts.jepa_pilot_driver as driver
    from luthi.v2.jepa_runner import JEPATrainer

    driver.OUTPUT_ROOT = Path(ns.out) / ns.child
    if not ns.registered_corpus:
        driver.ARM_FILELIST.pop(ns.arm, None)
    real_step = JEPATrainer.train_step
    real_due = JEPATrainer._checkpoint_if_due
    saved = []

    def due(self):
        # Every run saves once, at the same step, so the three runs differ
        # only in whether one of them dies. The save goes where the trainer
        # saves, after a step is complete. (Saving from inside train_step
        # would catch the loader after it had drawn the next batch, and the
        # resume would skip that batch: a fault of the check, not the run.)
        if self.global_step == ns.cut_at - LAG and not saved:
            self._checkpoint(reason="resume_check")
            saved.append(True)
        return real_due(self)

    def stepping(self, *a, **k):
        # The cut run dies LAG steps after its save, so it loses those
        # steps and has to train them again.
        if ns.die_at >= 0 and self.global_step >= ns.die_at:
            os._exit(KILLED)
        return real_step(self, *a, **k)

    JEPATrainer._checkpoint_if_due = due
    JEPATrainer.train_step = stepping
    args = argparse.Namespace(
        data_dir=ns.data_dir, tokenizer=ns.tokenizer, n_blocks=ns.n_blocks,
        epochs=1, batch_size=ns.batch_size, seq_len=ns.seq_len,
        stride=ns.seq_len, lr=3e-4, holdout_fraction=0.02,
        heldout_batches=2, probe_batches=2, kill_warmup=10**9,
        checkpoint_interval=10**9,   # only the one save above, and the end
        checkpoint_slots=3, max_batches_per_epoch=ns.steps,
    )
    driver._run_one(ns.arm, ns.d_model, ns.seed, args)
    return 0


def _flat(obj, at=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f"{at}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flat(v, f"{at}[{i}]")
    else:
        yield at, obj


def _final(run_root: Path) -> dict:
    import torch

    path = sorted(run_root.rglob("ckpt_*.pt"))[-1]
    return torch.load(path, map_location="cpu", weights_only=False)


def compare(a: dict, b: dict) -> dict:
    """How far apart two final checkpoints are.

    ``worst`` is the largest relative difference of any tensor
    (||a - b|| / ||a||). ``counters`` lists every non-tensor entry that
    differs: step counts, token counts, lived-state counters.
    """
    import torch

    fa, fb = dict(_flat(a)), dict(_flat(b))
    tensors, counters, worst, worst_at = 0, [], 0.0, ""
    for key, x in fa.items():
        if key not in fb or any(s in key for s in NOT_STATE):
            continue
        y = fb[key]
        if torch.is_tensor(x):
            if x.shape != y.shape:
                counters.append(f"{key}: shape {tuple(x.shape)} != {tuple(y.shape)}")
                continue
            if torch.equal(x, y):
                continue
            tensors += 1
            xf, yf = x.double(), y.double()
            rel = float((xf - yf).norm() / (xf.norm() + 1e-30))
            if rel > worst:
                worst, worst_at = rel, key
        elif isinstance(x, float):
            if not (x == y or (x != x and y != y)):
                if isinstance(y, float) and abs(x - y) <= 1e-9 * max(abs(x), abs(y), 1.0):
                    continue
                counters.append(f"{key}: {x!r} != {y!r}")
        elif x != y:
            counters.append(f"{key}: {x!r} != {y!r}"[:160])
    return {
        "only_in_one": sorted(set(fa) ^ set(fb)),
        "tensors_differing": tensors,
        "worst": worst,
        "worst_at": worst_at,
        "counters": counters,
    }


def verdict(noise: dict, resumed: dict, steps: int, final_step: int) -> tuple[bool, str]:
    """True if the resumed run is the same run, to within the backend's own
    noise. Counters must match exactly: a resume never changes a count."""
    if final_step != steps:
        return False, f"the resumed run ended at step {final_step}, not {steps}"
    if resumed["only_in_one"]:
        return False, f"entries present in only one run: {resumed['only_in_one'][:5]}"
    extra = [c for c in resumed["counters"] if c not in noise["counters"]]
    if extra:
        return False, f"{len(extra)} counter(s) differ after the resume, e.g. {extra[0]}"
    if resumed["tensors_differing"] == 0:
        return True, "the resumed run is identical to the uninterrupted one"
    if noise["tensors_differing"] == 0:
        return False, (
            "two uninterrupted runs were identical, and the resumed run was "
            f"not (worst {resumed['worst']:.3e} at {resumed['worst_at']})"
        )
    if resumed["worst"] <= 10.0 * noise["worst"]:
        return True, (
            f"the resumed run differs by {resumed['worst']:.3e}, within ten "
            f"times the backend's own noise ({noise['worst']:.3e})"
        )
    return False, (
        f"the resumed run differs by {resumed['worst']:.3e} at "
        f"{resumed['worst_at']}, more than ten times the backend's own "
        f"noise ({noise['worst']:.3e})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", default="lc768_living")
    p.add_argument("--d-model", dest="d_model", type=int, default=768)
    p.add_argument("--n-blocks", dest="n_blocks", type=int, default=2,
                   help="Driver default; arms that register a depth override it.")
    p.add_argument("--seed", type=int, default=46)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--cut-at", dest="cut_at", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--data-dir", dest="data_dir",
                   default=str(REPO_ROOT / "corpus_build" / "gutenberg_100"))
    p.add_argument("--tokenizer",
                   default=str(REPO_ROOT / "corpus_build" / "tokenizer_32k.json"))
    p.add_argument("--registered-corpus", dest="registered_corpus",
                   action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--die-at", dest="die_at", type=int, default=-1,
                   help=argparse.SUPPRESS)
    ns = p.parse_args()
    if ns.child:
        return _child(ns)
    if not LAG < ns.cut_at < ns.steps:
        p.error(f"--cut-at must be above {LAG} and inside the run")

    out = Path(ns.out) if ns.out else Path(tempfile.mkdtemp(prefix="luthi_resume_check_"))
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error(f"--out {out} is not empty; this check starts from nothing")
    base = [sys.executable, str(Path(__file__).resolve())]
    for key, value in vars(ns).items():
        if key in ("child", "die_at", "out", "registered_corpus") or value is None:
            continue
        base += [f"--{key}" if key in ("batch_size", "seq_len", "arm", "seed",
                                        "steps", "tokenizer")
                 else "--" + key.replace("_", "-"), str(value)]
    base += ["--out", str(out)]
    if ns.registered_corpus:
        base.append("--registered-corpus")

    def run(name: str, die_at: int = -1) -> int:
        cmd = base + ["--child", name, "--die-at", str(die_at)]
        with open(out / f"{name}.log", "ab") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode

    print(f"[resume-check] {ns.arm} {ns.d_model}d, {ns.steps} steps, cut at {ns.cut_at}")
    print(f"[resume-check] writing under {out}")
    for name in ("straight_a", "straight_b"):
        if run(name) != 0:
            print(f"[resume-check] {name} failed; see {out / (name + '.log')}")
            return 2
        print(f"[resume-check] {name}: done")
    code = run("cut", die_at=ns.cut_at)
    if code == 0:
        print("[resume-check] the cut run finished without being cut; nothing was tested")
        return 2
    print(f"[resume-check] cut: killed at step {ns.cut_at} (exit {code})")
    if run("cut") != 0:
        print(f"[resume-check] the resume failed; see {out / 'cut.log'}")
        return 2
    print("[resume-check] cut: resumed and finished")

    a, b, c = (_final(out / n) for n in ("straight_a", "straight_b", "cut"))
    noise, resumed = compare(a, b), compare(a, c)
    ok, why = verdict(noise, resumed, ns.steps, int(c["global_step"]))
    report = {
        "arm": ns.arm, "d_model": ns.d_model, "steps": ns.steps,
        "cut_at": ns.cut_at, "final_step_resumed": int(c["global_step"]),
        "noise_between_two_uninterrupted_runs": noise,
        "resumed_against_uninterrupted": resumed,
        "same_run": ok, "why": why,
    }
    (out / "resume_check.json").write_text(json.dumps(report, indent=2))
    print(f"[resume-check] two uninterrupted runs: {noise['tensors_differing']} "
          f"tensors differ, worst {noise['worst']:.3e}")
    print(f"[resume-check] resumed against uninterrupted: "
          f"{resumed['tensors_differing']} tensors differ, worst {resumed['worst']:.3e}")
    print(f"[resume-check] {'SAME RUN' if ok else 'NOT THE SAME RUN'}: {why}")
    print(f"[resume-check] report: {out / 'resume_check.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
