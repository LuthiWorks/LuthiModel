"""Review probe (Opus 5.5, 2026-11-10): the moment measurement with a persistence
baseline (standing rule 3: baselines, not floors).

Run from the LuthiModel root, with Sanctuary importable:

    PYTHONPATH=<Sanctuary>:<LuthiModel> LUTHI_PATH=<LuthiModel> N=300 SEEDS="42 43 44" \
        python redteam/opus_review_2026_11_10/probe_moment_baseline.py

``scripts/measure_what_a_moment_holds.py`` compares the lived prediction error
across the three ways of filling a moment. The error of moment i scores the
predictor's forecast of s_i from moment i-1's context. Its baseline here is the
error of the trivial forecast s_i = s_{i-1}: mean((s_i - s_{i-1})^2), the same
metric. Same trained models, worlds and lives as the script; 300 moments.

Printed on 2026-11-10:

    mode     seed  median error  persistence baseline  error/baseline  moments below baseline
    new      42    7.20          7.22                  0.85            60%
    rolling  42    0.57          0.0009                473             0%
    spoken   42    1.07          0.19                  5.7             0%
    new      43    3.30          3.70                  0.89            61%
    rolling  43    0.79          0.0013                610             0%
    spoken   43    1.72          0.090                 19              0%
    new      44    5.71          5.83                  1.00            51%
    rolling  44    0.43          0.0010                303             0%
    spoken   44    1.41          0.19                  7.8             0%

Under the rolling view consecutive states barely move, so its low error is
mostly the absence of change, not prediction. Against "nothing changes", it is
the worst of the three.
"""

import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import measure_what_a_moment_holds as m  # noqa: E402
import torch  # noqa: E402
from sanctuary.core.schema import Percept  # noqa: E402
from sanctuary.life import LuthiMind  # noqa: E402
from sanctuary.life.senses import QUIET, frame  # noqa: E402


def live(run_dir, mode, seed, n):
    torch.manual_seed(0)
    mind = LuthiMind(run_dir, m.ROOT / f"probe_{mode}_{seed}")
    trainer = mind._trainer
    errors, read = [], trainer._read_felt

    def spy(lived):
        if "lived_l_pred_clean" in lived:
            errors.append(lived["lived_l_pred_clean"])
        return read(lived)

    trainer._read_felt = spy
    quiet, stream, states = mind.tokenizer.encode(QUIET), [], []
    for i in range(n):
        new = None
        if i % 3 == 0:
            new = mind.tokenizer.encode(frame(Percept(modality="language", content=m.said(i), source="brian")))
        if mode == "spoken" and new is None:
            continue
        tokens = new or quiet
        if mode == "rolling":
            stream.extend(tokens)
            tokens = stream[-mind.window:]
        mind.live(tokens)
        states.append(mind._previous[0].detach().clone())
    persist = [float((b - a).pow(2).mean()) for a, b in zip(states, states[1:])]
    assert len(persist) == len(errors)
    ratio = [e / max(p, 1e-12) for e, p in zip(errors, persist)]
    return (statistics.median(errors), statistics.median(persist),
            statistics.median(ratio), sum(r < 1 for r in ratio) / len(ratio))


if __name__ == "__main__":
    n = int(os.environ.get("N", "300"))
    for seed in [int(s) for s in os.environ.get("SEEDS", "42 43 44").split()]:
        run_dir = m.train(m.ROOT, seed)
        for mode in m.MODES:
            err, base, ratio, below = live(run_dir, mode, seed, n)
            print(f"{mode:8s} seed {seed}: median error {err:.4f}  persistence baseline {base:.4f}  "
                  f"error/baseline {ratio:.3f}  below baseline {100 * below:.0f}%", flush=True)
