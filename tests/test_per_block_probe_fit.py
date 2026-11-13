"""Where scripts/per_block_probe.py fits its probes (2026-11-13). The fit
could only run on the CPU, and twelve jobs took 14 to 16 hours on Brian's
machine. It can now run on the model's device. This box has no card, so
what is pinned here is that the default is unchanged and that the device is
carried through; that the card gives the same answers is for the card."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.per_block_probe import _accuracy, _fit_probe, _standardizer  # noqa: E402


def _data(n: int = 600, d: int = 12, vocab: int = 7):
    gen = torch.Generator().manual_seed(3)
    y = torch.randint(0, vocab, (n,), generator=gen)
    centres = torch.randn(vocab, d, generator=gen) * 3
    x = centres[y] + torch.randn(n, d, generator=gen) * 0.3
    return x, y, vocab


def test_the_fit_learns_and_the_default_is_the_cpu_fit_it_always_was():
    x, y, vocab = _data()
    mu, sd = _standardizer(x)
    plain = _fit_probe(x, y, vocab, 80, 1e-2, 0, mu, sd)
    named = _fit_probe(x, y, vocab, 80, 1e-2, 0, mu, sd, torch.device("cpu"))
    assert torch.equal(plain.weight, named.weight) and torch.equal(plain.bias, named.bias)
    assert plain.weight.device.type == "cpu"
    acc = _accuracy(plain, x, y, False, 0, mu, sd)
    floor = _accuracy(plain, x, y, True, 0, mu, sd)
    assert acc["top1"] > 0.9 > 0.4 > floor["top1"]


def test_the_default_on_the_command_line_is_the_cpu():
    text = (REPO_ROOT / "scripts" / "per_block_probe.py").read_text(encoding="utf-8")
    assert '"--fit-device", choices=("cpu", "run"), default="cpu"' in text
