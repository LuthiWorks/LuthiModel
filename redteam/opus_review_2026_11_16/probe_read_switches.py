"""The five read switches of ``adaptation_eval``, checked where the tests do not reach.

Opus 5.5 review of 2026-11-16 (commits 3443831 and af4328d). The tests pin
the default path, the restore, and an exact null with all four routes off.
That null is shown on a toy whose consolidation never fires (its warmup is
1,000 forwards and a test reads 6), and on the card at 4 windows, which is
before consolidation first fires on a July run (window 15). So nothing yet
shows the consolidation switch holding the null while consolidation would
otherwise have fired. And every null so far is in eval mode.

On the tests' toy (two blocks, 32 wide), with consolidation set to fire on
every forward after a 4-forward warmup, and with the stores filled by a few
plastic reads first so consolidation has episodes to replay:

1. Only consolidation on (write, pull and episode writes off): does the score move?
2. Everything off: is the null exact while consolidation would fire?
3. Everything off with ``read_train_mode=True``: is that null exact too?
4. Each route alone and everything on: do the parts add up (at this tiny dose)?

Run from the repo root:  python redteam/opus_review_2026_11_16/probe_read_switches.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LUTHI_ALLOW_CPU", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from luthi.v2 import adaptation_eval as ae
from luthi.v2.adaptation_eval import adaptation_eval, holdout_documents
from luthi.v2.jepa_loss import JEPALoss
from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

V, D, L = 64, 32, 16


def toy() -> JEPALoss:
    from luthi.v2.consolidation import ConsolidationTracker
    torch.manual_seed(0)
    m = MultimodalPredictiveCodingLM(
        vocab_size=V, d_model=D, n_blocks=2, n_heads=4, max_seq_len=L,
        consolidation_enabled=True,
    )
    # The model passes no trigger settings through, so each layer gets a
    # quick trigger here: warm after 4 forwards, then fire on every forward.
    # And each layer's own store admits every episode (the toy's salience is
    # below the default floor of 0.1, so its store would stay empty and
    # consolidation would replay nothing).
    for block in m.blocks:
        block.living_ffn._consolidation_tracker = ConsolidationTracker(
            window_size=4, trigger_window=1, threshold_factor=1e9)
        block.living_ffn.salience_threshold = 0.0
    lm = JEPALoss(m, visreg_lambda=0.6, sigreg_projection="none")
    # Fill the stores and warm the trigger: a few plastic reads before the eval.
    g = torch.Generator().manual_seed(7)
    ae._read(m, torch.randint(0, V, (8, L), generator=g), repeats=1, read_train_mode=False)
    return lm


def docs() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    windows = torch.randint(0, V, (4 * 6 * 2, L), generator=g)
    return holdout_documents(windows, n_read=4, n_score=2, n_docs=4)


ON = dict()
OFF = dict(rate_scale=0.0, homeostatic_decay=0.0, read_consolidation=False, read_episode_writes=False)
ONLY = {
    "the write": dict(homeostatic_decay=0.0, read_consolidation=False, read_episode_writes=False),
    "the pull": dict(rate_scale=0.0, read_consolidation=False, read_episode_writes=False),
    "consolidation": dict(rate_scale=0.0, homeostatic_decay=0.0, read_episode_writes=False),
    "episode writes": dict(rate_scale=0.0, homeostatic_decay=0.0, read_consolidation=False),
}


def run(**kw) -> dict:
    """One eval on a fresh toy. Consolidation's own counters are living state
    and the eval restores them, so replays are counted from outside."""
    import luthi.v2.consolidation as consolidation
    lm = toy()
    replays = []
    real = consolidation.consolidate_layer

    def counted(layer, **k):
        n = real(layer, **k)
        replays.append(int(n or 0))
        return n

    consolidation.consolidate_layer = counted
    try:
        res = adaptation_eval(lm, docs(), n_read=4, read_repeats=2, **kw)
    finally:
        consolidation.consolidate_layer = real
    res["fired"] = len(replays)
    res["replayed"] = sum(replays)
    return res


def b_minus_a(res: dict) -> list[float]:
    return [p["B"]["l_pred"] - p["A"]["l_pred"] for p in res["per_doc"]]


def main() -> None:
    lm = toy()
    held = [int(b.living_ffn.episode_count) for b in lm.online_encoder.blocks]
    print(f"toy ready; consolidation fires on every forward after warmup; "
          f"episodes in each layer's own store: {held}")

    only_c = run(**ONLY["consolidation"])
    print(f"1. only consolidation: exact null {only_c['summary']['exact_null']}; "
          f"consolidation fired {only_c['fired']} times during the eval, replaying "
          f"{only_c['replayed']} episodes; B-A per document {[f'{x:+.2e}' for x in b_minus_a(only_c)]}")

    off = run(**OFF)
    print(f"2. everything off: exact null {off['summary']['exact_null']}; "
          f"consolidation fired {off['fired']} times during the eval")

    off_train = run(read_train_mode=True, **OFF)
    print(f"3. everything off, reads in train mode: exact null {off_train['summary']['exact_null']}")
    if not off_train["summary"]["exact_null"]:
        print(f"   B-A per document {[f'{x:+.2e}' for x in b_minus_a(off_train)]}")

    print("4. each route alone against everything on (mean B-A over documents):")
    total = 0.0
    for name, kw in ONLY.items():
        r = run(**kw)
        m = sum(b_minus_a(r)) / len(r["per_doc"])
        total += m
        print(f"   only {name:15s} {m:+.3e}")
    on = run(**ON)
    m_on = sum(b_minus_a(on)) / len(on["per_doc"])
    print(f"   sum of the parts      {total:+.3e}")
    print(f"   everything on         {m_on:+.3e}")
    leave = run(rate_scale=0.0)
    m_leave = sum(b_minus_a(leave)) / len(leave["per_doc"])
    print(f"   everything but the write {m_leave:+.3e}; so the write's part in context "
          f"(on minus everything-but) {m_on - m_leave:+.3e}, against only the write")


if __name__ == "__main__":
    main()
