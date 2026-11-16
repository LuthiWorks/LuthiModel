"""What the episode route does during a read and at the score, store by store.

Read 4 of the second card brief (2026-11-16). With the write, the pull and
consolidation off, a read reaches the frozen score only through episodes.
On July seed 42 that read is an exact zero on 32 documents; on 768 seed 97
it moves 5 documents of 32, one of them by +0.021. This says why, and tells
a store that is healthy and does nothing from one that is quiet.

There are two stores in every block:

- the block's ``EpisodeStore`` ("block"): recalls the best-matching stored
  output and blends it into the block's output;
- the living layer's own store ("layer"): recalls the best-matching stored
  weight and blends the difference into the weight it computes with.

For each document (the E1 layout: ``--read`` windows read, ``--score``
scored), with write rate 0, pull 0 and consolidation held off:

    A        frozen score, nothing read
    B        after reading its own windows, both stores writing
    B_block  the same, only the block stores writing
    B_layer  the same, only the layers' own stores writing
    C        after reading another document's windows, both stores writing

and for every store, in every one of those: how many times recall was
asked and fired, the best similarity each time, which slot was recalled;
how many episodes a read offered, their salience against the bar, how many
were written and to which slot; and whether the score then recalled a slot
that this read had just written.

Nothing in a run directory is written. Exploratory.

    python redteam/terminal_fable_2026_11_16/probe_episode_route.py \
        --run-dir <run> --out runs_meta/living_channel_dose_2026_11_16/<run>/episode_route.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=32)
    ap.add_argument("--read", type=int, default=4)
    ap.add_argument("--score", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    import torch

    from luthi.episode_store import EpisodeStore
    from luthi.v2 import adaptation_eval as ae
    from luthi.v2 import result_json
    from luthi.v2.living_layer_pc import PredictiveCodingLayer
    from luthi.v2.run_rebuild import rebuild_run, reconstruction
    from scripts.jepa_pilot_driver import _device

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    device = _device()
    run = rebuild_run(args.run_dir, device=device)
    lm = run.loss_module
    model = lm.online_encoder
    windows = run.text_ds.holdout_window_tokens().to(device)
    docs = ae.holdout_documents(windows, n_read=args.read, n_score=args.score,
                                n_docs=args.docs)
    shift = max(1, len(docs) // 2)

    blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, EpisodeStore)]
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, PredictiveCodingLayer)]
    if not blocks or not layers:
        raise RuntimeError(f"{len(blocks)} block stores and {len(layers)} living layers; "
                           "this probe wants both")

    mode = {"phase": None, "block_writes": True, "layer_writes": True}
    log: dict = {}

    def record(kind: str, name: str) -> dict:
        return log.setdefault((mode["phase"], kind, name), {
            "recall_asked": 0, "recall_fired": 0, "best_sims": [], "recalled_slots": [],
            "offered": 0, "offered_salience": [], "bar_when_offered": [],
            "written": 0, "written_slots": [],
        })

    def wrap_block(name: str, store: EpisodeStore) -> None:
        recall, write = store.recall, store.store

        def recall_and_note(context):
            r = record("block", name)
            r["recall_asked"] += 1
            n = int(store.episode_count.item())
            if n:
                sims = store.episode_contexts[:n] @ context
                r["best_sims"].append(float(sims.max().item()))
                slot = int(sims.argmax().item())
            got = recall(context)
            if got is not None:
                r["recall_fired"] += 1
                r["recalled_slots"].append(slot)
            return got

        def store_and_note(context, output, salience):
            r = record("block", name)
            r["offered"] += 1
            r["offered_salience"].append(float(salience))
            n = int(store.episode_count.item())
            full = n >= store.num_episodes
            r["bar_when_offered"].append(
                max(store.salience_threshold,
                    float(store.episode_saliences[:n].min().item()) if full else 0.0))
            before = store.episode_contexts.clone()
            writes = store.store_writes
            write(context, output, salience)
            if store.store_writes != writes:
                r["written"] += 1
                changed = (store.episode_contexts != before).any(dim=1).nonzero().flatten()
                r["written_slots"].extend(int(i) for i in changed)

        store.recall, store.store = recall_and_note, store_and_note

    def wrap_layer(name: str, layer: PredictiveCodingLayer) -> None:
        recall, write = layer._recall_episode, layer._store_episode

        def recall_and_note(context):
            r = record("layer", name)
            r["recall_asked"] += 1
            n = int(layer.episode_count.item())
            if n:
                sims = layer.episode_contexts[:n].to(context.dtype) @ context
                r["best_sims"].append(float(sims.max().item()))
                slot = int(sims.argmax().item())
            got = recall(context)
            if got is not None:
                r["recall_fired"] += 1
                r["recalled_slots"].append(slot)
            return got

        def store_and_note(context, salience, input_pattern):
            if not mode["layer_writes"]:
                return
            r = record("layer", name)
            r["offered"] += 1
            r["offered_salience"].append(float(salience))
            r["bar_when_offered"].append(layer._episode_store_stats()["episode_admission_bar"])
            before = layer.episode_contexts.clone()
            writes = int(layer.episode_writes.item())
            write(context, salience, input_pattern)
            if int(layer.episode_writes.item()) != writes:
                r["written"] += 1
                changed = (layer.episode_contexts != before).any(dim=1).nonzero().flatten()
                r["written_slots"].extend(int(i) for i in changed)

        layer._recall_episode, layer._store_episode = recall_and_note, store_and_note

    def as_loaded() -> list[dict]:
        rows = []
        for name, s in blocks:
            n = int(s.episode_count.item())
            sal = s.episode_saliences[:n]
            rows.append({
                "store": "block", "name": name, "episodes": n, "capacity": s.num_episodes,
                "salience_min": float(sal.min().item()) if n else None,
                "salience_median": float(sal.median().item()) if n else None,
                "salience_max": float(sal.max().item()) if n else None,
                "salience_threshold": s.salience_threshold,
                "recall_similarity_threshold": s.recall_similarity_threshold,
                "blend_factor": s.blend_factor,
            })
        for name, m in layers:
            stats = m._episode_store_stats()
            rows.append({
                "store": "layer", "name": name, "episodes": int(m.episode_count.item()),
                "capacity": m.num_episodes,
                "adaptive_episodes": bool(m.adaptive_episodes),
                "adaptive_recall": bool(m.adaptive_recall),
                "salience_threshold": m.salience_threshold,
                "episode_recall_threshold": m.episode_recall_threshold,
                "episode_blend": float(m.episode_blend),
                "recall_sim_count": int(m.recall_sim_count.item()) if hasattr(m, "recall_sim_count") else None,
                "recall_sim_mean": float(m.recall_sim_mean.item()) if hasattr(m, "recall_sim_mean") else None,
                **{k: stats[k] for k in ("episode_writes", "recall_fires", "episode_salience_floor",
                                         "episode_admission_bar", "episode_context_similarity")},
            })
        return rows

    loaded = as_loaded()
    for name, s in blocks:
        wrap_block(name, s)
    for name, m in layers:
        wrap_layer(name, m)

    prior = [(m.rate_scale, m.homeostatic_decay) for _, m in layers]
    for _, m in layers:
        m.rate_scale, m.homeostatic_decay = 0.0, 0.0
    base = ae._snapshot(model)

    def read(what: torch.Tensor, *, block_writes: bool, layer_writes: bool) -> None:
        mode["block_writes"], mode["layer_writes"] = block_writes, layer_writes
        for _, s in blocks:
            s._plasticity_frozen = not block_writes
        try:
            ae._read(model, what, repeats=1, read_train_mode=False, consolidation=False)
        finally:
            for _, s in blocks:
                s._plasticity_frozen = False
            mode["block_writes"], mode["layer_writes"] = True, True

    def take(phase_read: str | None, phase_score: str, tail, what=None, **kw) -> dict:
        if what is not None:
            mode["phase"] = phase_read
            read(what, **kw)
        mode["phase"] = phase_score
        return ae._score(lm, tail)

    per_doc = []
    try:
        for i, doc in enumerate(docs):
            head, tail = doc[:args.read], doc[args.read:]
            other = docs[(i + shift) % len(docs)][:args.read]
            log.clear()
            scores = {"A": take(None, "score_A", tail)}
            for key, what, kw in (
                ("B", head, {"block_writes": True, "layer_writes": True}),
                ("B_block", head, {"block_writes": True, "layer_writes": False}),
                ("B_layer", head, {"block_writes": False, "layer_writes": True}),
                ("C", other, {"block_writes": True, "layer_writes": True}),
            ):
                scores[key] = take(f"read_{key}", f"score_{key}", tail, what, **kw)
                ae._restore(model, base)
            events = [{"phase": phase, "store": kind, "name": name, **r}
                      for (phase, kind, name), r in log.items()]
            # Did a score recall a slot that the read before it had written?
            for e in events:
                if e["phase"].startswith("score_") and e["phase"] != "score_A":
                    before = log.get((e["phase"].replace("score_", "read_"), e["store"], e["name"]))
                    written = set(before["written_slots"]) if before else set()
                    e["recalled_a_slot_this_read_wrote"] = any(
                        s in written for s in e["recalled_slots"])
            per_doc.append({"doc": i, **scores, "events": events})
            a = scores["A"]["nmse"]
            print(f"doc {i:>2}  A={a:.5f}  " + "  ".join(
                f"{k}-A={scores[k]['nmse'] - a:+.2e}" for k in ("B", "B_block", "B_layer", "C")),
                flush=True)
    finally:
        ae._restore(model, base)
        for (_, m), (scale, decay) in zip(layers, prior):
            m.rate_scale, m.homeostatic_decay = scale, decay

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result_json.dumps({
        "what": "the episode route alone (write 0, pull 0, consolidation held off), store by store",
        "run": {"arm": run.result["arm"], "seed": run.result["seed"],
                "checkpoint": run.checkpoint.name, "global_step": run.global_step},
        "documents": {"n_docs": args.docs, "n_read": args.read, "n_score": args.score,
                      "phase": 0, "control_shift": shift,
                      "holdout_windows": int(windows.shape[0])},
        "stores_as_loaded": loaded,
        "per_doc": per_doc,
        "reconstruction": reconstruction(run.result["config"]),
    }), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
