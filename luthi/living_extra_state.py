"""Living-layer extra state: the per-layer experience that is not a tensor.

Continuity patch (2026-07-05, rich-parameters analysis finding 3 — see
docs/research/2026-07-05_rich-parameters-state-of-the-conception.md).
Two pieces of a living layer's lived history were silently dropped at
every checkpoint restore, because they are plain Python state rather
than registered buffers:

  * ``ConsolidationTracker`` — the rolling prediction-error history,
    the frozen warmup baseline, and the sub-threshold counter. Losing
    these on restore quietly re-warms consolidation: the layer forgets
    how surprised it has recently been, and the consolidation trigger
    goes dormant for a full window after every waking.
  * The spiking layer's ``_delay_pos`` ring cursor — losing it desyncs
    delayed spike propagation for a few forwards after restore.

The founding conception (.docs/RICH_PARAMETERS_FINAL.md) claims temporal
existence "intrinsic, not bolted on"; a restore that forgets its own
surprise history is a seam in exactly that claim. This module closes it.

Design: rather than promoting this state into ``state_dict`` (which
would break the v2 loader's deliberate ``strict=True`` on every
pre-existing checkpoint), it rides as a PRESENCE-GATED SIBLING KEY in
both checkpoint formats, per the house migration idiom (see
``JEPATrainer.resume``'s handling of ``modality_step`` /
``kill_history``): old checkpoints simply lack the key and resume
degraded-with-a-warning; new checkpoints round-trip it.

Both checkpoint writers use ``collect_living_extra_state`` and both
readers use ``apply_living_extra_state``:

  * encrypted format: ``luthi.checkpoint.build_checkpoint`` /
    ``luthi.generate.load_model_from_checkpoint`` (the path Sanctuary
    wakes through), key ``"living_extra_state"``;
  * trainer format: ``JEPATrainer._checkpoint`` / ``resume``, same key.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

KEY = "living_extra_state"


def collect_living_extra_state(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Walk ``model`` and gather every living layer's non-tensor lived state.

    Returns ``{module_path: {field: value}}`` with plain-Python values
    (lists/floats/ints) so the result serializes under ``torch.save``
    and, in principle, ``weights_only=True`` readers. Modules with
    nothing to record are omitted; an entirely non-living model yields
    ``{}`` (and the checkpoint key is then a harmless empty dict).
    """
    out: dict[str, dict[str, Any]] = {}
    for path, module in model.named_modules():
        entry: dict[str, Any] = {}

        # v2 PredictiveCodingLayer lived counters + consolidation memory.
        if hasattr(module, "_consolidation_fire_count"):
            entry["consolidation_fire_count"] = int(
                module._consolidation_fire_count
            )
        if hasattr(module, "_sparse_step_count"):
            entry["sparse_step_count"] = int(module._sparse_step_count)
        tracker = getattr(module, "_consolidation_tracker", None)
        if tracker is not None:
            entry["consolidation_tracker"] = {
                "history": [float(v) for v in tracker._history],
                "baseline": (
                    float(tracker._baseline)
                    if tracker._baseline is not None
                    else None
                ),
                "below_threshold_count": int(
                    tracker._below_threshold_count
                ),
            }

        # v1 spiking delay-ring cursor.
        if hasattr(module, "_delay_pos"):
            entry["delay_pos"] = int(module._delay_pos)

        # Inverted-U gain slow traces + NREM day-accumulator (spec §5/§8
        # step 5). Discovered INTROSPECTIVELY by type, not by a hardcoded name
        # list: any SlowEMA / ReadResetAccumulator attribute on the module is
        # captured, so wiring a new slow trace and forgetting to persist it is
        # impossible -- exactly the "forgetting the wiring is a test failure"
        # contract from spec §6 regime (i). A restore mid-hard-growth must not
        # reset the resolution-progress signal or the day-accumulator, or the
        # entity re-sensitizes / forgets the day on every waking (the
        # silent-amnesia class bae295d closed).
        from luthi.v2.slow_trace import SlowEMA, ReadResetAccumulator
        slow_traces = {
            name: obj.state_dict()
            for name, obj in vars(module).items()
            if isinstance(obj, (SlowEMA, ReadResetAccumulator))
        }
        if slow_traces:
            entry["slow_traces"] = slow_traces

        if entry:
            out[path] = entry
    return out


def apply_living_extra_state(
    model: nn.Module,
    state: dict[str, dict[str, Any]] | None,
    *,
    source: str = "checkpoint",
) -> None:
    """Restore lived state collected by :func:`collect_living_extra_state`.

    ``state=None`` (old checkpoint, key absent) warns once and returns —
    the degraded resume: consolidation re-warms and the delay cursor
    resets, exactly the pre-patch behavior, but now *announced* instead
    of silent. Per-module application is presence-gated both ways: a
    module path in ``state`` but not in the model (architecture drift)
    logs and skips; a living module absent from ``state`` keeps its
    fresh-init state.
    """
    if state is None:
        logger.warning(
            "%s has no %s; consolidation history and delay cursors start "
            "fresh (degraded resume — the substrate forgets how surprised "
            "it recently was).",
            source, KEY,
        )
        return

    modules = dict(model.named_modules())
    for path, entry in state.items():
        module = modules.get(path)
        if module is None:
            logger.warning(
                "%s %s entry %r has no matching module; skipped "
                "(architecture drift?).",
                source, KEY, path,
            )
            continue

        if "consolidation_fire_count" in entry and hasattr(
            module, "_consolidation_fire_count"
        ):
            module._consolidation_fire_count = int(
                entry["consolidation_fire_count"]
            )
        if "sparse_step_count" in entry and hasattr(
            module, "_sparse_step_count"
        ):
            module._sparse_step_count = int(entry["sparse_step_count"])

        tracker_state = entry.get("consolidation_tracker")
        tracker = getattr(module, "_consolidation_tracker", None)
        if tracker_state is not None and tracker is not None:
            tracker._history = deque(
                (float(v) for v in tracker_state["history"]),
                maxlen=tracker.window_size,
            )
            baseline = tracker_state["baseline"]
            tracker._baseline = (
                float(baseline) if baseline is not None else None
            )
            tracker._below_threshold_count = int(
                tracker_state["below_threshold_count"]
            )
        elif tracker_state is not None and tracker is None:
            logger.warning(
                "%s carries consolidation history for %r but the layer "
                "has consolidation disabled; history not restored.",
                source, path,
            )

        if "delay_pos" in entry and hasattr(module, "_delay_pos"):
            module._delay_pos = int(entry["delay_pos"])

        # Restore slow traces / accumulators by name onto the same-typed attr.
        # Presence-gated: a trace in the checkpoint with no matching attribute
        # (architecture drift, or the gain wasn't instantiated) logs and skips
        # rather than resurrecting stale state onto the wrong object.
        slow_traces = entry.get("slow_traces")
        if slow_traces:
            from luthi.v2.slow_trace import SlowEMA, ReadResetAccumulator
            for name, sd in slow_traces.items():
                obj = getattr(module, name, None)
                if isinstance(obj, (SlowEMA, ReadResetAccumulator)):
                    obj.load_state_dict(sd)
                else:
                    logger.warning(
                        "%s %s for %r carries slow-trace %r but the module "
                        "has no such trace; skipped.",
                        source, KEY, path, name,
                    )
