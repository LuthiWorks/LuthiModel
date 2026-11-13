"""Rebuild a trained JEPA pilot run from its own provenance (2026-10-19).

One shared, read-only path for the offline instruments
(``scripts/adaptation_eval.py``, ``scripts/living_write_probe.py``): the
model is constructed from the ``model_kwargs`` the run RECORDED in
``pilot_result.json`` (the 2026-08-05 mechanism-provenance block), the loss
module from the run's recorded loss settings, and the checkpoint's online
encoder, living extra state, predictor and projection heads are loaded.
Never from the live ARM_CONFIGS registry, which can drift from what a run
was actually trained with (Opus 5 audit, 2026-08-13).

Nothing here writes to the run directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

CORPUS_MOVED_NOTE = "recorded path is gone; same path under this repo"


@dataclass
class RebuiltRun:
    model: torch.nn.Module
    loss_module: torch.nn.Module
    text_ds: object
    result: dict
    checkpoint: Path
    global_step: object


def _where_the_corpus_is(recorded: str, cfg: dict, key: str) -> Path:
    """A run records its corpus by absolute path. The repo has moved since
    some runs were made (2026-07-23), and their recorded paths point at a
    place that is gone. When the recorded path does not exist and it lies
    inside a ``corpus_build`` directory, the same path under this repo's
    ``corpus_build`` is used, and the change is written into the config
    beside the original. Anything else is refused, as before.

    That the file is the same file is not checked here. For
    ``gutenberg_4x_filelist.txt`` it was checked on 2026-11-13: one commit
    ever (2026-07-18), and byte-identical to the copy in the pre-migration
    backup."""
    path = Path(recorded)
    if path.exists():
        return path
    parts = PureWindowsPath(recorded).parts if "\\" in recorded else path.parts
    if "corpus_build" in parts:
        here = REPO_ROOT.joinpath(*parts[parts.index("corpus_build"):])
        if here.exists():
            cfg[f"{key}_recorded"] = recorded
            cfg[key] = str(here)
            cfg[f"{key}_source"] = CORPUS_MOVED_NOTE
            logger.warning(
                "the run's recorded %s (%s) no longer exists; using the same "
                "path under this repo: %s", key, recorded, here,
            )
            return here
    raise FileNotFoundError(
        f"the run's recorded {key} does not exist: {recorded}. It is not "
        "under a corpus_build directory that this repo also has, so there "
        "is nothing to rebuild its data from."
    )


def _text_dataset(cfg: dict, seed: int):
    from luthi.v2.multimodal_data import TextDataset, TextDatasetConfig

    if cfg.get("file_list"):
        file_list = _where_the_corpus_is(cfg["file_list"], cfg, "file_list")
        source_paths = [
            line.strip()
            for line in file_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        source_paths = [str(_where_the_corpus_is(cfg["data_dir"], cfg, "data_dir"))]
    return TextDataset(TextDatasetConfig(
        source_paths=source_paths,
        tokenizer_path=REPO_ROOT / "corpus_build" / "tokenizer_32k.json",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        base_seed=seed,
        holdout_fraction=cfg["holdout_fraction"],
    ))


# Runs made before 2026-08-05 did not record what their model was built with.
# For three arms that was checked, on 2026-11-13, against the driver as it
# stood when they ran (the dead runs under f33838b of 2026-07-19 and cc30226
# of 2026-07-20; the living runs under 13e5cd25 of 2026-07-21 and 0230cc9f of
# 2026-07-26): the driver's base settings and each arm's registry entry were
# then what is written here, and no default of the model or the living layer
# has changed or been removed since. One default of the loss has: see
# PRE_PROVENANCE_SIGREG_PROJECTION below. The values are frozen here, not
# read from the live registry, so a later change to the registry cannot
# quietly rebuild an old run as a different model. The checkpoint must still
# fit exactly (see _load_older_online_state for the one allowance), which
# catches any difference of shape. Any other arm is refused.
PRE_PROVENANCE_NOTE = (
    "reconstructed: checked 2026-11-13 against commits f33838b, cc30226, "
    "13e5cd25 and 0230cc9f, and against a run trained by the code of 0230cc9f"
)
_LIVING_V5_4X_D4 = {
    "backward_pass_enabled": True, "consolidation_enabled": True,
    "episode_recall_threshold": 0.7, "learning_gain_enabled": True,
    "mu_pc_enabled": True, "mu_pc_exponent": 0.25, "n_blocks": 4,
    "relative_trust": True,
}
PRE_PROVENANCE_ARMS: dict[str, dict] = {
    "dead_4x": {"dead_ffn": True},
    "living_v5_4x_d4": _LIVING_V5_4X_D4,
    "living_v5_4x_d4_rerun": _LIVING_V5_4X_D4,
}


def _pre_provenance_model_kwargs(result: dict, result_path: Path) -> dict:
    arm = result["arm"]
    if arm not in PRE_PROVENANCE_ARMS:
        raise RuntimeError(
            f"{result_path} predates the 2026-08-05 provenance block "
            f"(no model_kwargs), and arm {arm!r} is not one whose earlier "
            "configuration has been checked against history "
            f"({', '.join(sorted(PRE_PROVENANCE_ARMS))}). Check it the same "
            "way before adding it; do not guess here."
        )
    cfg = result["config"]
    kwargs = dict(
        d_model=result["d_model"], n_blocks=cfg["n_blocks"], n_heads=4,
        ffn_expansion=1, max_seq_len=cfg["seq_len"], backward_pass_enabled=False,
    )
    kwargs.update(PRE_PROVENANCE_ARMS[arm])
    return kwargs


# What the loss and the trainer were given for the same runs. Neither rides in
# model_kwargs, and neither was recorded then.
#
# - The projection head in front of SIGReg was a linear layer and a BatchNorm
#   until commit 8ec9d07 (2026-07-29), which made a bare linear layer the
#   default. Found on 2026-11-13 by training a run with the code of
#   2026-07-26 and loading it here: the heads did not fit.
# - The living arm trained with the plasticity taper, so it ended its run
#   with the living rates at the taper's floor (0.2). No checkpoint carries
#   that scale, so a rebuilt model's living layers stand at 1.0. It is
#   written into the config so an instrument can measure at either.
PRE_PROVENANCE_SIGREG_PROJECTION = "linear_bn"
TAPER_FLOOR = 0.2   # TaperConfig's floor; no driver has ever set another
PRE_PROVENANCE_TAPER: dict[str, bool] = {
    "dead_4x": False, "living_v5_4x_d4": True, "living_v5_4x_d4_rerun": True,
}

# State the living layer has stored in its checkpoints since 2026-07-27 to
# 2026-07-29, which runs made before then do not have: the episode store's
# admission and age bookkeeping (adaptive_episodes), adaptive recall, the
# drive modes, and the homeostatic band. Each belongs to a mechanism that is
# off unless asked for, and none of the runs above asked. With them off the
# layer reads none of these values, so leaving them as a new layer has them
# changes nothing the layer computes. That is checked two ways:
# tests/test_rebuild_pre_provenance_runs.py scrambles them and requires every
# forward to come out the same to the bit, and on 2026-11-13 a run trained
# with the code of 2026-07-26 was loaded both by that code and by this, and
# the two computed the same numbers (docs/research/2026-11-13_rebuilding-july-runs.md).
LIVING_STATE_SINCE_2026_07_27 = frozenset({
    "episode_steps", "episode_step_counter", "episode_writes", "last_write_step",
    "salience_window", "salience_window_pos", "salience_window_filled",
    "salience_level", "salience_drift", "salience_dev",
    "recall_sim_mean", "recall_sim_var", "recall_sim_count", "recall_fires",
    "error_rms", "drive_ref", "drive_ref_drift", "drive_dev", "drive_calls",
    "drive_gain", "drive_fire_count", "drive_gain_sum",
    "act_mean", "act_var", "act_count", "band_boost_rows", "band_damp_rows",
})


def trained_with_taper(result: dict) -> bool | None:
    """Whether a run trained with the plasticity taper, so that it ended with
    its living rates at the taper's floor. None when the run did not record
    it and is not one of the runs checked against history."""
    cfg = result.get("config", {})
    if "taper" in cfg:
        return bool(cfg["taper"])
    if not cfg.get("model_kwargs"):
        return PRE_PROVENANCE_TAPER.get(result.get("arm"))
    return None


# Not the recorded path itself. A path from before the repo moved lies in a
# user's home folder, and a result is committed to a public repo: the first
# results written with this field (2026-11-13) put that path in ten files.
# The run's own pilot_result.json, on the machine that made it, still has it.
RECONSTRUCTION_KEYS = (
    "model_kwargs_source", "living_state_absent_at_training",
    "file_list_source", "data_dir_source",
)


def reconstruction(cfg: dict) -> dict:
    """What ``rebuild_run`` had to supply or change to rebuild this run, for
    an instrument to write into its result. Empty when the run recorded
    everything and nothing was moved. Call it after ``rebuild_run``, on the
    config that call returned."""
    marks = {key: cfg[key] for key in RECONSTRUCTION_KEYS if key in cfg}
    if "model_kwargs_source" in marks:
        # Supplied with the settings, from the same history.
        marks["sigreg_projection"] = cfg["sigreg_projection"]
        marks["taper"] = cfg["taper"]
    return marks


def _reads_none_of_it(layer) -> bool:
    """True when every mechanism that reads LIVING_STATE_SINCE_2026_07_27 is
    off in this layer."""
    return (
        not layer.adaptive_episodes and not layer.adaptive_recall
        and layer.drive_mode == "raw" and not layer.drive_normalize
        and not layer.homeostatic_band_enabled
    )


def _load_older_online_state(model, online_state: dict, ckpt: Path) -> list[str]:
    """Load a pre-provenance run's encoder. Everything must fit exactly, as
    in a strict load, with one allowance: a living layer may lack the state
    named in LIVING_STATE_SINCE_2026_07_27, and only when the layer has every
    mechanism that reads it switched off. Returns the names that were absent."""
    from luthi.v2.living_layer_pc import PredictiveCodingLayer

    missing, unexpected = model.load_state_dict(online_state, strict=False)
    modules = dict(model.named_modules())
    refused = []
    for key in missing:
        path, _, name = key.rpartition(".")
        layer = modules.get(path)
        if not (
            isinstance(layer, PredictiveCodingLayer)
            and name in LIVING_STATE_SINCE_2026_07_27
            and _reads_none_of_it(layer)
        ):
            refused.append(key)
    if refused or unexpected:
        raise RuntimeError(
            f"{ckpt} does not fit the model rebuilt for it. Missing, and not "
            f"state an older living layer may lack: {refused}. Not expected: "
            f"{list(unexpected)}."
        )
    return sorted({key.rpartition(".")[2] for key in missing})


def rebuild_run(
    run_dir: str | Path,
    *,
    device: torch.device,
    ckpt_name: str | None = None,
    with_dataset: bool = True,
) -> RebuiltRun:
    """Load ``run_dir`` (a jepa_pilot run directory) at its latest (or the
    named) checkpoint. Raises loudly on missing provenance -- a rebuild that
    guesses its configuration reports numbers for a different model."""
    from luthi.living_extra_state import apply_living_extra_state
    from luthi.v2.jepa_loss import JEPALoss
    from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

    run_dir = Path(run_dir)
    result_path = run_dir / "pilot_result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"{result_path} missing: no recorded provenance to rebuild from"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = result["config"]
    pre_provenance = not cfg.get("model_kwargs")
    if pre_provenance:
        cfg["model_kwargs"] = _pre_provenance_model_kwargs(result, result_path)
        cfg["model_kwargs_source"] = PRE_PROVENANCE_NOTE
        cfg.setdefault("sigreg_projection", PRE_PROVENANCE_SIGREG_PROJECTION)
        cfg.setdefault("taper", PRE_PROVENANCE_TAPER[result["arm"]])
        logger.warning(
            "%s has no recorded model_kwargs; arm %r was rebuilt from the "
            "configuration checked against git history on 2026-11-13.",
            result_path, result["arm"],
        )
    model_kwargs = dict(cfg["model_kwargs"])

    text_ds = _text_dataset(cfg, result["seed"]) if with_dataset else None
    if text_ds is not None:
        model_kwargs["vocab_size"] = text_ds.vocab_size()
    model = MultimodalPredictiveCodingLM(**model_kwargs).to(device)

    loss_kwargs = dict(
        sigreg_lambd=cfg.get("sigreg_lambd", 0.1),
        sigreg_projection=cfg.get("sigreg_projection", "linear"),
        interior_sigreg_alpha=cfg.get("interior_sigreg_alpha", 0.0),
        w_ntp=cfg.get("w_ntp", 0.0),
        visreg_lambda=cfg.get("visreg_lambda", 0.0),
        visreg_num_proj=cfg.get("visreg_num_proj", 1024),
    )
    loss_module = JEPALoss(online_encoder=model, **loss_kwargs).to(device)

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {run_dir / 'checkpoints'}")
    ckpt = (run_dir / "checkpoints" / ckpt_name) if ckpt_name else ckpts[-1]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if pre_provenance:
        absent = _load_older_online_state(model, state["online_state_dict"], ckpt)
        if absent:
            cfg["living_state_absent_at_training"] = absent
            logger.warning(
                "%s was written before the living layer stored %d of today's "
                "values; they are left as a new layer has them, and no "
                "mechanism that reads them is on.", ckpt, len(absent),
            )
    else:
        model.load_state_dict(state["online_state_dict"])
    apply_living_extra_state(
        model, state.get("living_extra_state"), source=f"rebuild_run {ckpt.name}",
    )
    if "predictor_state_dict" in state:
        loss_module.predictor.load_state_dict(state["predictor_state_dict"])
    if "projection_heads_state_dict" in state:
        loss_module.projection_heads.load_state_dict(
            state["projection_heads_state_dict"]
        )
    model.eval()
    loss_module.eval()
    return RebuiltRun(
        model=model, loss_module=loss_module, text_ds=text_ds, result=result,
        checkpoint=ckpt, global_step=state.get("global_step"),
    )


__all__ = [
    "RebuiltRun", "TAPER_FLOOR", "rebuild_run", "reconstruction", "trained_with_taper",
]
