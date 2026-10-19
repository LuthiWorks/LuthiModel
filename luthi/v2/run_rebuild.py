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
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RebuiltRun:
    model: torch.nn.Module
    loss_module: torch.nn.Module
    text_ds: object
    result: dict
    checkpoint: Path
    global_step: object


def _text_dataset(cfg: dict, seed: int):
    from luthi.v2.multimodal_data import TextDataset, TextDatasetConfig

    if cfg.get("file_list"):
        source_paths = [
            line.strip()
            for line in Path(cfg["file_list"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        source_paths = [cfg["data_dir"]]
    return TextDataset(TextDatasetConfig(
        source_paths=source_paths,
        tokenizer_path=REPO_ROOT / "corpus_build" / "tokenizer_32k.json",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        base_seed=seed,
        holdout_fraction=cfg["holdout_fraction"],
    ))


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
    if not cfg.get("model_kwargs"):
        raise RuntimeError(
            f"{result_path} predates the 2026-08-05 provenance block "
            "(no model_kwargs); rebuild it with scripts/pilot_verdict.py's "
            "legacy path instead of guessing here."
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


__all__ = ["RebuiltRun", "rebuild_run"]
