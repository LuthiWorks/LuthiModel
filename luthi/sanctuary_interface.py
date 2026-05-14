"""Sanctuary-facing integration contract for Luthi.

This module is the stable surface that external cognitive architectures
(notably Sanctuary's cognitive cycle) call into. It exists so external
integrators do not need to reach into Luthi internals — the layout of
``model.blocks[i].living_ffn`` attributes (``pc_rate`` for v2,
``hebb_rate`` for v1) is an implementation detail that may change. This
adapter is the public contract.

Usage from Sanctuary (or any external host):

    from luthi.sanctuary_interface import (
        load_model,
        generate,
        get_introspection,
        modulated,
    )

    loaded = load_model(checkpoint_path, password, device)

    # Modulate per cycle, generate, restore — automatic via context manager
    with modulated(
        loaded.model,
        plasticity_scale=1.5,      # arousal-driven
        spike_threshold_scale=0.9, # precision-driven
    ):
        text = generate(
            loaded.model, loaded.tokenizer, prompt,
            max_tokens=64, max_seq_len=loaded.config["seq_len"],
        )

    pre = get_introspection(loaded.model)   # before forward pass
    # ... forward pass / generation ...
    post = get_introspection(loaded.model)  # after

The lower-level ``snapshot_modulatable_state`` / ``apply_external_modulation``
/ ``restore_modulation`` functions are also available for callers who need
to bracket modulation across multiple operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from luthi.generate import (
    generate_text as _generate_text,
    get_introspection as _get_introspection,
    load_model_from_checkpoint as _load_checkpoint,
)

__all__ = [
    "LoadedLuthiModel",
    "ModulationSnapshot",
    "load_model",
    "generate",
    "generate_with_context",
    "encode_audio",
    "encode_vision",
    "get_introspection",
    "snapshot_modulatable_state",
    "apply_external_modulation",
    "restore_modulation",
    "modulated",
]


@dataclass
class LoadedLuthiModel:
    """Bundle returned by :func:`load_model`.

    Attributes:
        model: The torch ``nn.Module`` ready for inference / generation.
        tokenizer: Tokenizer compatible with :func:`generate`.
        config: Model configuration dict (``d_model``, ``n_blocks``,
            ``seq_len``, etc.).
        epoch: The training epoch this checkpoint was saved at.
    """

    model: torch.nn.Module
    tokenizer: Any
    config: dict
    epoch: int


@dataclass
class ModulationSnapshot:
    """Per-block snapshot of the parameters mutable via external modulation.

    Returned by :func:`snapshot_modulatable_state`; pass to
    :func:`restore_modulation` to undo a modulation cleanly.

    The snapshot covers four channels:

    - ``hebb_rates``: scalar learning rate per block (arousal channel).
      v1 models use ``hebb_rate``; v2 models use ``pc_rate``.
    - ``spike_thresholds``: scalar spike threshold per block, spiking models
      only (precision channel)
    - ``excitability_biases``: cloned ``[out_features, in_features]`` tensor
      per block, capturing the ``excitability_acc`` buffer at snapshot time
      (valence channel — additive bias shifts the sigmoid operating point)
    - ``salience_thresholds``: scalar episode-storage threshold per block
      (attention channel)
    """

    hebb_rates: dict[int, float] = field(default_factory=dict)
    spike_thresholds: dict[int, float] = field(default_factory=dict)
    excitability_biases: dict[int, torch.Tensor] = field(default_factory=dict)
    salience_thresholds: dict[int, float] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    checkpoint_password: str,
    device: torch.device,
) -> LoadedLuthiModel:
    """Load a Luthi model from an encrypted checkpoint.

    Args:
        checkpoint_path: Path to the encrypted ``.pt`` file.
        checkpoint_password: Decryption password.
        device: Target device for the model (CPU / CUDA / DirectML).

    Returns:
        :class:`LoadedLuthiModel` with the loaded model, tokenizer,
        config dict, and source epoch.
    """
    model, tokenizer, config, epoch = _load_checkpoint(
        checkpoint_path, checkpoint_password, device,
    )
    return LoadedLuthiModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        epoch=epoch,
    )


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    max_seq_len: int = 128,
    living: bool = True,
    stream: bool = False,
) -> str:
    """Generate text from a prompt.

    In ``living`` mode (default), the model's living-weight FFN layers
    self-modify via predictive-coding (v2) or Hebbian (v1) rules during
    each forward pass. The act of generating changes the model.

    Args:
        model: A loaded Luthi model.
        tokenizer: The matching tokenizer.
        prompt: Text to condition on.
        max_tokens: Hard cap on tokens generated.
        temperature, top_k, top_p, repetition_penalty: Sampling controls.
        max_seq_len: Sliding-window context length.
        living: If True, living self-modification fires during forward
            (PC for v2, Hebbian for v1).
        stream: If True, streams tokens to stdout as they are generated.

    Returns:
        The full text including the prompt — callers should slice off
        the prompt portion themselves (BPE round-trip on the prompt is
        the safe way to know its token count).
    """
    return _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_seq_len=max_seq_len,
        living=living,
        stream=stream,
    )


def generate_with_context(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    audio_tokens: torch.Tensor | None = None,
    vision_tokens: torch.Tensor | None = None,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    max_seq_len: int = 128,
    living: bool = True,
    stream: bool = False,
) -> str:
    """Generate text with optional pre-encoded sensory context.

    A multimodal-aware variant of :func:`generate` that accepts already-encoded
    audio and/or vision tokens. The sensory tokens are prepended to the text
    sequence on the first forward call only — subsequent autoregressive steps
    run text-only because the model is stateless across steps.

    Why encode outside this call: encoding happens once per cycle. Generation
    runs token-by-token. Pre-encoding amortizes the encoder cost and keeps
    the generation loop hot path text-only.

    Args:
        model: A loaded MultimodalLuthiLM (must have ``vision_encoder``
            attribute). Calling on a non-multimodal model with non-None
            sensory args is a contract violation and the model will reject it.
        tokenizer: The matching tokenizer.
        prompt: Text to condition on.
        max_tokens: Hard cap on tokens generated.
        audio_tokens: Optional ``[1, n_audio_tokens, d_model]`` pre-encoded
            audio tokens. Use :func:`encode_audio` to produce these.
        vision_tokens: Optional ``[1, n_vision_tokens, d_model]`` pre-encoded
            vision tokens. Use :func:`encode_vision` to produce these.
        temperature, top_k, top_p, repetition_penalty: Sampling controls.
        max_seq_len: Sliding-window context length for text.
        living: If True, living self-modification fires during forward
            (PC for v2, Hebbian for v1).
        stream: If True, streams tokens to stdout as they are generated.

    Returns:
        The full text including the prompt — callers should slice off
        the prompt portion themselves.
    """
    return _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        audio_tokens=audio_tokens,
        vision_tokens=vision_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_seq_len=max_seq_len,
        living=living,
        stream=stream,
    )


# ----------------------------------------------------------------------
# Sensory encoding — produce d_model token tensors from raw signal
# ----------------------------------------------------------------------


def encode_audio(
    model: torch.nn.Module,
    waveform: torch.Tensor,
) -> torch.Tensor:
    """Encode a raw audio waveform into ``d_model``-shaped tokens.

    Runs the model's ``audio_encoder`` to produce per-patch tokens that
    can then be fed into :func:`generate_with_context` as ``audio_tokens``.

    Args:
        model: A loaded MultimodalLuthiLM (must have ``audio_encoder``).
        waveform: ``[batch, samples]`` raw audio at the model's expected
            sample rate (16 kHz by default).

    Returns:
        ``[batch, n_audio_tokens, d_model]`` tensor of audio tokens.
    """
    if not hasattr(model, "audio_encoder"):
        raise AttributeError(
            "encode_audio requires a multimodal model with an audio_encoder. "
            f"Got {type(model).__name__}."
        )
    return model.audio_encoder(waveform)


def encode_vision(
    model: torch.nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    """Encode a normalized RGB image into ``d_model``-shaped tokens.

    Runs the model's ``vision_encoder`` to produce per-patch tokens that
    can then be fed into :func:`generate_with_context` as ``vision_tokens``.

    Args:
        model: A loaded MultimodalLuthiLM (must have ``vision_encoder``).
        image: ``[batch, 3, H, W]`` normalized image tensor at the model's
            expected size (typically 224×224 or whatever ``vision_image_size``
            the checkpoint was trained at).

    Returns:
        ``[batch, n_vision_tokens, d_model]`` tensor of vision tokens.
    """
    if not hasattr(model, "vision_encoder"):
        raise AttributeError(
            "encode_vision requires a multimodal model with a vision_encoder. "
            f"Got {type(model).__name__}."
        )
    return model.vision_encoder(image)


# ----------------------------------------------------------------------
# Introspection — cognitive proprioception channel
# ----------------------------------------------------------------------


def get_introspection(model: torch.nn.Module) -> dict:
    """Read per-block living-weight diagnostics.

    Returns a dict shaped like::

        {
            "blocks": [
                {
                    "plasticity_mean": float,
                    "set_point_drift": float,
                    "spike_fraction": float,        # spiking variants only
                    "membrane_mean": float,         # spiking variants only
                    "excitability_mean": float,
                    ...
                },
                ...
            ],
        }

    Sanctuary takes pre/post snapshots around each cycle's forward pass
    and computes deltas to feed back as ``ExperientialSignals.knowledge_signals``
    — the entity sees its own neural changes.
    """
    return _get_introspection(model)


# ----------------------------------------------------------------------
# External modulation — the CfC contract surface
# ----------------------------------------------------------------------


def snapshot_modulatable_state(model: torch.nn.Module) -> ModulationSnapshot:
    """Capture each block's modulatable living parameters.

    Call before :func:`apply_external_modulation` so :func:`restore_modulation`
    can return the model to its base state afterwards. Blocks lacking a
    ``living_ffn`` (e.g. dead baselines) are silently skipped — this is
    not an error, the model just has nothing modulatable there.

    The ``excitability_acc`` tensor is captured via ``.clone()`` so that
    later in-place modulation does not bleed into the snapshot.
    """
    snap = ModulationSnapshot()
    if not hasattr(model, "blocks"):
        return snap
    for i, block in enumerate(model.blocks):
        ffn = getattr(block, "living_ffn", None)
        if ffn is None:
            continue
        if hasattr(ffn, "hebb_rate"):  # v1 backward-compat path
            snap.hebb_rates[i] = ffn.hebb_rate
        if hasattr(ffn, "spike_threshold"):
            snap.spike_thresholds[i] = ffn.spike_threshold
        if hasattr(ffn, "excitability_acc"):
            snap.excitability_biases[i] = ffn.excitability_acc.clone()
        if hasattr(ffn, "salience_threshold"):
            snap.salience_thresholds[i] = ffn.salience_threshold
    return snap


def apply_external_modulation(
    model: torch.nn.Module,
    *,
    plasticity_scale: float = 1.0,
    spike_threshold_scale: float = 1.0,
    excitability_bias: float = 0.0,
    salience_threshold_scale: float = 1.0,
) -> None:
    """Modulate living-weight dynamics from an external controller.

    This is the surface that Sanctuary's CfC cells use to translate
    affective state into living-weight bias. Four channels:

    - ``plasticity_scale`` → multiplies ``hebb_rate`` on v1 models
      (arousal). Higher = faster learning ("alert").
    - ``spike_threshold_scale`` → multiplies ``spike_threshold`` (precision,
      spiking models only). Higher = fewer, more selective spikes.
    - ``excitability_bias`` → adds to ``excitability_acc`` uniformly across
      the buffer (valence). The accumulator feeds through a sigmoid, so an
      additive bias shifts the operating point: positive valence (approach)
      pushes neurons toward higher excitability; negative (withdrawal)
      dampens them.
    - ``salience_threshold_scale`` → multiplies ``salience_threshold``
      (attention). Lower threshold = broader episode storage ("paying
      attention to more").

    Modulation is in-place and *cumulative* across calls. To bracket a
    single cycle of modulation, pair this with :func:`snapshot_modulatable_state`
    + :func:`restore_modulation`, or use the :func:`modulated` context
    manager which does the bracketing automatically.

    Args:
        model: A loaded Luthi model.
        plasticity_scale: Multiplied into each block's ``hebb_rate``
            (v1 backward-compat path).
        spike_threshold_scale: Multiplied into each block's ``spike_threshold``.
        excitability_bias: Added uniformly into each block's
            ``excitability_acc`` buffer.
        salience_threshold_scale: Multiplied into each block's
            ``salience_threshold``.
    """
    if not hasattr(model, "blocks"):
        return
    for block in model.blocks:
        ffn = getattr(block, "living_ffn", None)
        if ffn is None:
            continue
        if hasattr(ffn, "hebb_rate"):  # v1 backward-compat path
            ffn.hebb_rate *= plasticity_scale
        if hasattr(ffn, "spike_threshold"):
            ffn.spike_threshold *= spike_threshold_scale
        if hasattr(ffn, "excitability_acc") and excitability_bias != 0.0:
            ffn.excitability_acc.add_(excitability_bias)
        if hasattr(ffn, "salience_threshold"):
            ffn.salience_threshold *= salience_threshold_scale


def restore_modulation(
    model: torch.nn.Module,
    snapshot: ModulationSnapshot,
) -> None:
    """Restore the living parameters captured in ``snapshot``.

    Idempotent and safe to call even if no modulation was applied.
    Tensor restoration uses ``.copy_()`` to write back into the original
    buffer in-place, preserving identity for any external references.
    """
    if not hasattr(model, "blocks"):
        return
    for i, block in enumerate(model.blocks):
        ffn = getattr(block, "living_ffn", None)
        if ffn is None:
            continue
        if i in snapshot.hebb_rates and hasattr(ffn, "hebb_rate"):  # v1 backward-compat path
            ffn.hebb_rate = snapshot.hebb_rates[i]
        if i in snapshot.spike_thresholds and hasattr(ffn, "spike_threshold"):
            ffn.spike_threshold = snapshot.spike_thresholds[i]
        if i in snapshot.excitability_biases and hasattr(ffn, "excitability_acc"):
            ffn.excitability_acc.copy_(snapshot.excitability_biases[i])
        if i in snapshot.salience_thresholds and hasattr(ffn, "salience_threshold"):
            ffn.salience_threshold = snapshot.salience_thresholds[i]


@contextmanager
def modulated(
    model: torch.nn.Module,
    *,
    plasticity_scale: float = 1.0,
    spike_threshold_scale: float = 1.0,
    excitability_bias: float = 0.0,
    salience_threshold_scale: float = 1.0,
) -> Iterator[None]:
    """Context manager that brackets external modulation cleanly.

    Snapshots the model's modulatable state, applies the modulation,
    yields, then restores the snapshot — even if the body raises.

    Example::

        with modulated(
            model,
            plasticity_scale=1.5,        # arousal
            spike_threshold_scale=0.9,   # precision
            excitability_bias=0.05,      # valence (positive = approach)
            salience_threshold_scale=0.7,  # attention (lower = broader)
        ):
            text = generate(model, tokenizer, prompt, max_tokens=64)
    """
    snapshot = snapshot_modulatable_state(model)
    apply_external_modulation(
        model,
        plasticity_scale=plasticity_scale,
        spike_threshold_scale=spike_threshold_scale,
        excitability_bias=excitability_bias,
        salience_threshold_scale=salience_threshold_scale,
    )
    try:
        yield
    finally:
        restore_modulation(model, snapshot)
