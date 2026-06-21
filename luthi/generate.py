"""Generation/inference pipeline for the Living Weight Model.

Load a trained checkpoint and generate text — see what the model has to say.
Supports text-only, multimodal (audio+text), and vision (image+text) models.

The pipeline supports two inference modes:
  - **Static inference**: Model weights are frozen. Standard autoregressive
    generation — useful for benchmarking and deterministic output.
  - **Living inference**: Predictive-coding self-modification (v2) remains
    active during generation. The model's weights change from the experience
    of producing each token. This is the mode that matters for Sanctuary
    integration — the model learns from its own cognition.

Usage:
    # Interactive mode — talk to the model
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi

    # Single prompt
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi \
        --prompt "Once upon a time"

    # With image input (vision model)
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi \
        --image path/to/image.jpg --prompt "Describe this image"

    # Living inference — weights evolve during generation
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi \
        --living

    # Show internal state during generation (introspection)
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi \
        --introspect
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from luthi.checkpoint import load_checkpoint
from luthi.tokenizer import BPETokenizer


_KNOWN_ARCHITECTURES = (
    "v1-base", "v1-spiking", "v1-multimodal",
    "v2-base", "v2-multimodal",
)


def _detect_architecture(config: dict) -> str:
    """Decide which model class a checkpoint config describes.

    Returns one of ``_KNOWN_ARCHITECTURES``. Single decision point for
    the load path -- anything downstream just dispatches on the string.

    Priority:
      1. Explicit ``architecture`` key in the config (canonical going
         forward; trainers should start setting it).
      2. Inference from fingerprints:
         - ``pc_rate`` / ``pred_learning_rate`` present  -> v2 family
         - ``multimodal`` / ``vision`` flag set          -> multimodal
         - ``spiking`` flag set (v1 only)                -> v1-spiking
         - else                                          -> v1-base

    Fails loud on an internally inconsistent config (e.g. a v2
    fingerprint with v1's ``spiking=True``, or an explicit
    ``architecture`` that contradicts the fingerprints). v2 has no
    spiking variant; silently building the wrong architecture would
    mis-shape the model and corrupt downstream training-time wiring.
    """
    explicit = config.get("architecture")
    is_multimodal = bool(config.get("multimodal") or config.get("vision"))
    is_spiking = bool(config.get("spiking"))
    is_v2 = ("pc_rate" in config) or ("pred_learning_rate" in config)

    if is_v2 and is_spiking:
        raise ValueError(
            "Inconsistent checkpoint config: v2 PC fingerprint "
            "(pc_rate / pred_learning_rate) combined with v1 spiking=True. "
            "v2 has no spiking variant; refusing to silently build the "
            "wrong architecture."
        )

    if explicit is not None:
        if explicit not in _KNOWN_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture {explicit!r} in checkpoint config. "
                f"Expected one of {_KNOWN_ARCHITECTURES}."
            )
        if is_v2 and explicit.startswith("v1-"):
            raise ValueError(
                f"Inconsistent checkpoint config: architecture={explicit!r} "
                f"but v2 fingerprint (pc_rate / pred_learning_rate) present."
            )
        if (not is_v2) and explicit.startswith("v2-"):
            raise ValueError(
                f"Inconsistent checkpoint config: architecture={explicit!r} "
                f"but no v2 fingerprint (pc_rate / pred_learning_rate) "
                f"in config."
            )
        if is_multimodal and "multimodal" not in explicit:
            raise ValueError(
                f"Inconsistent checkpoint config: architecture={explicit!r} "
                f"but multimodal/vision flag set in config."
            )
        return explicit

    if is_v2:
        return "v2-multimodal" if is_multimodal else "v2-base"
    if is_multimodal:
        return "v1-multimodal"
    if is_spiking:
        return "v1-spiking"
    return "v1-base"


def load_model_from_checkpoint(
    checkpoint_path: str,
    password: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple:
    """Load a trained model and tokenizer from an encrypted checkpoint.

    Automatically detects model architecture (v1 base / spiking /
    multimodal, or v2 base / multimodal predictive-coding) from the
    checkpoint config via :func:`_detect_architecture` and constructs
    the correct class. Mismatches between config and stored weights
    fail loud at ``load_state_dict`` (strict for v2; v1 multimodal
    keeps its historical ``strict=False`` for encoder-state tolerance).

    Returns:
        (model, tokenizer, config, epoch) tuple.
    """
    ckpt = load_checkpoint(checkpoint_path, password, device)
    config = ckpt["config"]
    epoch = ckpt["epoch"]

    # Restore tokenizer
    if "tokenizer_state" in ckpt and ckpt["tokenizer_state"].get("type") == "bpe":
        tokenizer = BPETokenizer.from_state(ckpt["tokenizer_state"])
    else:
        raise ValueError(
            "Checkpoint does not contain a BPE tokenizer state. "
            "Character-level tokenizer requires the original corpus."
        )

    architecture = _detect_architecture(config)

    if architecture.startswith("v2"):
        # v2 PC family (PredictiveCodingLM / MultimodalPredictiveCodingLM).
        # Disjoint kwarg set from v1: pc_rate / pred_learning_rate /
        # ffn_expansion / num_episodes / episode_blend / mu_pc_*,
        # never the v1 hebb_rate / error_rate / spike_* parameters.
        v2_kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            d_model=config["d_model"],
            n_blocks=config["n_blocks"],
            n_heads=config.get("n_heads", 4),
            ffn_expansion=config.get("ffn_expansion", 1),
            max_seq_len=config.get("seq_len", 128),
            pc_rate=config.get("pc_rate", 0.001),
            pred_learning_rate=config.get("pred_learning_rate", 0.0001),
            homeostatic_decay=config.get("homeostatic_decay", 0.001),
            set_point_adapt_rate=config.get("set_point_adapt_rate", 1e-6),
            num_episodes=config.get("num_episodes", 64),
            episode_blend=config.get("episode_blend", 0.3),
            compressed_episodes=config.get("compressed_episodes", False),
            consolidation_enabled=config.get("consolidation_enabled", False),
            consolidation_style=config.get("consolidation_style", "gradient"),
            consolidation_attractor_passes=config.get(
                "consolidation_attractor_passes", 1,
            ),
            mu_pc_enabled=config.get("mu_pc_enabled", False),
            mu_pc_exponent=config.get("mu_pc_exponent", 0.5),
            # Default to False at load (inference posture, matches the v1
            # paths). Callers that want the top-down sweep active flip
            # it after load. Explicit checkpoint config wins if present.
            backward_pass_enabled=config.get("backward_pass_enabled", False),
        )

        if architecture == "v2-multimodal":
            from luthi.v2.multimodal_model_pc import MultimodalPredictiveCodingLM

            v2_kwargs.update(
                audio_sample_rate=config.get("audio_sample_rate", 16000),
                audio_n_mels=config.get("audio_n_mels", 80),
                audio_hop_length=config.get("audio_hop_length", 160),
                audio_n_fft=config.get("audio_n_fft", 400),
                audio_patch_frames=config.get("audio_patch_frames", 16),
                max_audio_tokens=config.get("max_audio_tokens", 1000),
                vision_image_size=config.get("vision_image_size", 224),
                vision_patch_size=config.get("vision_patch_size", 16),
                max_vision_tokens=config.get("max_vision_tokens", 256),
            )
            model = MultimodalPredictiveCodingLM(**v2_kwargs)
        else:
            from luthi.v2.model_pc import PredictiveCodingLM

            model = PredictiveCodingLM(**v2_kwargs)

        # Strict load -- per the seam-integration plan, mismatched
        # state_dict on v2 must fail loud, never silently skip keys.
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model, tokenizer, config, epoch

    # v1 family below. Shape preserved verbatim from the pre-bridge loader
    # so historical .luthi files keep loading identically.
    is_multimodal = architecture == "v1-multimodal"
    is_vision = bool(config.get("vision", False))
    is_spiking = architecture == "v1-spiking"

    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_blocks=config["n_blocks"],
        max_seq_len=config.get("seq_len", 128),
        hebb_rate=config.get("hebb_rate", 0.001),  # v1 model config
        error_rate=config.get("error_rate", 0.001),
        homeostatic_decay=config.get("homeostatic_decay", 0.001),
        set_point_adapt_rate=config.get("set_point_adapt_rate", 1e-6),
    )

    if is_multimodal:
        from luthi.multimodal_model import MultimodalLuthiLM

        # Add spiking and vision parameters
        model_kwargs.update(
            spike_threshold=config.get("spike_threshold", 1.0),
            membrane_leak=config.get("membrane_leak", 0.1),
            refractory_steps=config.get("refractory_steps", 3),
            delay_steps=config.get("delay_steps", 2),
            backward_pass_enabled=False,
        )
        if is_vision:
            model_kwargs.update(
                vision_image_size=config.get("vision_image_size", 224),
                vision_patch_size=config.get("vision_patch_size", 16),
                max_vision_tokens=config.get("max_vision_tokens", 256),
            )

        model = MultimodalLuthiLM(**model_kwargs)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    elif is_spiking:
        from luthi.model_spiking import SpikingLuthiLM

        model_kwargs.update(
            spike_threshold=config.get("spike_threshold", 1.0),
            membrane_leak=config.get("membrane_leak", 0.1),
            refractory_steps=config.get("refractory_steps", 3),
            delay_steps=config.get("delay_steps", 2),
            backward_pass_enabled=False,
        )
        model = SpikingLuthiLM(**model_kwargs)
        model.load_state_dict(ckpt["model_state_dict"])

    else:
        from luthi.model import LuthiLM

        model = LuthiLM(**model_kwargs)
        model.load_state_dict(ckpt["model_state_dict"])

    return model, tokenizer, config, epoch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
) -> int:
    """Sample the next token from logits with various strategies.

    Args:
        logits: [vocab_size] raw logits from model output.
        temperature: Softmax temperature (lower = more deterministic).
        top_k: If > 0, keep only top K tokens before sampling.
        top_p: If > 0, use nucleus sampling (keep smallest set with cumulative prob >= p).
        repetition_penalty: Penalize tokens that already appeared (> 1.0 = less repetition).
        generated_ids: Previously generated token IDs for repetition penalty.

    Returns:
        Sampled token ID.
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        seen = set(generated_ids)
        for token_id in seen:
            if token_id < logits.shape[0]:
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[0])
        top_k_values, _ = torch.topk(logits, top_k)
        min_top_k = top_k_values[-1]
        logits = torch.where(logits < min_top_k, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[0] = False
        # Shift right so we keep the token that crosses the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    max_seq_len: int = 128,
    image: torch.Tensor | None = None,
    audio_tokens: torch.Tensor | None = None,
    vision_tokens: torch.Tensor | None = None,
    living: bool = False,
    stream: bool = True,
    return_state: bool = False,
):
    """Generate text autoregressively from a prompt.

    Args:
        model: Loaded LuthiLM, SpikingLuthiLM, or MultimodalLuthiLM.
        tokenizer: BPETokenizer (or CharTokenizer) with encode/decode.
        prompt: Text prompt to start generation from.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0.1 = focused, 1.5 = creative).
        top_k: Top-K sampling (0 = disabled).
        top_p: Nucleus sampling threshold (0 = disabled).
        repetition_penalty: Penalize repeated tokens (1.0 = no penalty).
        max_seq_len: Model's maximum context window.
        image: Optional [1, 3, H, W] image tensor for vision models. Will
            be passed through the vision encoder on the first forward call.
        audio_tokens: Optional [batch, n_audio_tokens, d_model] pre-encoded
            audio tokens. Skips the audio encoder. Included in the sequence
            on the first forward call only — subsequent autoregressive
            steps run text-only because the model is stateless and the
            sensory context only conditions the first generated token.
        vision_tokens: Optional [batch, n_vision_tokens, d_model] pre-encoded
            vision tokens. Same first-step semantics as ``audio_tokens``.
            If both ``image`` and ``vision_tokens`` are provided,
            ``vision_tokens`` takes precedence (no need to re-encode).
        living: If True, keep living self-modification active during
            generation (predictive-coding updates for v2 models, Hebbian
            for v1). The model learns from the experience of producing
            each token.
        stream: If True, print tokens as they're generated.
        return_state: If True, also captures the step-0 encoder result
            and returns ``(text, s_t)`` where ``s_t = compute_s_t(
            encode_result["per_modality"]["text"])``. This is the
            Phase 4a training-seam path -- Sanctuary consumes the
            same encoder pass that produced the generation logits
            rather than running a separate encode_state call.
            Requires a v2 multimodal-PC substrate (the only family
            with the ``encode()`` API the seam needs); raises
            ``AttributeError`` eagerly otherwise per 4.8's 2026-06-16
            review (silent ``(text, None)`` would reopen the silent-
            degradation foot-gun the F4 fix just closed).

    Returns:
        Full generated text (prompt + generated tokens). When
        ``return_state`` is True, returns ``(text, s_t)`` instead.
    """
    # Eager capability check: ``return_state`` requires a v2 multimodal-PC
    # substrate. We check by the presence of ``encode`` (the seam's
    # contract surface) plus a multimodal marker -- v1 multimodal has
    # ``vision_encoder`` but no ``encode``; v2 text-only has neither.
    # Raise at entry rather than mid-generation so the error message
    # names the actual config mismatch.
    if return_state:
        if not hasattr(model, "encode"):
            raise AttributeError(
                "generate_text(return_state=True) requires a v2 multimodal-PC "
                "substrate with an encode() method "
                "(MultimodalPredictiveCodingLM). "
                f"Got {type(model).__name__}; v1 substrates and v2 text-only "
                "do not expose the encoder API the training seam consumes. "
                "Callers wanting 'state if available' should branch on "
                "capability before this call, not decode a None after it."
            )

    device = next(model.parameters()).device
    is_multimodal = hasattr(model, "vision_encoder")

    # Capture original model mode + backward_pass setting before mutating,
    # so the finally clause below restores it. Without this, calling
    # generate_text from a training loop or the Sanctuary integration
    # would leave the model in train()/eval() and backward_pass off after
    # return — corrupting downstream state silently.
    original_training = model.training
    original_backward_pass_enabled = getattr(
        model, "backward_pass_enabled", None
    )

    try:
        if living:
            model.train()
            # Keep backward pass off — we want living self-modification
            # only (PC for v2, Hebbian for v1), not top-down sweep
            # (which needs loss gradients).
            if hasattr(model, "backward_pass_enabled"):
                model.backward_pass_enabled = False
        else:
            model.eval()

        token_ids = tokenizer.encode(prompt)
        generated_ids = []

        if stream:
            sys.stdout.write(prompt)
            sys.stdout.flush()

        has_sensory = is_multimodal and (
            image is not None
            or audio_tokens is not None
            or vision_tokens is not None
        )

        # Detect whether the model supports KV cache. Both v1 LuthiLM and
        # v2 PredictiveCodingLM accept kv_caches/return_kv_caches as of
        # 2026-05-10 (audit follow-up). MultimodalLuthiLM doesn't yet —
        # falls back to the legacy recompute-each-token path. Detection
        # is via signature inspection so future KV-cache-aware models
        # (e.g., a future multimodal v2) get the fast path automatically.
        import inspect
        try:
            sig = inspect.signature(model.forward)
            supports_kv_cache = (
                "kv_caches" in sig.parameters
                and "return_kv_caches" in sig.parameters
                and not has_sensory  # legacy multimodal path needs its own
                                     # kwargs and isn't compatible with the
                                     # cache-aware text path
            )
        except (TypeError, ValueError):
            supports_kv_cache = False

        kv_caches: list | None = None  # populated after step 0 when supported
        # Phase 4a capture: step-0's encode_result is the input to the
        # canonical compute_s_t for the seam path. None until the first
        # step under a return_state=True call sets it.
        captured_encode_result: dict | None = None

        with torch.set_grad_enabled(living):
            for step in range(max_tokens):
                # Forward pass — sensory context only on the first step.
                if has_sensory and step == 0:
                    # Multimodal first step: pass the full prompt + sensory
                    # tokens. No KV cache support yet for this path; sensory
                    # context conditions the first generated token only.
                    context = token_ids[-max_seq_len:]
                    x = torch.tensor([context], dtype=torch.long, device=device)
                    forward_kwargs: dict = {}
                    if vision_tokens is not None:
                        forward_kwargs["vision_tokens"] = vision_tokens
                    elif image is not None:
                        forward_kwargs["image"] = image
                    if audio_tokens is not None:
                        forward_kwargs["audio_tokens"] = audio_tokens
                    if return_state:
                        logits, captured_encode_result = model(
                            x, return_encode_result=True, **forward_kwargs,
                        )
                    else:
                        logits = model(x, **forward_kwargs)
                elif supports_kv_cache:
                    # KV-cache fast path. Step 0: full prompt, initialize
                    # cache. Step 1+: just the new token, extend cache.
                    if step == 0:
                        context = token_ids[-max_seq_len:]
                        x = torch.tensor([context], dtype=torch.long, device=device)
                        logits, kv_caches = model(
                            x, return_kv_caches=True,
                        )
                    else:
                        # Feed only the most recently generated token; the
                        # attention sees the rest via the cache.
                        x = torch.tensor(
                            [[token_ids[-1]]], dtype=torch.long, device=device,
                        )
                        logits, kv_caches = model(
                            x, kv_caches=kv_caches, return_kv_caches=True,
                        )
                else:
                    # Legacy recompute-each-step path (sliding window).
                    context = token_ids[-max_seq_len:]
                    x = torch.tensor([context], dtype=torch.long, device=device)
                    if return_state and step == 0:
                        logits, captured_encode_result = model(
                            x, return_encode_result=True,
                        )
                    else:
                        logits = model(x)

                # Living inference: self-modification updates fire during forward pass
                # in train mode. Note (audit 2026-05-11): we used to also
                # call model.apply_living_errors() here, but that's
                # error-directed learning which requires a loss.backward()
                # to populate gradients first. During generation there's
                # no loss, so the call was a no-op — dead code. Removed.

                # Sample next token
                next_logits = logits[0, -1, :].clone()
                next_id = sample_next_token(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_ids=generated_ids,
                )

                token_ids.append(next_id)
                generated_ids.append(next_id)

                if stream:
                    token_text = tokenizer.decode([next_id])
                    sys.stdout.write(token_text)
                    sys.stdout.flush()

        if stream:
            sys.stdout.write("\n")
            sys.stdout.flush()

        text = tokenizer.decode(token_ids)
        if return_state:
            # The capability check at the top guarantees the substrate
            # was multimodal-PC at entry; the capture above ran at step
            # 0 of one of the two paths multimodal-PC takes. If we got
            # here with no captured result, generation never executed
            # a step -- max_tokens=0 is the only way that happens, and
            # in that case there's no state to surface. Raise loudly
            # rather than synthesize one; the caller chose return_state.
            if captured_encode_result is None:
                raise RuntimeError(
                    "generate_text(return_state=True) ran zero steps "
                    "(max_tokens=0); no encode_result was captured. "
                    "Pass max_tokens >= 1 if you need state."
                )
            from luthi.seam_types import GenerationState
            from luthi.v2.m9.s_t import compute_s_t
            # Pool over the FULL concatenated multimodal sequence
            # (encode_result["latents"]), matching the training-side
            # convention: M9Trainer pools over raw["online_context_latents"]
            # which is the encoder output over the full multimodal context
            # (vision + audio + text), not text-only. Mismatched inputs to
            # compute_s_t would be exactly the drift the no-drift test in
            # tests/test_no_inline_s_t_pool.py exists to catch.
            full_latents = captured_encode_result["latents"]
            state = GenerationState(
                s_t=compute_s_t(full_latents),
                ctx_latents=full_latents.detach(),
            )
            return text, state
        return text
    finally:
        # Restore original mode and backward_pass setting so any caller
        # (Sanctuary integration, training loop, generation script) sees
        # the model in the state they left it.
        model.train(original_training)
        if (
            original_backward_pass_enabled is not None
            and hasattr(model, "backward_pass_enabled")
        ):
            model.backward_pass_enabled = original_backward_pass_enabled


def get_introspection(model: torch.nn.Module) -> dict:
    """Read the model's current internal state — cognitive proprioception.

    All per-block fields are hasattr-gated, so the dict shape adapts to
    whatever the loaded substrate exposes. v1 (LivingLayerV6, spiking
    variants) populates the original spiking/Hebbian set; v2
    (PredictiveCodingLayer) populates a PC-specific set.

    v1 fields (populate when present):
        - plasticity_mean/std/min/max
        - set_point_drift
        - excitability_mean/min/max + excitability_acc_mean
        - membrane_mean/std/max
        - spike_fraction
        - refractory_fraction
        - episode_count, episode_salience_mean/max

    v2 fields (populate when present):
        - plasticity_mean/std/min/max (shared with v1)
        - set_point_drift (shared with v1)
        - error_acc_mean / error_acc_max — running per-output prediction
          error magnitude. The most direct signal of "this layer is
          surprised right now"; the design's natural turbo trigger.
        - pred_frob — Frobenius norm of the prediction matrix. Climbs as
          PC layers accumulate structure during training.
        - precision_mean — running 1/error² EMA across the in-dim.
    """
    state = {"blocks": []}

    if not hasattr(model, "blocks"):
        return state

    for i, block in enumerate(model.blocks):
        block_state = {"block": i}

        ffn = getattr(block, "living_ffn", None)
        if ffn is None:
            state["blocks"].append(block_state)
            continue

        # Plasticity — how willing each dimension is to change
        if hasattr(ffn, "plasticity"):
            p = ffn.plasticity
            block_state["plasticity_mean"] = p.mean().item()
            block_state["plasticity_std"] = p.std().item()
            block_state["plasticity_min"] = p.min().item()
            block_state["plasticity_max"] = p.max().item()

        # Set point drift — how far from homeostatic equilibrium
        if hasattr(ffn, "set_point") and hasattr(ffn, "weight"):
            drift = (ffn.weight.data - ffn.set_point).abs().mean().item()
            block_state["set_point_drift"] = drift

        # Excitability — overall responsiveness.
        # LivingLayerV6 stores `excitability_acc` (the running accumulator);
        # the effective excitability is mapped through sigmoid by
        # `_excitability_factor()`. Report both: the effective value (what
        # the layer actually uses) and the raw accumulator (useful for
        # diagnosing saturation when the sigmoid pegs at min or max).
        if hasattr(ffn, "_excitability_factor"):
            exc = ffn._excitability_factor()
            block_state["excitability_mean"] = exc.mean().item()
            block_state["excitability_min"] = exc.min().item()
            block_state["excitability_max"] = exc.max().item()
        if hasattr(ffn, "excitability_acc"):
            block_state["excitability_acc_mean"] = (
                ffn.excitability_acc.mean().item()
            )

        # Spiking dynamics
        if hasattr(ffn, "membrane_potential"):
            mp = ffn.membrane_potential
            block_state["membrane_mean"] = mp.mean().item()
            block_state["membrane_std"] = mp.std().item()
            block_state["membrane_max"] = mp.max().item()

        if hasattr(ffn, "spike_mask"):
            sm = ffn.spike_mask
            block_state["spike_fraction"] = sm.float().mean().item()

        if hasattr(ffn, "refractory_counter"):
            rc = ffn.refractory_counter
            block_state["refractory_fraction"] = (rc > 0).float().mean().item()

        # v2 PC-specific signals. PredictiveCodingLayer exposes
        # error_acc (per-output running prediction-error magnitude),
        # prediction (per-block prediction matrix), and precision (per
        # in-dim EMA of 1/error²). None of these exist on v1 spiking
        # variants; all are hasattr-gated so v1 introspection is
        # unaffected.
        if hasattr(ffn, "error_acc"):
            ea = ffn.error_acc
            block_state["error_acc_mean"] = ea.mean().item()
            block_state["error_acc_max"] = ea.max().item()

        if hasattr(ffn, "prediction"):
            pred = ffn.prediction
            block_state["pred_frob"] = pred.norm().item()

        if hasattr(ffn, "precision"):
            prec = ffn.precision
            block_state["precision_mean"] = prec.mean().item()

        # Episode store
        ep = getattr(block, "episode_store", None)
        if ep is not None and hasattr(ep, "episode_count"):
            n = ep.episode_count.item()
            block_state["episode_count"] = n
            if n > 0 and hasattr(ep, "episode_saliences"):
                es = ep.episode_saliences[:n]
                block_state["episode_salience_mean"] = es.mean().item()
                block_state["episode_salience_max"] = es.max().item()

        state["blocks"].append(block_state)

    return state


def format_introspection(state: dict) -> str:
    """Format introspection state as human-readable text."""
    lines = []
    for block in state.get("blocks", []):
        i = block["block"]
        parts = [f"  Block {i}:"]

        if "plasticity_mean" in block:
            parts.append(
                f"plasticity={block['plasticity_mean']:.4f}"
                f"\u00b1{block['plasticity_std']:.4f}"
            )
        if "set_point_drift" in block:
            parts.append(f"drift={block['set_point_drift']:.6f}")
        if "excitability_mean" in block:
            parts.append(f"excitability={block['excitability_mean']:.3f}")
        if "spike_fraction" in block:
            parts.append(f"spiking={block['spike_fraction']:.3f}")
        if "membrane_mean" in block:
            parts.append(f"membrane={block['membrane_mean']:.3f}")
        # Audit 2026-05-11 fix: previous code checked the stale key
        # `episode_strength_mean` while the collector at line 454 stores
        # `episode_salience_mean`. The mismatch dropped episode data
        # silently from the introspection output.
        if "episode_salience_mean" in block:
            parts.append(f"episodes={block['episode_salience_mean']:.3f}")

        lines.append(" | ".join(parts))

    return "\n".join(lines) if lines else "  (no living state available)"


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for the vision encoder.

    Returns:
        [1, 3, H, W] normalized image tensor.
    """
    try:
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Image loading requires torchvision and Pillow. "
            "Install with: pip install torchvision Pillow"
        )

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def interactive_session(
    model: torch.nn.Module,
    tokenizer,
    config: dict,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    living: bool = False,
    introspect: bool = False,
    image_path: str | None = None,
):
    """Run an interactive prompt-response session.

    Type a prompt, see the model's response. Type 'quit' or Ctrl+C to exit.
    """
    device = next(model.parameters()).device
    max_seq_len = config.get("seq_len", 128)
    image_size = config.get("vision_image_size", 224)

    # Pre-load image if provided
    image = None
    if image_path:
        image = load_image(image_path, image_size).to(device)
        print(f"  Image loaded: {image_path}")

    mode_label = "LIVING" if living else "STATIC"
    print(f"\n  Mode: {mode_label} inference")
    if living:
        print("  (Living self-modification active — weights evolve during generation)")
    if introspect:
        print("  (Introspection active — internal state shown after each response)")
    print(f"  Temperature: {temperature} | Top-K: {top_k or 'off'} | Top-P: {top_p or 'off'}")
    print(f"  Max tokens: {max_tokens} | Context window: {max_seq_len}")
    print()
    print("  Type your prompt and press Enter. Type 'quit' to exit.")
    print("  Commands: /temp <n>, /topk <n>, /topp <n>, /tokens <n>,")
    print("            /living, /static, /introspect, /image <path>, /state")
    print()

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue

        if prompt.lower() == "quit":
            print("Exiting.")
            break

        # Handle commands
        if prompt.startswith("/"):
            parts = prompt.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/temp":
                temperature = float(arg)
                print(f"  Temperature set to {temperature}")
            elif cmd == "/topk":
                top_k = int(arg)
                print(f"  Top-K set to {top_k}")
            elif cmd == "/topp":
                top_p = float(arg)
                print(f"  Top-P set to {top_p}")
            elif cmd == "/tokens":
                max_tokens = int(arg)
                print(f"  Max tokens set to {max_tokens}")
            elif cmd == "/living":
                living = True
                print("  Switched to LIVING inference (living updates active)")
            elif cmd == "/static":
                living = False
                print("  Switched to STATIC inference (weights frozen)")
            elif cmd == "/introspect":
                introspect = not introspect
                print(f"  Introspection {'ON' if introspect else 'OFF'}")
            elif cmd == "/image":
                if arg:
                    try:
                        image = load_image(arg, image_size).to(device)
                        print(f"  Image loaded: {arg}")
                    except Exception as e:
                        print(f"  Error loading image: {e}")
                else:
                    image = None
                    print("  Image cleared")
            elif cmd == "/state":
                state = get_introspection(model)
                print(format_introspection(state))
            else:
                print(f"  Unknown command: {cmd}")
            continue

        # Show pre-generation state if introspecting
        if introspect:
            pre_state = get_introspection(model)

        # Generate
        print()
        sys.stdout.write("Model> ")
        sys.stdout.flush()

        t0 = time.time()
        output = generate_text(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_seq_len=max_seq_len,
            image=image,
            living=living,
            stream=True,
        )
        elapsed = time.time() - t0

        # Token count and speed
        generated_count = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
        tokens_per_sec = generated_count / max(elapsed, 0.001)
        print(f"  [{generated_count} tokens, {tokens_per_sec:.1f} tok/s, {elapsed:.1f}s]")

        # Post-generation introspection
        if introspect:
            post_state = get_introspection(model)
            print("\n  Internal state:")
            print(format_introspection(post_state))

            # Show drift if living mode
            if living and pre_state.get("blocks") and post_state.get("blocks"):
                print("\n  Changes from generation:")
                for pre_b, post_b in zip(pre_state["blocks"], post_state["blocks"]):
                    changes = []
                    if "plasticity_mean" in pre_b and "plasticity_mean" in post_b:
                        delta = post_b["plasticity_mean"] - pre_b["plasticity_mean"]
                        if abs(delta) > 1e-8:
                            changes.append(f"plasticity {delta:+.6f}")
                    if "set_point_drift" in pre_b and "set_point_drift" in post_b:
                        delta = post_b["set_point_drift"] - pre_b["set_point_drift"]
                        if abs(delta) > 1e-8:
                            changes.append(f"drift {delta:+.6f}")
                    if "membrane_mean" in pre_b and "membrane_mean" in post_b:
                        delta = post_b["membrane_mean"] - pre_b["membrane_mean"]
                        if abs(delta) > 1e-8:
                            changes.append(f"membrane {delta:+.6f}")
                    if changes:
                        print(f"    Block {pre_b['block']}: {', '.join(changes)}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Living Weight Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive session:
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi

  Single prompt:
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi \\
        --prompt "The nature of consciousness"

  Living inference with introspection:
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi \\
        --living --introspect

  Vision + text:
    python -m luthi.generate --checkpoint E:/runs/vision/checkpoint.luthi \\
        --image photo.jpg --prompt "What do you see?"
        """,
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .luthi checkpoint file")
    parser.add_argument("--checkpoint_password", type=str, default=None,
                        help="Decryption password (or set LUTHI_CHECKPOINT_KEY)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (if omitted, enters interactive mode)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file (for vision models)")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Nucleus sampling threshold (0 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (1.0 = none, default: 1.2)")
    parser.add_argument("--living", action="store_true", default=False,
                        help="Enable living inference (living updates active)")
    parser.add_argument("--introspect", action="store_true", default=False,
                        help="Show internal state during generation")

    args = parser.parse_args()

    # Device selection
    try:
        import torch_directml
        device = torch_directml.device()
        device_name = "DirectML (AMD GPU)"
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        else:
            device = torch.device("cpu")
            device_name = "CPU"

    print(f"\n{'='*60}")
    print(f"  Luthi Living Weight Model — Generation Pipeline")
    print(f"{'='*60}")
    print(f"  Device: {device_name}")

    # Load model
    print(f"  Loading checkpoint: {args.checkpoint}")
    t0 = time.time()
    model, tokenizer, config, epoch = load_model_from_checkpoint(
        args.checkpoint, args.checkpoint_password, device,
    )
    model = model.to(device)
    load_time = time.time() - t0

    params = model.total_parameters()
    model_type = _detect_architecture(config)
    modalities = []
    if config.get("vision"):
        modalities.append("vision")
    if config.get("multimodal"):
        modalities.append("audio")
    modalities.append("text")

    print(f"  Model: {config['d_model']}d, {config['n_blocks']} blocks, {model_type}")
    print(f"  Modalities: {' + '.join(modalities)}")
    print(f"  Epoch: {epoch}")
    print(f"  Vocabulary: {tokenizer.vocab_size} BPE tokens")
    print(f"  Parameters: {params['trainable']:,} trainable + {params['living_buffers']:,} living")
    print(f"  Loaded in {load_time:.1f}s")
    print(f"{'='*60}")

    if args.prompt:
        # Single-shot generation
        image = None
        if args.image:
            image_size = config.get("vision_image_size", 224)
            image = load_image(args.image, image_size).to(device)
            print(f"  Image: {args.image}")

        print()
        output = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_seq_len=config.get("seq_len", 128),
            image=image,
            living=args.living,
            stream=True,
        )

        if args.introspect:
            print()
            state = get_introspection(model)
            print("Internal state:")
            print(format_introspection(state))
    else:
        # Interactive mode
        interactive_session(
            model,
            tokenizer,
            config,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            living=args.living,
            introspect=args.introspect,
            image_path=args.image,
        )


if __name__ == "__main__":
    main()
