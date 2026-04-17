"""Generation/inference pipeline for the Living Weight Model.

Load a trained checkpoint and generate text — see what the model has to say.
Supports text-only, multimodal (audio+text), and vision (image+text) models.

The pipeline supports two inference modes:
  - **Static inference**: Model weights are frozen. Standard autoregressive
    generation — useful for benchmarking and deterministic output.
  - **Living inference**: Hebbian self-modification remains active during
    generation. The model's weights change from the experience of producing
    each token. This is the mode that matters for Sanctuary integration —
    the model learns from its own cognition.

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


def load_model_from_checkpoint(
    checkpoint_path: str,
    password: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple:
    """Load a trained model and tokenizer from an encrypted checkpoint.

    Automatically detects model type (text-only, spiking, multimodal)
    from the checkpoint config and constructs the correct architecture.

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

    # Determine model type from config
    is_multimodal = config.get("multimodal", False)
    is_vision = config.get("vision", False)
    is_spiking = config.get("spiking", False)

    # Common model parameters
    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_blocks=config["n_blocks"],
        max_seq_len=config.get("seq_len", 128),
        hebb_rate=config.get("hebb_rate", 0.001),
        error_rate=config.get("error_rate", 0.001),
        homeostatic_decay=config.get("homeostatic_decay", 0.001),
        set_point_adapt_rate=config.get("set_point_adapt_rate", 1e-6),
    )

    if is_multimodal or is_vision:
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
    living: bool = False,
    stream: bool = True,
) -> str:
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
        image: Optional [1, 3, H, W] image tensor for vision models.
        living: If True, keep Hebbian self-modification active during
            generation. The model learns from the experience of producing
            each token.
        stream: If True, print tokens as they're generated.

    Returns:
        Full generated text (prompt + generated tokens).
    """
    device = next(model.parameters()).device
    is_multimodal = hasattr(model, "vision_encoder")

    if living:
        model.train()
        # Keep backward pass off — we want Hebbian updates only,
        # not top-down sweep (which needs loss gradients)
        if hasattr(model, "backward_pass_enabled"):
            model.backward_pass_enabled = False
    else:
        model.eval()

    token_ids = tokenizer.encode(prompt)
    generated_ids = []

    if stream:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    with torch.set_grad_enabled(living):
        for step in range(max_tokens):
            # Sliding window — use only the last max_seq_len tokens
            context = token_ids[-max_seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)

            # Forward pass
            if is_multimodal and image is not None and step == 0:
                # First step: include image context
                logits = model(x, image=image)
            elif is_multimodal:
                logits = model(x)
            else:
                logits = model(x)

            # Living inference: Hebbian updates fire during forward pass
            # in train mode. Optionally apply error-directed updates too.
            if living and hasattr(model, "apply_living_errors"):
                model.apply_living_errors()

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

    return tokenizer.decode(token_ids)


def get_introspection(model: torch.nn.Module) -> dict:
    """Read the model's current internal state — cognitive proprioception.

    Returns a dict of observable living weight states:
        - plasticity: per-block mean and std of plasticity values
        - set_point_drift: how far weights have moved from homeostatic targets
        - spiking: membrane potential stats, spike fractions, refractory states
        - episodes: active episode count and mean activation strength
        - excitability: per-block mean excitability
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

        # Excitability — overall responsiveness
        if hasattr(ffn, "excitability"):
            block_state["excitability_mean"] = ffn.excitability.mean().item()

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
        if "episode_strength_mean" in block:
            parts.append(f"episodes={block['episode_strength_mean']:.3f}")

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
        print("  (Hebbian self-modification active — weights evolve during generation)")
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
                print("  Switched to LIVING inference (Hebbian updates active)")
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
                        help="Enable living inference (Hebbian updates active)")
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
    model_type = "multimodal" if config.get("multimodal") else "spiking" if config.get("spiking") else "base"
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
