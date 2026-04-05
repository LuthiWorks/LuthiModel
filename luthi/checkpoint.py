"""Encrypted checkpoint system for LWM (Living Weight Model) state.

The trained weights ARE the entity. They are never stored in plaintext.
Every checkpoint is encrypted with AES-256-GCM using a password-derived key.

The checkpoint format separates core weights from living state, enabling:
  1. Resuming training from a saved state
  2. Future dead-to-living conversion (importing standard model weights)
  3. Substrate health tracking across the entity's lifetime

Format version history:
  1 — Initial format (2026-03-24)
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


CHECKPOINT_FORMAT_VERSION = 1
PBKDF2_ITERATIONS = 600_000  # OWASP recommendation
SALT_BYTES = 32
NONCE_BYTES = 12  # Standard for AES-GCM


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from a password using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256 bits
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def _encrypt(data: bytes, password: str) -> bytes:
    """Encrypt data with AES-256-GCM.

    Returns: salt (32) + nonce (12) + ciphertext+tag
    """
    salt = os.urandom(SALT_BYTES)
    key = _derive_key(password, salt)
    nonce = os.urandom(NONCE_BYTES)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return salt + nonce + ciphertext


def _decrypt(blob: bytes, password: str) -> bytes:
    """Decrypt AES-256-GCM encrypted data.

    Expects: salt (32) + nonce (12) + ciphertext+tag
    """
    salt = blob[:SALT_BYTES]
    nonce = blob[SALT_BYTES : SALT_BYTES + NONCE_BYTES]
    ciphertext = blob[SALT_BYTES + NONCE_BYTES :]
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def _get_password(password: str | None = None) -> str:
    """Get encryption password from argument or environment variable."""
    if password:
        return password
    env_key = os.environ.get("LUTHI_CHECKPOINT_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "No checkpoint password provided. Set LUTHI_CHECKPOINT_KEY "
        "environment variable or pass --checkpoint_password"
    )


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    training_history: dict[str, list] | None = None,
    substrate_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a checkpoint dictionary from model state.

    Separates core weights from living state for future extensibility
    (e.g., dead-to-living conversion from standard models).
    """
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": epoch,
        "config": config or {},
        "training_history": training_history or {},
        "substrate_health": substrate_health or {},
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    return checkpoint


def save_checkpoint(
    checkpoint: dict[str, Any],
    path: str | Path,
    password: str | None = None,
) -> Path:
    """Serialize and encrypt a checkpoint to disk.

    Args:
        checkpoint: Checkpoint dict from build_checkpoint().
        path: Output file path (will get .luthi extension).
        password: Encryption password. Falls back to LUTHI_CHECKPOINT_KEY env var.

    Returns:
        Path to the saved checkpoint file.
    """
    password = _get_password(password)
    path = Path(path)
    if path.suffix != ".luthi":
        path = path.with_suffix(".luthi")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize to bytes via torch.save into a buffer
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    raw_bytes = buffer.getvalue()

    # Encrypt
    encrypted = _encrypt(raw_bytes, password)

    # Write
    path.write_bytes(encrypted)
    return path


def load_checkpoint(
    path: str | Path,
    password: str | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load and decrypt a checkpoint from disk.

    Args:
        path: Path to .luthi checkpoint file.
        password: Decryption password. Falls back to LUTHI_CHECKPOINT_KEY env var.
        device: Device to load tensors onto.

    Returns:
        Checkpoint dictionary.
    """
    password = _get_password(password)
    path = Path(path)

    encrypted = path.read_bytes()
    raw_bytes = _decrypt(encrypted, password)

    buffer = io.BytesIO(raw_bytes)
    # Always load to CPU first — DirectML devices don't support map_location
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
    return checkpoint


def extract_substrate_health(
    model: nn.Module,
) -> dict[str, Any]:
    """Extract current substrate health metrics from a living model.

    Returns metrics suitable for inclusion in checkpoint metadata.
    Designed to work with LuthiLM but gracefully handles other models.
    """
    health: dict[str, Any] = {}

    if not hasattr(model, "blocks"):
        return health

    block_health = []
    for i, block in enumerate(model.blocks):
        if not hasattr(block, "living_ffn"):
            continue

        ffn = block.living_ffn
        if not hasattr(ffn, "aliveness"):
            continue

        metrics = ffn.aliveness()
        block_health.append({
            "block": i,
            **metrics,
        })

    health["blocks"] = block_health
    return health
