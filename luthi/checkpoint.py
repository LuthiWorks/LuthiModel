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
GCM_TAG_BYTES = 16  # AES-GCM authentication tag, appended by cryptography lib

# Envelope format v2: chunked encryption to handle checkpoints exceeding
# AES-GCM's 2^31 - 1 byte plaintext limit per nonce. Plaintext is split
# into CHUNK_SIZE chunks; each chunk gets its own random nonce.
# Format: magic(4) | salt(32) | chunk_count(8 LE u64) |
#         [chunk_size(8 LE u64) | nonce(12) | ciphertext_with_tag(chunk_size bytes)]*
# v1 envelopes (salt(32) | nonce(12) | ciphertext_with_tag) remain loadable;
# detection is via the magic at offset 0.
ENVELOPE_MAGIC_V2 = b"LTH2"
CHUNK_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB, well under AES-GCM's 2.15 GB ceiling


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
    """Encrypt data with AES-256-GCM, chunked (envelope v2).

    Splits plaintext into CHUNK_SIZE chunks, encrypts each with its own
    random nonce, and packs into the v2 envelope. Chunking lifts the
    AES-GCM 2^31-1 byte plaintext limit, which a single-call encrypt
    hits at ~2.15 GB.

    Each chunk's GCM tag is bound to additional authenticated data
    (AAD): magic || salt || chunk_count || chunk_index. This makes
    chunk reordering, count truncation, and cross-file splicing fail
    loud at GCM verification instead of silently producing corrupted
    plaintext. AAD is not stored — the reader reconstructs it from the
    header fields already on disk.

    Returns: magic(4) + salt(32) + chunk_count(8) + [chunk_size(8) +
             nonce(12) + ciphertext_with_tag(chunk_size)]*
    """
    salt = os.urandom(SALT_BYTES)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)

    n = len(data)
    # range(0, 0, CHUNK_SIZE) is empty; chunk_count = 0 is the empty-data
    # case. For nonzero len that's an exact multiple of CHUNK_SIZE, the
    # last index is (k-1)*CHUNK_SIZE and the last chunk is exactly
    # CHUNK_SIZE bytes — no zero-length trailing chunk.
    chunk_offsets = list(range(0, n, CHUNK_SIZE))
    chunk_count = len(chunk_offsets)

    aad_prefix = ENVELOPE_MAGIC_V2 + salt + chunk_count.to_bytes(8, "little")

    parts: list[bytes] = [
        ENVELOPE_MAGIC_V2,
        salt,
        chunk_count.to_bytes(8, "little"),
    ]
    for i, offset in enumerate(chunk_offsets):
        chunk = data[offset : offset + CHUNK_SIZE]
        nonce = os.urandom(NONCE_BYTES)
        aad = aad_prefix + i.to_bytes(8, "little")
        ciphertext = aesgcm.encrypt(nonce, chunk, aad)
        # ciphertext length = len(chunk) + GCM_TAG_BYTES
        parts.append(len(ciphertext).to_bytes(8, "little"))
        parts.append(nonce)
        parts.append(ciphertext)
    return b"".join(parts)


def _decrypt(blob: bytes, password: str) -> bytes:
    """Decrypt an AES-256-GCM checkpoint envelope (v1 or v2).

    Version detected by the magic at offset 0. v2 envelopes parse chunked;
    v1 envelopes (no magic) fall through to the legacy single-blob path.
    Truncated v2 envelopes fail explicitly at the integrity check —
    never silently return partial data.
    """
    if blob[:len(ENVELOPE_MAGIC_V2)] == ENVELOPE_MAGIC_V2:
        return _decrypt_v2(blob, password)
    return _decrypt_v1(blob, password)


def _decrypt_v1(blob: bytes, password: str) -> bytes:
    """Legacy single-blob envelope: salt(32) + nonce(12) + ciphertext_with_tag."""
    salt = blob[:SALT_BYTES]
    nonce = blob[SALT_BYTES : SALT_BYTES + NONCE_BYTES]
    ciphertext = blob[SALT_BYTES + NONCE_BYTES :]
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def _decrypt_v2(blob: bytes, password: str) -> bytes:
    """Chunked envelope: magic + salt + chunk_count + [chunk_size + nonce + ct]*."""
    pos = len(ENVELOPE_MAGIC_V2)
    salt = blob[pos : pos + SALT_BYTES]
    if len(salt) != SALT_BYTES:
        raise ValueError(
            f"checkpoint envelope v2: truncated header (salt incomplete)"
        )
    pos += SALT_BYTES

    if len(blob) < pos + 8:
        raise ValueError(
            f"checkpoint envelope v2: truncated header (chunk_count missing)"
        )
    chunk_count = int.from_bytes(blob[pos : pos + 8], "little")
    pos += 8

    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)

    # AAD binds each chunk's GCM tag to: format magic, key-derivation
    # salt, declared chunk count, and chunk index. Reconstructed here
    # from header fields, not stored on disk. A reordered, dropped, or
    # cross-file-spliced chunk produces a different AAD on decrypt and
    # raises InvalidTag — silent structural corruption fails loud.
    aad_prefix = ENVELOPE_MAGIC_V2 + salt + chunk_count.to_bytes(8, "little")

    out: list[bytes] = []
    for i in range(chunk_count):
        if len(blob) < pos + 8:
            raise ValueError(
                f"checkpoint envelope v2: truncated at chunk {i}/"
                f"{chunk_count} (chunk_size header missing)"
            )
        chunk_size = int.from_bytes(blob[pos : pos + 8], "little")
        pos += 8

        if len(blob) < pos + NONCE_BYTES:
            raise ValueError(
                f"checkpoint envelope v2: truncated at chunk {i}/"
                f"{chunk_count} (nonce missing)"
            )
        nonce = blob[pos : pos + NONCE_BYTES]
        pos += NONCE_BYTES

        if len(blob) < pos + chunk_size:
            raise ValueError(
                f"checkpoint envelope v2: truncated at chunk {i}/"
                f"{chunk_count} (declared chunk_size={chunk_size}, "
                f"have {len(blob) - pos} bytes remaining)"
            )
        ciphertext = blob[pos : pos + chunk_size]
        pos += chunk_size

        aad = aad_prefix + i.to_bytes(8, "little")
        out.append(aesgcm.decrypt(nonce, ciphertext, aad))

    if pos != len(blob):
        raise ValueError(
            f"checkpoint envelope v2: trailing {len(blob) - pos} bytes "
            f"after declared {chunk_count} chunks"
        )
    return b"".join(out)


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

    # Living-layer lived state that is not a tensor (consolidation
    # history/baseline, spiking delay cursor). Sibling key, not part of
    # model_state_dict, so old checkpoints keep loading strict=True and
    # readers restore it presence-gated (continuity patch 2026-07-05).
    from luthi.living_extra_state import collect_living_extra_state
    checkpoint["living_extra_state"] = collect_living_extra_state(model)

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
    trusted: bool = False,
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
    # Always load to CPU first — DirectML devices don't support map_location.
    #
    # `trusted` controls the pickle security posture:
    # - `trusted=False` (default): `weights_only=True`. Strict — only
    #   torch/numpy primitives on the allowlist are unpickled. If the
    #   checkpoint format uses anything else (e.g., DirectML's save format
    #   uses `_rebuild_device_tensor_from_numpy`), the load raises
    #   `pickle.UnpicklingError` and the caller sees a loud failure.
    # - `trusted=True`: caller explicitly affirms the file is trusted —
    #   typically because the AES-256-GCM decryption gate above is the
    #   actual security boundary and they hold the key. Falls back to the
    #   permissive `weights_only=False`. This is an opt-in, not a silent
    #   fallback. The project's "no silent fallbacks" rule (CLAUDE.md)
    #   requires the unsafe path to be a conscious caller decision, not
    #   library behavior that flips on a caught exception.
    if trusted:
        checkpoint = torch.load(
            buffer, map_location="cpu", weights_only=False,
        )
    else:
        checkpoint = torch.load(
            buffer, map_location="cpu", weights_only=True,
        )
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
