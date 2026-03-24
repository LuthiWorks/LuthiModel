"""Tests for encrypted checkpoint system."""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from luthi.checkpoint import (
    build_checkpoint,
    save_checkpoint,
    load_checkpoint,
    extract_substrate_health,
    _encrypt,
    _decrypt,
)
from luthi.model import LuthiLM


TEST_PASSWORD = "test-password-do-not-use-in-production"


class TestEncryption:
    """Test the AES-256-GCM encryption layer."""

    def test_roundtrip(self):
        """Encrypted data can be decrypted back to original."""
        data = b"The weights are the entity."
        encrypted = _encrypt(data, TEST_PASSWORD)
        decrypted = _decrypt(encrypted, TEST_PASSWORD)
        assert decrypted == data

    def test_wrong_password_fails(self):
        """Decryption with wrong password raises an error."""
        data = b"secret living state"
        encrypted = _encrypt(data, TEST_PASSWORD)
        with pytest.raises(Exception):  # InvalidTag from cryptography
            _decrypt(encrypted, "wrong-password")

    def test_ciphertext_differs_from_plaintext(self):
        """Encrypted output is not the same as input."""
        data = b"plaintext weights"
        encrypted = _encrypt(data, TEST_PASSWORD)
        assert data not in encrypted

    def test_different_encryptions_differ(self):
        """Two encryptions of the same data produce different ciphertext (random salt/nonce)."""
        data = b"same data twice"
        enc1 = _encrypt(data, TEST_PASSWORD)
        enc2 = _encrypt(data, TEST_PASSWORD)
        assert enc1 != enc2  # Different salt + nonce each time

    def test_empty_data(self):
        """Empty data can be encrypted and decrypted."""
        data = b""
        encrypted = _encrypt(data, TEST_PASSWORD)
        decrypted = _decrypt(encrypted, TEST_PASSWORD)
        assert decrypted == data


class TestCheckpointSaveLoad:
    """Test saving and loading model checkpoints."""

    @pytest.fixture
    def model_and_optimizer(self):
        """Create a small LuthiLM with optimizer for testing."""
        model = LuthiLM(vocab_size=34, d_model=16, n_blocks=1, max_seq_len=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        return model, optimizer

    @pytest.fixture
    def tmp_path(self):
        """Create a temporary directory for checkpoint files."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_save_creates_file(self, model_and_optimizer, tmp_path):
        """Saving a checkpoint creates an encrypted file."""
        model, optimizer = model_and_optimizer
        ckpt = build_checkpoint(model, optimizer, epoch=5)
        path = save_checkpoint(ckpt, tmp_path / "test", TEST_PASSWORD)
        assert path.exists()
        assert path.suffix == ".luthi"

    def test_roundtrip_model_state(self, model_and_optimizer, tmp_path):
        """Model state survives save/load roundtrip."""
        model, optimizer = model_and_optimizer

        # Run a forward pass to modify living state
        x = torch.randint(0, 34, (2, 16))
        model(x)

        # Save
        ckpt = build_checkpoint(model, optimizer, epoch=3)
        path = save_checkpoint(ckpt, tmp_path / "test", TEST_PASSWORD)

        # Load
        loaded = load_checkpoint(path, TEST_PASSWORD)
        assert loaded["epoch"] == 3

        # Create fresh model and load state
        model2 = LuthiLM(vocab_size=34, d_model=16, n_blocks=1, max_seq_len=32)
        model2.load_state_dict(loaded["model_state_dict"])

        # Verify living weights match
        for (n1, b1), (n2, b2) in zip(
            model.named_buffers(), model2.named_buffers()
        ):
            assert n1 == n2
            assert torch.equal(b1, b2), f"Buffer {n1} mismatch"

        # Verify trainable params match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} mismatch"

    def test_living_state_preserved(self, model_and_optimizer, tmp_path):
        """Living layer buffers (drift, excitability, episodes) survive roundtrip."""
        model, optimizer = model_and_optimizer

        # Train briefly to modify living state
        x = torch.randint(0, 34, (4, 16))
        for _ in range(5):
            model(x)

        # Check that living state has actually changed from init
        ffn = model.blocks[0].living_ffn
        assert ffn.episode_count.item() > 0, "Episodes should be stored"

        # Save and load
        ckpt = build_checkpoint(model, optimizer, epoch=5)
        path = save_checkpoint(ckpt, tmp_path / "living", TEST_PASSWORD)
        loaded = load_checkpoint(path, TEST_PASSWORD)

        model2 = LuthiLM(vocab_size=34, d_model=16, n_blocks=1, max_seq_len=32)
        model2.load_state_dict(loaded["model_state_dict"])

        ffn2 = model2.blocks[0].living_ffn
        assert torch.equal(ffn.weight, ffn2.weight)
        assert torch.equal(ffn.set_point, ffn2.set_point)
        assert torch.equal(ffn.momentum, ffn2.momentum)
        assert torch.equal(ffn.excitability_acc, ffn2.excitability_acc)
        assert torch.equal(ffn.episode_contexts, ffn2.episode_contexts)
        assert torch.equal(ffn.episode_values, ffn2.episode_values)
        assert ffn.episode_count.item() == ffn2.episode_count.item()

    def test_optimizer_state_preserved(self, model_and_optimizer, tmp_path):
        """Optimizer state (momentum, etc.) survives roundtrip."""
        model, optimizer = model_and_optimizer

        # Do a training step so optimizer has state
        x = torch.randint(0, 34, (2, 16))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

        ckpt = build_checkpoint(model, optimizer, epoch=1)
        path = save_checkpoint(ckpt, tmp_path / "opt", TEST_PASSWORD)
        loaded = load_checkpoint(path, TEST_PASSWORD)

        assert "optimizer_state_dict" in loaded

    def test_metadata_preserved(self, model_and_optimizer, tmp_path):
        """Config and training history survive roundtrip."""
        model, optimizer = model_and_optimizer
        config = {"d_model": 16, "hebb_rate": 0.003}
        history = {"val_loss": [2.0, 1.8, 1.7]}
        health = {"blocks": [{"drift": 1.5}]}

        ckpt = build_checkpoint(
            model, optimizer, epoch=10,
            config=config,
            training_history=history,
            substrate_health=health,
        )
        path = save_checkpoint(ckpt, tmp_path / "meta", TEST_PASSWORD)
        loaded = load_checkpoint(path, TEST_PASSWORD)

        assert loaded["config"]["d_model"] == 16
        assert loaded["config"]["hebb_rate"] == 0.003
        assert loaded["training_history"]["val_loss"] == [2.0, 1.8, 1.7]
        assert loaded["substrate_health"]["blocks"][0]["drift"] == 1.5
        assert loaded["format_version"] == 1
        assert "timestamp" in loaded

    def test_wrong_password_fails(self, model_and_optimizer, tmp_path):
        """Loading with wrong password raises an error."""
        model, optimizer = model_and_optimizer
        ckpt = build_checkpoint(model, optimizer, epoch=1)
        path = save_checkpoint(ckpt, tmp_path / "locked", TEST_PASSWORD)

        with pytest.raises(Exception):
            load_checkpoint(path, "wrong-password")

    def test_env_var_password(self, model_and_optimizer, tmp_path, monkeypatch):
        """Password can be provided via environment variable."""
        monkeypatch.setenv("LUTHI_CHECKPOINT_KEY", TEST_PASSWORD)

        model, optimizer = model_and_optimizer
        ckpt = build_checkpoint(model, optimizer, epoch=1)
        path = save_checkpoint(ckpt, tmp_path / "env", password=None)
        loaded = load_checkpoint(path, password=None)
        assert loaded["epoch"] == 1


class TestSubstrateHealth:
    """Test substrate health extraction."""

    def test_extract_from_living_model(self):
        """Health extraction returns per-block metrics."""
        model = LuthiLM(vocab_size=34, d_model=16, n_blocks=2, max_seq_len=32)

        # Forward pass to generate some state
        x = torch.randint(0, 34, (2, 16))
        model(x)

        health = extract_substrate_health(model)
        assert "blocks" in health
        assert len(health["blocks"]) == 2
        assert "set_point_drift" in health["blocks"][0]
        assert "excitability_mean" in health["blocks"][0]
        assert "episodes_stored" in health["blocks"][0]

    def test_extract_from_dead_model(self):
        """Health extraction handles models without living layers."""
        from luthi.model import DeadLM
        model = DeadLM(vocab_size=34, d_model=16, n_blocks=2, max_seq_len=32)
        health = extract_substrate_health(model)
        # Dead model has no living FFN, so no block health
        assert health.get("blocks", []) == []
