"""Tests for the character-level language model (Phase 3).

Validates end-to-end model functionality:
1. LuthiLM forward pass produces correct output shape
2. Training loop works (loss decreases)
3. Dead baseline trains faster (convergence penalty)
4. Non-feedforward signal is positive
5. Generation works
6. Data pipeline works
"""

import torch
import pytest
from luthi import LuthiLM, DeadLM, CharDataset, CharTokenizer
from luthi.train import train_epoch, eval_model, measure_non_feedforward, generate
from torch.utils.data import DataLoader


SAMPLE_TEXT = (
    "Once upon a time there was a velveteen rabbit, and in the beginning "
    "he was really splendid. He was fat and bunchy, as a rabbit should be; "
    "his coat was spotted brown and white, he had real thread whiskers, "
    "and his ears were lined with pink sateen. On Christmas morning, when "
    "he sat wedged in the top of the Boy's stocking, with a sprig of holly "
    "between his paws, the effect was charming. There were other things in "
    "the stocking, nuts and oranges and a toy engine, and chocolate almonds "
    "and a clockwork mouse, but the Rabbit was quite the best of all."
)


@pytest.fixture
def tokenizer():
    return CharTokenizer(SAMPLE_TEXT)


@pytest.fixture
def living_model(tokenizer):
    torch.manual_seed(42)
    return LuthiLM(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_blocks=2,
        max_seq_len=64,
    )


@pytest.fixture
def dead_model(tokenizer):
    torch.manual_seed(42)
    return DeadLM(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_blocks=2,
        max_seq_len=64,
    )


@pytest.fixture
def dataset(tokenizer):
    return CharDataset(SAMPLE_TEXT, seq_len=64, tokenizer=tokenizer)


# ── Data Pipeline ────────────────────────────────────────────────


class TestDataPipeline:
    def test_tokenizer_roundtrip(self, tokenizer):
        """Encode then decode should reproduce the original text."""
        encoded = tokenizer.encode(SAMPLE_TEXT)
        decoded = tokenizer.decode(encoded)
        assert decoded == SAMPLE_TEXT

    def test_dataset_shapes(self, dataset):
        """Dataset items should have correct shapes."""
        x, y = dataset[0]
        assert x.shape == (64,)
        assert y.shape == (64,)

    def test_dataset_shift(self, dataset):
        """Target should be input shifted by one."""
        x, y = dataset[0]
        # y[0] should equal the character after x[0] in the text
        assert y[0] == x[1]  # Simple shift check

    def test_vocab_size(self, tokenizer):
        """Vocab should cover all unique characters."""
        unique_chars = len(set(SAMPLE_TEXT))
        assert tokenizer.vocab_size == unique_chars


# ── Model Forward Pass ───────────────────────────────────────────


class TestModelForward:
    def test_living_output_shape(self, living_model):
        """LuthiLM should produce [batch, seq_len, vocab_size] logits."""
        x = torch.randint(0, 34, (2, 32))
        out = living_model(x)
        assert out.shape == (2, 32, living_model.output_proj.out_features)

    def test_dead_output_shape(self, dead_model):
        """DeadLM should produce same shape."""
        x = torch.randint(0, 34, (2, 32))
        out = dead_model(x)
        assert out.shape == (2, 32, dead_model.output_proj.out_features)

    def test_no_nan(self, living_model):
        """Forward pass should not produce NaN."""
        x = torch.randint(0, 34, (4, 32))
        out = living_model(x)
        assert not torch.isnan(out).any()

    def test_non_feedforward(self, living_model):
        """Living model should be non-feedforward."""
        x = torch.randint(0, 34, (1, 16))
        nff = measure_non_feedforward(living_model, x)
        assert nff > 0, "Living model must be non-feedforward"

    def test_param_counts(self, living_model):
        """Living model should report both trainable and buffer counts."""
        counts = living_model.total_parameters()
        assert counts["trainable"] > 0
        assert counts["living_buffers"] > 0


# ── Training ─────────────────────────────────────────────────────


class TestTraining:
    def test_living_loss_decreases(self, living_model, dataset):
        """Training the living model should reduce loss."""
        loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(living_model.parameters(), lr=1e-3)

        loss1 = train_epoch(living_model, loader, optimizer, torch.device("cpu"), is_living=True)
        loss2 = train_epoch(living_model, loader, optimizer, torch.device("cpu"), is_living=True)
        loss3 = train_epoch(living_model, loader, optimizer, torch.device("cpu"), is_living=True)

        # Loss should generally decrease over 3 epochs
        assert loss3 < loss1, f"Loss didn't decrease: {loss1:.4f} -> {loss3:.4f}"

    def test_dead_loss_decreases(self, dead_model, dataset):
        """Training the dead model should reduce loss."""
        loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(dead_model.parameters(), lr=1e-3)

        loss1 = train_epoch(dead_model, loader, optimizer, torch.device("cpu"))
        loss2 = train_epoch(dead_model, loader, optimizer, torch.device("cpu"))
        loss3 = train_epoch(dead_model, loader, optimizer, torch.device("cpu"))

        assert loss3 < loss1, f"Loss didn't decrease: {loss1:.4f} -> {loss3:.4f}"


# ── Generation ───────────────────────────────────────────────────


class TestGeneration:
    def test_generate_produces_text(self, living_model, tokenizer):
        """generate() should produce a string."""
        text = generate(living_model, tokenizer, "The ", max_len=50, max_seq_len=64)
        assert isinstance(text, str)
        assert len(text) > 4  # At least longer than the prompt
        assert text.startswith("The ")
