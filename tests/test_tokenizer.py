"""Tests for the BPE tokenizer and load_corpus fixes.

Validates:
1. BPE training produces merges
2. Encode/decode roundtrip is lossless
3. Save/load preserves state
4. Integration with CharDataset and LuthiLM
5. load_corpus handles directories correctly
"""

import json
import torch
import pytest
from pathlib import Path

from luthi.tokenizer import BPETokenizer
from luthi.data import CharDataset, CharTokenizer, Tokenizer, load_corpus
from luthi.model import LuthiLM
from luthi.train import generate


SAMPLE_TEXT = (
    "Once upon a time there was a velveteen rabbit, and in the beginning "
    "he was really splendid. He was fat and bunchy, as a rabbit should be; "
    "his coat was spotted brown and white, he had real thread whiskers, "
    "and his ears were lined with pink sateen. On Christmas morning, when "
    "he sat wedged in the top of the Boy's stocking, with a sprig of holly "
    "between his paws, the effect was charming. There were other things in "
    "the stocking, nuts and oranges and a toy engine, and chocolate almonds "
    "and a clockwork mouse, but the Rabbit was quite the best of all. "
    "For at least two hours the Boy loved him, and then Aunts and Uncles "
    "came to dinner, and there was a great rustling of tissue paper and "
    "unwrapping of parcels, and in the excitement of looking at all the "
    "new presents the Velveteen Rabbit was forgotten."
)


# ======================================================================
# BPE Training
# ======================================================================

class TestBPETraining:
    def test_train_produces_merges(self):
        tok = BPETokenizer(target_vocab_size=300)
        tok.train(SAMPLE_TEXT)
        assert len(tok.merges) > 0
        assert len(tok.merges) == 300 - 256

    def test_vocab_size_matches_target(self):
        tok = BPETokenizer(target_vocab_size=280)
        tok.train(SAMPLE_TEXT)
        assert tok.vocab_size == 280

    def test_base_vocab_is_bytes(self):
        tok = BPETokenizer(target_vocab_size=260)
        tok.train(SAMPLE_TEXT)
        # First 256 entries should be single bytes
        for i in range(256):
            assert tok.vocab[i] == bytes([i])

    def test_merges_build_on_base(self):
        tok = BPETokenizer(target_vocab_size=270)
        tok.train(SAMPLE_TEXT)
        # Each merge should produce a token that is the concatenation of its parts
        for i, (a, b) in enumerate(tok.merges):
            new_id = 256 + i
            assert tok.vocab[new_id] == tok.vocab[a] + tok.vocab[b]

    def test_small_corpus(self):
        tok = BPETokenizer(target_vocab_size=260)
        tok.train("hello world")
        assert tok.vocab_size <= 260
        assert len(tok.merges) > 0


# ======================================================================
# Encode / Decode
# ======================================================================

class TestBPEEncodeDecode:
    @pytest.fixture
    def trained_tok(self):
        tok = BPETokenizer(target_vocab_size=350)
        tok.train(SAMPLE_TEXT)
        return tok

    def test_roundtrip_training_text(self, trained_tok):
        """decode(encode(text)) == text for training corpus."""
        encoded = trained_tok.encode(SAMPLE_TEXT)
        decoded = trained_tok.decode(encoded)
        assert decoded == SAMPLE_TEXT

    def test_roundtrip_unseen_text(self, trained_tok):
        """Roundtrip works on text not in training corpus."""
        text = "The quick brown fox jumps over the lazy dog."
        encoded = trained_tok.encode(text)
        decoded = trained_tok.decode(encoded)
        assert decoded == text

    def test_encode_empty_string(self, trained_tok):
        assert trained_tok.encode("") == []

    def test_decode_empty_list(self, trained_tok):
        assert trained_tok.decode([]) == ""

    def test_encoding_is_shorter_than_bytes(self, trained_tok):
        """BPE should compress text — fewer tokens than raw bytes."""
        raw_len = len(SAMPLE_TEXT.encode("utf-8"))
        encoded_len = len(trained_tok.encode(SAMPLE_TEXT))
        assert encoded_len < raw_len

    def test_single_character(self, trained_tok):
        assert trained_tok.decode(trained_tok.encode("a")) == "a"

    def test_unicode_roundtrip(self, trained_tok):
        text = "Hello \u00e9\u00e0\u00fc \u2014 world"
        assert trained_tok.decode(trained_tok.encode(text)) == text

    def test_newlines_and_whitespace(self, trained_tok):
        text = "line one\nline two\ttab"
        assert trained_tok.decode(trained_tok.encode(text)) == text


# ======================================================================
# Save / Load
# ======================================================================

class TestBPESaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        tok = BPETokenizer(target_vocab_size=300)
        tok.train(SAMPLE_TEXT)
        path = tmp_path / "tokenizer.json"
        tok.save(path)

        loaded = BPETokenizer.load(path)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.merges == tok.merges
        assert loaded.encode(SAMPLE_TEXT) == tok.encode(SAMPLE_TEXT)

    def test_loaded_decode_matches(self, tmp_path):
        tok = BPETokenizer(target_vocab_size=300)
        tok.train(SAMPLE_TEXT)
        encoded = tok.encode(SAMPLE_TEXT)

        path = tmp_path / "tokenizer.json"
        tok.save(path)
        loaded = BPETokenizer.load(path)

        assert loaded.decode(encoded) == SAMPLE_TEXT

    def test_save_file_is_valid_json(self, tmp_path):
        tok = BPETokenizer(target_vocab_size=270)
        tok.train(SAMPLE_TEXT)
        path = tmp_path / "tokenizer.json"
        tok.save(path)
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "merges" in data

    def test_get_state_from_state_roundtrip(self):
        tok = BPETokenizer(target_vocab_size=300)
        tok.train(SAMPLE_TEXT)
        state = tok.get_state()
        restored = BPETokenizer.from_state(state)
        assert restored.encode(SAMPLE_TEXT) == tok.encode(SAMPLE_TEXT)
        assert restored.decode(tok.encode(SAMPLE_TEXT)) == SAMPLE_TEXT


# ======================================================================
# Protocol compliance
# ======================================================================

class TestTokenizerProtocol:
    def test_char_tokenizer_satisfies_protocol(self):
        tok = CharTokenizer(SAMPLE_TEXT)
        assert isinstance(tok, Tokenizer)

    def test_bpe_tokenizer_satisfies_protocol(self):
        tok = BPETokenizer(target_vocab_size=270)
        tok.train(SAMPLE_TEXT)
        assert isinstance(tok, Tokenizer)


# ======================================================================
# Integration with existing pipeline
# ======================================================================

class TestBPEIntegration:
    @pytest.fixture
    def bpe_tok(self):
        tok = BPETokenizer(target_vocab_size=300)
        tok.train(SAMPLE_TEXT)
        return tok

    def test_dataset_with_bpe(self, bpe_tok):
        """CharDataset works with BPETokenizer."""
        ds = CharDataset(SAMPLE_TEXT, seq_len=32, tokenizer=bpe_tok, stride=8)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)
        # All token IDs should be valid
        assert x.max().item() < bpe_tok.vocab_size
        assert y.max().item() < bpe_tok.vocab_size

    def test_model_forward_with_bpe(self, bpe_tok):
        """LuthiLM forward pass works with BPE vocab size."""
        model = LuthiLM(
            vocab_size=bpe_tok.vocab_size,
            d_model=32,
            n_blocks=1,
            max_seq_len=32,
        )
        ds = CharDataset(SAMPLE_TEXT, seq_len=32, tokenizer=bpe_tok, stride=8)
        x, y = ds[0]
        logits = model(x.unsqueeze(0))
        assert logits.shape == (1, 32, bpe_tok.vocab_size)

    def test_generate_with_bpe(self, bpe_tok):
        """generate() produces text with BPE tokenizer."""
        model = LuthiLM(
            vocab_size=bpe_tok.vocab_size,
            d_model=32,
            n_blocks=1,
            max_seq_len=32,
        )
        text = generate(model, bpe_tok, "The ", max_len=20, max_seq_len=32)
        assert len(text) > 4  # At least the prompt

    def test_bpe_vs_char_vocab_size(self, bpe_tok):
        """BPE vocab is larger than char vocab for same corpus."""
        char_tok = CharTokenizer(SAMPLE_TEXT)
        assert bpe_tok.vocab_size > char_tok.vocab_size

    def test_bpe_fewer_tokens_than_char(self, bpe_tok):
        """BPE produces fewer tokens than char for same text."""
        char_tok = CharTokenizer(SAMPLE_TEXT)
        bpe_len = len(bpe_tok.encode(SAMPLE_TEXT))
        char_len = len(char_tok.encode(SAMPLE_TEXT))
        assert bpe_len < char_len


# ======================================================================
# load_corpus directory handling fix
# ======================================================================

class TestLoadCorpus:
    def test_load_single_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        corpus = load_corpus(f)
        assert "hello world" in corpus

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("file one", encoding="utf-8")
        (tmp_path / "b.txt").write_text("file two", encoding="utf-8")
        corpus = load_corpus(tmp_path)
        assert "file one" in corpus
        assert "file two" in corpus

    def test_load_directory_no_txt_raises(self, tmp_path):
        (tmp_path / "data.csv").write_text("not a txt", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path)

    def test_load_mixed_files_and_dirs(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "a.txt").write_text("from dir", encoding="utf-8")
        f = tmp_path / "direct.txt"
        f.write_text("direct file", encoding="utf-8")
        corpus = load_corpus(f, subdir)
        assert "direct file" in corpus
        assert "from dir" in corpus
