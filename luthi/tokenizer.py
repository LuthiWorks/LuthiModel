"""BPE (Byte Pair Encoding) tokenizer for LuthiLM.

Learns subword tokens from a training corpus by iteratively merging
the most frequent adjacent byte pairs. This gives the model a richer
vocabulary than raw characters while remaining lossless — any UTF-8
text can be encoded and decoded perfectly.

The base vocabulary is 256 byte-level tokens. Merges are learned on
top of these, building subword units like "th", "the", "ing", etc.

Usage:
    tokenizer = BPETokenizer(vocab_size=4096)
    tokenizer.train(corpus_text)
    tokenizer.save("tokenizer.json")

    # Later:
    tokenizer = BPETokenizer.load("tokenizer.json")
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


class BPETokenizer:
    """Byte Pair Encoding tokenizer.

    Starts with 256 byte-level tokens and learns merge rules to build
    a subword vocabulary of the target size. Lossless by construction —
    any byte sequence can be represented.
    """

    def __init__(self, target_vocab_size: int = 4096):
        self.target_vocab_size = target_vocab_size
        self.merges: list[tuple[int, int]] = []
        # Base vocabulary: one token per byte value
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size: int = 256

    def train(self, text: str) -> None:
        """Learn BPE merge rules from a text corpus.

        Uses numpy vectorization for pair counting and merge application,
        making it fast enough to train on tens of MB in minutes.

        Args:
            text: Training corpus as a string.
        """
        raw = text.encode("utf-8")
        tokens = np.array(list(raw), dtype=np.int32)

        num_merges = self.target_vocab_size - 256
        for i in range(num_merges):
            if len(tokens) < 2:
                break

            # Count adjacent pairs using numpy vectorization
            left = tokens[:-1]
            right = tokens[1:]
            max_id = 256 + i  # Current max possible token ID
            pair_ids = left.astype(np.int64) * max_id + right
            counts = np.bincount(pair_ids)

            if counts.max() == 0:
                break

            best_pair_id = counts.argmax()
            best_freq = int(counts[best_pair_id])
            best_a = int(best_pair_id // max_id)
            best_b = int(best_pair_id % max_id)
            best_pair = (best_a, best_b)
            new_id = 256 + i

            # Record the merge
            self.merges.append(best_pair)
            self.vocab[new_id] = self.vocab[best_a] + self.vocab[best_b]

            # Apply merge using numpy
            tokens = self._apply_merge_np(tokens, best_a, best_b, new_id)

            if (i + 1) % 500 == 0:
                print(f"  BPE merge {i + 1}/{num_merges} "
                      f"({self.vocab[new_id]!r}, freq={best_freq})")

        self.vocab_size = 256 + len(self.merges)
        self._merge_rank = self._build_merge_rank()
        print(f"BPE training complete: {self.vocab_size} tokens "
              f"({len(self.merges)} merges)")

    @staticmethod
    def _apply_merge_np(
        tokens: np.ndarray, a: int, b: int, new_id: int
    ) -> np.ndarray:
        """Replace all non-overlapping occurrences of (a, b) with new_id."""
        # Find positions where the pair occurs
        match = (tokens[:-1] == a) & (tokens[1:] == b)
        positions = np.where(match)[0]

        if len(positions) == 0:
            return tokens

        # Handle overlaps: if two matches are adjacent, keep the first
        if len(positions) > 1:
            keep = np.ones(len(positions), dtype=bool)
            keep[1:] = np.diff(positions) > 1
            positions = positions[keep]

        # Mark second element of each pair for removal
        skip = np.zeros(len(tokens), dtype=bool)
        skip[positions + 1] = True

        # Replace first element with new_id
        tokens[positions] = new_id

        # Remove skipped elements
        return tokens[~skip]

    @staticmethod
    def _apply_merge(
        tokens: list[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        """Replace all occurrences of pair in tokens with new_id (list version)."""
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def _build_merge_rank(self) -> dict[tuple[int, int], int]:
        """Build lookup from pair -> merge rank (lower = higher priority)."""
        return {pair: i for i, pair in enumerate(self.merges)}

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """Encode a single chunk using greedy best-first merge.

        Instead of applying all merges sequentially (O(merges * n)),
        this finds the highest-priority applicable merge and applies it,
        repeating until no more merges apply. Much faster for large texts.
        """
        if not chunk_bytes:
            return []
        tokens = list(chunk_bytes)
        merge_rank = self._merge_rank

        while len(tokens) >= 2:
            # Find the pair with the lowest merge rank (highest priority)
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break  # No more applicable merges

            new_id = 256 + best_rank
            tokens = self._apply_merge(tokens, best_pair, new_id)

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs.

        Splits text into lines and encodes each chunk separately for
        performance. Newlines are preserved as byte tokens.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        if not text:
            return []

        # Cache merge rank lookup
        if not hasattr(self, "_merge_rank") or self._merge_rank is None:
            self._merge_rank = self._build_merge_rank()

        # Encode in line-sized chunks for performance
        raw = text.encode("utf-8")
        chunks = raw.split(b"\n")
        result = []
        for i, chunk in enumerate(chunks):
            if chunk:
                result.extend(self._encode_chunk(chunk))
            if i < len(chunks) - 1:
                result.append(ord("\n"))  # Preserve newline as byte token
        return result

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        if not ids:
            return ""
        raw = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> Path:
        """Save tokenizer state to a JSON file.

        Only the merges are saved — the full vocabulary is reconstructed
        deterministically from the base 256 bytes + merge rules.

        Args:
            path: Output file path.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        data = {
            "version": 1,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a saved JSON file.

        Args:
            path: Path to tokenizer JSON file.

        Returns:
            Reconstructed BPETokenizer.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        tok = cls(target_vocab_size=data["target_vocab_size"])
        tok.merges = [tuple(pair) for pair in data["merges"]]
        # Reconstruct vocabulary from merges
        for i, (a, b) in enumerate(tok.merges):
            new_id = 256 + i
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.vocab_size = data["vocab_size"]
        tok._merge_rank = tok._build_merge_rank()
        return tok

    def get_state(self) -> dict:
        """Return tokenizer state for embedding in checkpoints."""
        return {
            "type": "bpe",
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        """Reconstruct tokenizer from checkpoint-embedded state."""
        tok = cls(target_vocab_size=state["target_vocab_size"])
        tok.merges = [tuple(pair) for pair in state["merges"]]
        for i, (a, b) in enumerate(tok.merges):
            new_id = 256 + i
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.vocab_size = state["vocab_size"]
        tok._merge_rank = tok._build_merge_rank()
        return tok
