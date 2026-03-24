"""Character-level dataset for training.

Handles text loading, character tokenization, and sequence generation.
No external tokenizer needed — each unique character is a token.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class CharTokenizer:
    """Minimal character-level tokenizer.

    Maps each unique character in the corpus to an integer index.
    Vocabulary is built from the training data.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in indices)


class CharDataset(Dataset):
    """Dataset of overlapping character sequences for next-char prediction.

    Each item is a (input, target) pair where target is input shifted
    by one character.
    """

    def __init__(self, text: str, seq_len: int = 128, tokenizer: CharTokenizer | None = None):
        self.seq_len = seq_len

        if tokenizer is None:
            self.tokenizer = CharTokenizer(text)
        else:
            self.tokenizer = tokenizer

        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        # Number of complete sequences we can extract
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]  # Input: characters 0..seq_len-1
        y = chunk[1:]   # Target: characters 1..seq_len
        return x, y


def load_corpus(*paths: str | Path) -> str:
    """Load and concatenate text files into a single corpus.

    Strips Project Gutenberg headers/footers if detected.
    """
    texts = []
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        text = _strip_gutenberg(text)
        texts.append(text)
    return "\n\n".join(texts)


def _strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer if present."""
    # Find start marker
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip past the marker line
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break

    # Find end marker
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()
