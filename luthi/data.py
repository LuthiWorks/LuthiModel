"""Dataset and tokenization for training.

Handles text loading, tokenization, and sequence generation.
Supports both character-level (CharTokenizer) and subword (BPETokenizer)
tokenization.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Interface that any tokenizer must satisfy."""
    vocab_size: int
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


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

    Accepts either a text string or a pre-encoded tensor (for streaming
    large corpora without loading the full text into memory).
    """

    def __init__(self, text_or_data: str | torch.Tensor, seq_len: int = 128,
                 tokenizer: Any = None, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride

        if isinstance(text_or_data, torch.Tensor):
            self.data = text_or_data
            self.tokenizer = tokenizer
        else:
            if tokenizer is None:
                self.tokenizer = CharTokenizer(text_or_data)
            else:
                self.tokenizer = tokenizer
            self.data = self._encode_chunked(text_or_data)

    def _encode_chunked(self, text: str, chunk_chars: int = 10_000_000) -> torch.Tensor:
        """Encode text in chunks to avoid giant Python list intermediary.

        Each chunk is encoded and converted to a tensor immediately,
        then all chunks are concatenated. This keeps peak memory
        proportional to chunk size rather than full corpus size.
        """
        if len(text) <= chunk_chars:
            return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        chunks = []
        # Split on newlines near chunk boundaries to avoid breaking BPE tokens
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            # Find a newline near the boundary to split cleanly
            if end < len(text):
                newline = text.rfind("\n", start + chunk_chars // 2, end)
                if newline != -1:
                    end = newline + 1
            encoded = self.tokenizer.encode(text[start:end])
            chunks.append(torch.tensor(encoded, dtype=torch.long))
            start = end

        return torch.cat(chunks)

    def __len__(self) -> int:
        # Number of complete sequences we can extract
        return max(0, (len(self.data) - self.seq_len) // self.stride)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = self.data[start : start + self.seq_len + 1]
        x = chunk[:-1]  # Input: characters 0..seq_len-1
        y = chunk[1:]   # Target: characters 1..seq_len
        return x, y


def load_corpus(*paths: str | Path) -> str:
    """Load and concatenate text files into a single corpus.

    Accepts both file paths and directories. Directories are automatically
    globbed for *.txt files. Strips Project Gutenberg headers/footers
    if detected.
    """
    texts = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            file_paths = sorted(path.glob("*.txt"))
            if not file_paths:
                raise FileNotFoundError(f"No .txt files found in {path}")
            for fp in file_paths:
                text = fp.read_text(encoding="utf-8")
                text = _strip_gutenberg(text)
                texts.append(text)
        else:
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


def load_file_list(list_path: str | Path) -> list[Path]:
    """Load an ordered list of file paths from a file_list.txt.

    Each line is an absolute path to a .txt file. Empty lines and
    lines starting with # are skipped. Order is preserved — this is
    how curriculum ordering flows into the training pipeline.
    """
    list_path = Path(list_path)
    if not list_path.exists():
        raise FileNotFoundError(f"File list not found: {list_path}")

    paths = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if p.exists():
            paths.append(p)

    if not paths:
        raise ValueError(f"No valid file paths found in {list_path}")

    return paths


def _resolve_file_paths(*paths: str | Path) -> list[Path]:
    """Resolve paths into a flat list of .txt file paths."""
    file_paths = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            fps = sorted(path.glob("*.txt"))
            if not fps:
                raise FileNotFoundError(f"No .txt files found in {path}")
            file_paths.extend(fps)
        else:
            file_paths.append(path)
    return file_paths


def load_corpus_sample(*paths: str | Path, max_bytes: int = 20_000_000) -> str:
    """Load a representative sample from files, striding across the corpus.

    Used for tokenizer training — needs coverage across all domains,
    not just the first N files. Strides evenly through the file list
    so that every part of the corpus contributes to the sample.
    Strips Gutenberg headers/footers.
    """
    file_paths = _resolve_file_paths(*paths)
    n_files = len(file_paths)
    if n_files == 0:
        return ""

    # Stride through files evenly to get cross-corpus coverage
    # Start with a stride that would touch ~2x more files than needed,
    # then stop when we hit max_bytes
    stride = max(1, n_files // max(1, n_files))  # start at 1, try every file
    # But if we have way more files than needed, skip some
    # Estimate: average file ~600KB, so for 200MB we need ~330 files
    estimated_files_needed = max_bytes // 500_000  # conservative estimate
    if n_files > estimated_files_needed * 2:
        stride = n_files // estimated_files_needed

    texts = []
    total = 0
    idx = 0
    while idx < n_files and total < max_bytes:
        fp = file_paths[idx]
        try:
            text = fp.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            idx += stride
            continue
        text = _strip_gutenberg(text)
        texts.append(text)
        total += len(text.encode("utf-8"))
        idx += stride
    return "\n\n".join(texts)


def load_corpus_as_tensor(
    *paths: str | Path,
    tokenizer: Any,
    progress: bool = True,
) -> torch.Tensor:
    """Stream-encode files into a single tensor without loading full corpus.

    Processes one file at a time: read → strip gutenberg → encode → tensor.
    Peak memory is proportional to the largest single file, not the total
    corpus size. Returns a concatenated int64 tensor ready for CharDataset.
    """
    file_paths = _resolve_file_paths(*paths)
    chunks: list[torch.Tensor] = []
    total_tokens = 0

    for i, fp in enumerate(file_paths):
        text = fp.read_text(encoding="utf-8")
        text = _strip_gutenberg(text)
        if not text:
            continue
        encoded = tokenizer.encode(text)
        if encoded:
            chunks.append(torch.tensor(encoded, dtype=torch.long))
            total_tokens += len(encoded)
        # Free the string immediately
        del text, encoded

        if progress and (i + 1) % 1000 == 0:
            print(f"  Encoded {i + 1}/{len(file_paths)} files ({total_tokens:,} tokens)")

    if not chunks:
        raise ValueError("No text found in corpus files")

    if progress:
        print(f"  Encoded {len(file_paths)}/{len(file_paths)} files ({total_tokens:,} tokens)")

    return torch.cat(chunks)
