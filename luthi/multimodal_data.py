"""Multimodal dataset for audio-text paired training.

Handles LibriSpeech-format datasets where each utterance has a FLAC audio
file and a corresponding transcript. Returns fixed-length (audio, text)
pairs suitable for the MultimodalLuthiLM.

LibriSpeech directory structure:
    <subset>/
        <speaker_id>/
            <chapter_id>/
                <speaker_id>-<chapter_id>-<utterance_id>.flac
                <speaker_id>-<chapter_id>.trans.txt
"""

from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    """Audio-text paired dataset from LibriSpeech-format directories.

    Each item returns a dict with:
        audio: [max_audio_samples] raw waveform (padded/truncated)
        text_input: [max_text_tokens] encoded text tokens (for prediction)
        text_target: [max_text_tokens] shifted tokens (prediction target)

    Args:
        root: Path to the LibriSpeech subset directory
            (e.g., LibriSpeech/train-clean-100).
        tokenizer: Text tokenizer with encode() method.
        max_audio_samples: Fixed audio length in samples. Default 160000
            (10 seconds at 16kHz).
        max_text_tokens: Fixed text sequence length for next-token prediction.
        sample_rate: Expected sample rate. Audio is resampled if different.
    """

    def __init__(
        self,
        root: str | Path,
        tokenizer,
        max_audio_samples: int = 160000,
        max_text_tokens: int = 128,
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_audio_samples = max_audio_samples
        self.max_text_tokens = max_text_tokens
        self.sample_rate = sample_rate

        # Scan directory for (audio_path, transcript) pairs
        self.entries: list[tuple[Path, str]] = []
        self._scan_directory()

        if not self.entries:
            raise ValueError(
                f"No audio-transcript pairs found in {self.root}. "
                f"Expected LibriSpeech directory structure."
            )

    def _scan_directory(self) -> None:
        """Walk the directory tree and collect (audio_path, transcript) pairs."""
        for trans_file in sorted(self.root.rglob("*.trans.txt")):
            audio_dir = trans_file.parent

            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: UTTERANCE_ID TRANSCRIPT TEXT HERE
                    parts = line.split(maxsplit=1)
                    if len(parts) < 2:
                        continue
                    utterance_id, transcript = parts

                    audio_path = audio_dir / f"{utterance_id}.flac"
                    if audio_path.exists():
                        # Lowercase to match Gutenberg training data case
                        self.entries.append((audio_path, transcript.lower()))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        audio_path, transcript = self.entries[idx]

        # --- Load and prepare audio ---
        waveform, sr = torchaudio.load(audio_path)
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [samples]

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate to fixed length
        n_samples = waveform.shape[0]
        if n_samples > self.max_audio_samples:
            waveform = waveform[:self.max_audio_samples]
        elif n_samples < self.max_audio_samples:
            pad = torch.zeros(self.max_audio_samples - n_samples)
            waveform = torch.cat([waveform, pad])

        # --- Encode text ---
        token_ids = self.tokenizer.encode(transcript)

        # Pad or truncate to max_text_tokens + 1 (need input + target)
        target_len = self.max_text_tokens + 1
        if len(token_ids) > target_len:
            token_ids = token_ids[:target_len]
        else:
            # Pad with zeros (will be masked in loss)
            token_ids = token_ids + [0] * (target_len - len(token_ids))

        tokens = torch.tensor(token_ids, dtype=torch.long)
        text_input = tokens[:-1]   # [max_text_tokens]
        text_target = tokens[1:]   # [max_text_tokens]

        return {
            "audio": waveform,
            "text_input": text_input,
            "text_target": text_target,
        }
