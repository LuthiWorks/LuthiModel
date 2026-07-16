"""Multimodal data pipeline for M8 v2.

Implements the ``MultimodalDataLoader`` Protocol from ``jepa_runner.py``.

**Build order (per 4.8 2026-06-06):** text first, smoke alone, then audio,
then vision -- so the first end-to-end run is clean and any break
attributes to one modality. This first cut implements text fully; audio
and vision are loud ``NotImplementedError`` stubs that participate in the
``MultimodalDataLoader`` dispatch but fail visibly if the sampler picks
them. That guarantees a misconfigured run cannot silently train only on
text.

**Critical contract from 4.8 (2026-06-06): yield RAW inputs, NOT
pre-encoded tokens.** ``MultimodalPredictiveCodingLM.encode`` runs the
perceptual encoder only when the batch carries ``image`` / ``audio_waveform``
/ ``text_tokens``; if it carries ``vision_tokens`` / ``audio_tokens``, the
perceptual encoder is bypassed and never gets a gradient -- it would
silently not train. This pipeline therefore yields::

    text   -> {"text_tokens": Tensor[B, seq_len]}        (BPE token IDs,
                                                          input to nn.Embedding)
    audio  -> {"audio_waveform": Tensor[B, samples]}     (raw waveform; the
                                                          AudioEncoder runs)
    vision -> {"image": Tensor[B, 3, H, W]}              (raw image; the
                                                          VisionEncoder runs)

The ``*_tokens`` path on ``encode()`` is an inference/efficiency bypass,
not a training path.

**Raw human data only** (per ``feedback_no_inherited_corpora``):
gutenberg_4gb text files, LibriSpeech waveforms, COCO images. No
pseudo-labels, no derived features, no inherited model artifacts.

**Without-replacement, resumable iteration.** Each per-modality dataset
draws sequences without replacement within an epoch using a deterministic
shuffle (seed XOR epoch number). ``state_dict`` / ``load_state_dict`` round-
trip the per-modality (epoch_index, position_within_epoch) so resume
preserves the without-replacement guarantee across crashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from luthi.data import load_corpus_as_tensor
from luthi.tokenizer import BPETokenizer
from luthi.v2.jepa_loss import MODALITIES


# Canonical M8 text corpus (v0.5 §2): the gutenberg_4gb directory -- ~2.2B
# tokens across 11,113 .txt files. Single source of truth for "the M8 text
# corpus"; default to this, NOT the M7 curriculum subset
# (corpus_build/m7_filelist.txt is a different ~1.27B-token corpus).
M8_TEXT_CORPUS = Path("E:/data/gutenberg_4gb")


# ---------------------------------------------------------------------------
# Text dataset
# ---------------------------------------------------------------------------


@dataclass
class TextDatasetConfig:
    """Configuration for the text path.

    source_paths: list of file/directory paths passed straight to
        ``load_corpus_as_tensor`` (which globs ``*.txt`` in directories
        and strips Project Gutenberg headers). For the M8 baseline per
        v0.5 §2 this is the gutenberg_4gb corpus (~2.2B tokens, 11,113
        files); typically a single directory like
        ``Path("E:/data/gutenberg_4gb")``. Use ``_read_filelist`` to
        materialize a filelist into a list of paths if you prefer that
        form (M7 used ``corpus_build/m7_filelist.txt``, a different
        curriculum subset; reconcile with Brian if you want the M7
        subset over the v0.5 spec corpus).
    tokenizer_path: path to a saved BPETokenizer (e.g.
        ``corpus_build/tokenizer_32k.json``).
    batch_size: sequences per batch.
    seq_len: tokens per sequence.
    stride: step between adjacent sequence starts. Default 64 matches the
        M6/M7 baseline (50% overlap with seq_len=128). Sets the number of
        sequences per epoch to roughly ``unique_tokens / stride``.
    base_seed: deterministic seed for shuffling. Combined with epoch index
        so each epoch sees a different shuffle.

    **Memory.** The corpus loads as a single int64 tensor: ~10 GB RAM for
    1.27B tokens, ~18 GB for 2.2B. Confirm fit at the chosen corpus.
    int64 is 4x heavier than needed for a 32K vocab; that's
    ``load_corpus_as_tensor``'s convention, not a regression here.

    **Determinism.** ``torch.randperm`` is deterministic within a given
    torch version but not guaranteed identical across versions. Resume
    on the same machine + same torch is reproducible; cross-version
    resume may serve a different shuffle order (still uniform-without-
    replacement -- just not the same sequence).
    """

    source_paths: list[Path | str]
    tokenizer_path: Path
    batch_size: int
    seq_len: int = 128
    stride: int = 64
    base_seed: int = 42
    # Held-out fraction for the JEPA eval harness (2026-07-15). The
    # holdout is the CONTIGUOUS TAIL of the token stream, separated from
    # the training region by a seq_len gap — never a random sequence
    # split: with stride < seq_len adjacent sequences share tokens, so a
    # random split leaks up to (seq_len - stride)/seq_len of every
    # held-out sequence into training and biases held-out error
    # optimistically. 0.0 = no holdout (legacy behavior, bit-identical).
    holdout_fraction: float = 0.0


def compute_text_split(
    n_tokens: int,
    seq_len: int,
    stride: int,
    holdout_fraction: float,
) -> tuple[int, list[int]]:
    """Leakage-safe train/holdout split for an overlapping-window token
    stream. Pure function so the arithmetic is directly testable.

    Returns ``(n_train_sequences, holdout_starts)``:
    - training sequences start at ``0, stride, ...`` and must END at or
      before ``train_boundary`` — where ``train_boundary`` leaves a
      ``seq_len`` GAP before the holdout region, so no training window
      overlaps (or abuts into) any holdout window;
    - holdout windows are NON-overlapping (stride = seq_len), fixed
      deterministic order, covering the tail region.

    Raises if the requested fraction leaves no training sequences or no
    holdout windows — a silent empty split would fake a perfect eval.
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in [0, 1); got {holdout_fraction}")
    max_start = n_tokens - seq_len
    if max_start < 0:
        raise ValueError(f"Corpus has {n_tokens} tokens, less than seq_len={seq_len}")
    if holdout_fraction == 0.0:
        return (max_start // stride) + 1, []

    holdout_tokens = int(n_tokens * holdout_fraction)
    holdout_begin = n_tokens - holdout_tokens
    # Training windows must end seq_len BEFORE the holdout begins (the gap).
    train_boundary = holdout_begin - seq_len
    train_max_start = train_boundary - seq_len
    if train_max_start < 0:
        raise ValueError(
            f"holdout_fraction={holdout_fraction} leaves no training "
            f"sequences (corpus {n_tokens} tokens, seq_len {seq_len})."
        )
    n_train_sequences = (train_max_start // stride) + 1

    holdout_starts = list(range(holdout_begin, n_tokens - seq_len + 1, seq_len))
    if not holdout_starts:
        raise ValueError(
            f"holdout_fraction={holdout_fraction} yields zero holdout "
            f"windows (needs >= seq_len={seq_len} tokens; got {holdout_tokens})."
        )
    return n_train_sequences, holdout_starts


class TextDataset:
    """BPE-encoded text token stream from a filelist.

    Loads the corpus into a single int64 token tensor on construction
    (cached on disk via ``load_corpus_as_tensor`` for fast re-launches).
    Within an epoch, draws sequences in a shuffled-without-replacement
    order; reshuffles between epochs with a per-epoch deterministic seed.

    Yields **raw text token IDs** in ``{"text_tokens": Tensor[B, L]}`` --
    these go straight to ``nn.Embedding`` inside the model; the perceptual
    "encoder" for text is just the embedding lookup, which always runs.
    """

    def __init__(self, config: TextDatasetConfig):
        self.config = config

        # Tokenizer + corpus encoding (cached).
        self._tokenizer = BPETokenizer.load(str(config.tokenizer_path))
        if not config.source_paths:
            raise ValueError("TextDatasetConfig.source_paths must be non-empty")
        # load_corpus_as_tensor accepts file and directory paths, globs
        # *.txt in directories, strips Gutenberg headers, and caches the
        # encoded tensor on disk keyed by file metadata + tokenizer
        # fingerprint, so launches after the first are fast.
        self._tokens: torch.Tensor = load_corpus_as_tensor(
            *config.source_paths, tokenizer=self._tokenizer,
        )
        # We may receive long dtype already; ensure it's contiguous int64
        # so views/index ops are cheap and predictable.
        if self._tokens.dtype != torch.long:
            self._tokens = self._tokens.long()

        # Iteration state.
        # Sequences are indexed by their starting position. Start positions
        # are 0, stride, 2*stride, ... such that start + seq_len <= len(tokens).
        # With holdout_fraction > 0, training positions are additionally
        # capped so no training window overlaps the leakage-gapped holdout
        # tail (see compute_text_split).
        n = self._tokens.numel()
        self._n_sequences, self._holdout_starts = compute_text_split(
            n_tokens=n,
            seq_len=config.seq_len,
            stride=config.stride,
            holdout_fraction=config.holdout_fraction,
        )

        self._epoch_index = 0
        self._pos_within_epoch = 0
        self._shuffle: Optional[torch.Tensor] = None  # built lazily
        self._build_shuffle_for_epoch(self._epoch_index)

    # -- Public API (matches MultimodalDataLoader Protocol's per-modality shape) --

    def next_batch(self) -> dict:
        """Return one batch of ``{"text_tokens": Tensor[B, seq_len]}``.

        Advances the per-epoch position; when an epoch is exhausted,
        reshuffles for the next epoch automatically. The runner's
        coverage anchor decides when "epoch" semantics matter; this
        method just keeps producing batches indefinitely.
        """
        bs = self.config.batch_size
        if self._shuffle is None:
            self._build_shuffle_for_epoch(self._epoch_index)
        assert self._shuffle is not None

        end_pos = self._pos_within_epoch + bs
        if end_pos > self._n_sequences:
            # Epoch boundary mid-batch: complete this batch from the next
            # epoch's shuffle so we don't return a short batch. This
            # subtly mixes the previous epoch's tail with the next
            # epoch's head, which is fine for SSL training (sequences
            # are i.i.d. enough at this granularity) and keeps batch size
            # uniform.
            partial_starts = self._shuffle[self._pos_within_epoch :].tolist()
            self._epoch_index += 1
            self._build_shuffle_for_epoch(self._epoch_index)
            self._pos_within_epoch = 0
            remaining = bs - len(partial_starts)
            second_starts = self._shuffle[:remaining].tolist()
            self._pos_within_epoch = remaining
            starts = partial_starts + second_starts
        else:
            starts = self._shuffle[self._pos_within_epoch : end_pos].tolist()
            self._pos_within_epoch = end_pos

        seq_len = self.config.seq_len
        batch = torch.stack(
            [self._tokens[s : s + seq_len] for s in starts],
            dim=0,
        )
        return {"text_tokens": batch}

    def batch_token_count(self, batch: dict) -> int:
        """Overlap-counted token count for a text batch: ``B * seq_len``.
        Text has no padding, so every emitted element counts. Coverage
        accumulates in this unit; see ``tokens_per_pass`` for the
        commensurate per-pass threshold."""
        return int(batch["text_tokens"].numel())

    def tokens_per_pass(self) -> int:
        """The runner-side unit: how many overlap-counted tokens equal
        one full pass over the dataset (every sequence served once).

        **Unit reconciliation (per 4.8 review 2026-06-06).** The runner's
        coverage anchor and the sampler's alpha-balance both ride on
        ``corpus_sizes_tokens``. Since ``batch_token_count`` counts
        overlap-included sequence tokens (``B * seq_len``), the threshold
        it is compared against must also be overlap-counted -- otherwise
        the anchor fires at ``stride/seq_len`` of a real pass. With the
        M6/M7 defaults (seq_len=128, stride=64), each unique token sits
        in ~2 sequences, so the overlap-counted full-pass size is
        ~2x the unique token count.

        Concretely: ``tokens_per_pass = n_sequences * seq_len``. When
        ``tokens_consumed[modality] >= tokens_per_pass``, every sequence
        in the modality has been served once.
        """
        return int(self._n_sequences) * int(self.config.seq_len)

    def unique_token_count(self) -> int:
        """Number of distinct tokens in the corpus (no overlap counting).
        For transparency / debugging; the runner does not use this --
        see ``tokens_per_pass`` for the coverage unit."""
        return int(self._tokens.numel())

    def vocab_size(self) -> int:
        return int(self._tokenizer.vocab_size)

    def holdout_batch_count(self, batch_size: int) -> int:
        """Number of full holdout batches available (0 = no holdout)."""
        return len(self._holdout_starts) // batch_size

    def holdout_batches(self, batch_size: int):
        """Yield held-out batches in fixed deterministic order.

        Non-overlapping windows from the leakage-gapped tail; the same
        windows every call, so held-out numbers are comparable across
        checkpoints and arms. Yields ``{"text_tokens": Tensor[B, L]}``;
        drops the final partial batch (fixed set > slightly larger set).
        """
        seq_len = self.config.seq_len
        for i in range(self.holdout_batch_count(batch_size)):
            starts = self._holdout_starts[i * batch_size : (i + 1) * batch_size]
            yield {
                "text_tokens": torch.stack(
                    [self._tokens[s : s + seq_len] for s in starts], dim=0,
                )
            }

    def state_dict(self) -> dict:
        """Snapshot the shuffle position so resume preserves
        without-replacement (4.8 review 2026-06-06)."""
        return {
            "epoch_index": int(self._epoch_index),
            "pos_within_epoch": int(self._pos_within_epoch),
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch_index = int(state["epoch_index"])
        self._pos_within_epoch = int(state["pos_within_epoch"])
        # Regenerate the shuffle deterministically from (base_seed,
        # epoch_index); no need to round-trip the giant permutation.
        self._build_shuffle_for_epoch(self._epoch_index)

    # -- Internals --

    def _build_shuffle_for_epoch(self, epoch_index: int) -> None:
        seed = self.config.base_seed ^ (epoch_index * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        # Use a CPU generator so the shuffle is deterministic regardless
        # of GPU state.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed & 0x7FFFFFFFFFFFFFFF))
        perm = torch.randperm(self._n_sequences, generator=gen)
        # Convert permutation indices to starting token positions.
        self._shuffle = perm * self.config.stride


# ---------------------------------------------------------------------------
# Audio / Vision stubs
# ---------------------------------------------------------------------------


class _ModalityStub:
    """Stub per-modality dataset. Loud failure on next_batch so a
    misconfigured run that requests the modality cannot silently
    fall back to a different one."""

    def __init__(self, modality_name: str):
        self._modality = modality_name

    def next_batch(self) -> dict:
        raise NotImplementedError(
            f"{self._modality} dataset not yet implemented. "
            f"Build order per 4.8 2026-06-06: text first, then audio, "
            f"then vision. If the sampler is hitting this stub, the "
            f"runner config has a modality enabled that the data "
            f"pipeline does not yet support."
        )

    def batch_token_count(self, batch: dict) -> int:
        raise NotImplementedError(f"{self._modality}.batch_token_count")

    def tokens_per_pass(self) -> int:
        raise NotImplementedError(
            f"{self._modality}.tokens_per_pass -- pipeline implementation "
            f"pending; configure SamplerConfig.corpus_sizes_tokens with an "
            f"overlap-counted per-pass value derived elsewhere or omit this "
            f"modality from the run."
        )

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


# ---------------------------------------------------------------------------
# Audio dataset (LibriSpeech)
# ---------------------------------------------------------------------------


@dataclass
class AudioDatasetConfig:
    """Configuration for the LibriSpeech audio path.

    root: directory containing .flac files (recursively scanned). For M8
        baseline: ``Path("E:/data/LibriSpeech/train-clean-100")``; use
        ``dev-clean`` for held-out evaluation.
    batch_size: waveforms per batch.
    max_audio_samples: fixed sample length; waveforms padded/truncated to
        this. Default 80000 = 5 seconds at 16 kHz.
    sample_rate: target sample rate; loaded waveforms resampled if
        different. LibriSpeech is 16 kHz natively.
    audio_encoder_hop_length / audio_encoder_patch_frames: must match
        the model's AudioEncoder so ``tokens_per_pass`` matches the actual
        per-batch token output. Defaults match
        ``MultimodalPredictiveCodingLM`` defaults.
    base_seed: deterministic seed for shuffling.

    **Yields RAW waveforms** -- ``{"audio_waveform": Tensor[B, samples]}``,
    NOT pre-encoded audio_tokens. The AudioEncoder runs inside the model's
    forward; yielding tokens here would bypass the encoder and starve it
    of gradient (per 4.8 review 2026-06-06 yield-raw contract).
    """

    root: Path
    batch_size: int
    max_audio_samples: int = 80000  # 5s @ 16 kHz
    sample_rate: int = 16000
    audio_encoder_hop_length: int = 160
    audio_encoder_patch_frames: int = 16
    base_seed: int = 42


class AudioDataset:
    """LibriSpeech waveform stream. Yields raw waveform tensors.

    Indexes .flac files recursively from ``config.root`` on construction;
    per-batch lazy-loads waveforms via ``torchaudio.load``. One sample per
    file per epoch in a deterministic shuffled order; reshuffles between
    epochs.
    """

    def __init__(self, config: AudioDatasetConfig):
        # Lazy import torchaudio so a text-only run doesn't require it
        # at module-load time.
        import torchaudio
        self._torchaudio = torchaudio
        self.config = config

        root = Path(config.root)
        if not root.exists():
            raise FileNotFoundError(
                f"AudioDataset root does not exist: {root}. For M8 the "
                f"typical path is E:/data/LibriSpeech/train-clean-100."
            )
        self._audio_paths = sorted(root.rglob("*.flac"))
        if not self._audio_paths:
            raise ValueError(
                f"AudioDataset found no .flac files under {root}. "
                f"LibriSpeech subset directories contain "
                f"<speaker>/<chapter>/*.flac."
            )
        self._n_samples = len(self._audio_paths)

        # tokens_per_sample matches AudioEncoder's internal patch math:
        # mel frames = samples // hop_length; tokens = frames // patch_frames.
        n_frames = config.max_audio_samples // config.audio_encoder_hop_length
        self._tokens_per_sample = n_frames // config.audio_encoder_patch_frames
        if self._tokens_per_sample < 1:
            raise ValueError(
                f"AudioDataset config yields < 1 token per sample "
                f"(max_audio_samples={config.max_audio_samples}, "
                f"hop_length={config.audio_encoder_hop_length}, "
                f"patch_frames={config.audio_encoder_patch_frames})."
            )

        # Iteration state (mirrors TextDataset).
        self._epoch_index = 0
        self._pos_within_epoch = 0
        self._shuffle: Optional[torch.Tensor] = None
        self._build_shuffle_for_epoch(self._epoch_index)

    # -- Public API --

    def next_batch(self) -> dict:
        bs = self.config.batch_size
        if self._shuffle is None:
            self._build_shuffle_for_epoch(self._epoch_index)
        assert self._shuffle is not None

        end_pos = self._pos_within_epoch + bs
        if end_pos > self._n_samples:
            partial = self._shuffle[self._pos_within_epoch :].tolist()
            self._epoch_index += 1
            self._build_shuffle_for_epoch(self._epoch_index)
            self._pos_within_epoch = 0
            rem = bs - len(partial)
            second = self._shuffle[:rem].tolist()
            self._pos_within_epoch = rem
            indices = partial + second
        else:
            indices = self._shuffle[self._pos_within_epoch : end_pos].tolist()
            self._pos_within_epoch = end_pos

        waveforms = [self._load_one(self._audio_paths[i]) for i in indices]
        return {"audio_waveform": torch.stack(waveforms, dim=0)}

    def batch_token_count(self, batch: dict) -> int:
        return int(batch["audio_waveform"].shape[0]) * int(self._tokens_per_sample)

    def tokens_per_pass(self) -> int:
        return int(self._n_samples) * int(self._tokens_per_sample)

    def unique_sample_count(self) -> int:
        return int(self._n_samples)

    def state_dict(self) -> dict:
        return {
            "epoch_index": int(self._epoch_index),
            "pos_within_epoch": int(self._pos_within_epoch),
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch_index = int(state["epoch_index"])
        self._pos_within_epoch = int(state["pos_within_epoch"])
        self._build_shuffle_for_epoch(self._epoch_index)

    # -- Internals --

    def _load_one(self, path: Path) -> torch.Tensor:
        """Load + mono + resample + pad/truncate."""
        waveform, sr = self._torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [samples]
        if sr != self.config.sample_rate:
            resampler = self._torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
        n = waveform.shape[0]
        if n > self.config.max_audio_samples:
            waveform = waveform[: self.config.max_audio_samples]
        elif n < self.config.max_audio_samples:
            pad = torch.zeros(self.config.max_audio_samples - n)
            waveform = torch.cat([waveform, pad])
        return waveform

    def _build_shuffle_for_epoch(self, epoch_index: int) -> None:
        seed = self.config.base_seed ^ (epoch_index * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed & 0x7FFFFFFFFFFFFFFF))
        self._shuffle = torch.randperm(self._n_samples, generator=gen)


# ---------------------------------------------------------------------------
# Vision dataset (COCO)
# ---------------------------------------------------------------------------


@dataclass
class VisionDatasetConfig:
    """Configuration for the COCO vision path.

    root: directory containing .jpg images. For M8 baseline:
        ``Path("E:/data/coco/train2017")``; use ``val2017`` for held-out
        evaluation.
    batch_size: images per batch.
    image_size: square size in pixels. Default 224 matches VisionEncoder.
    patch_size: patch size in pixels. Default 16 matches VisionEncoder
        (224 / 16 = 14, so 14*14 = 196 tokens per image).
    normalization: ``"imagenet"`` applies ImageNet mean/std (recommended);
        ``"none"`` skips. VisionEncoder expects approximately zero-mean
        unit-variance input.
    base_seed: deterministic seed for shuffling.

    **Yields RAW images** -- ``{"image": Tensor[B, 3, H, W]}`` --
    NOT pre-encoded vision_tokens. The VisionEncoder runs inside the
    model's forward.
    """

    root: Path
    batch_size: int
    image_size: int = 224
    patch_size: int = 16
    normalization: str = "imagenet"
    base_seed: int = 42


class VisionDataset:
    """COCO image stream. Yields raw normalized image tensors.

    Indexes .jpg files (flat) from ``config.root`` on construction;
    per-batch lazy-loads via PIL + torchvision transforms. One sample
    per image per epoch in a deterministic shuffled order.
    """

    def __init__(self, config: VisionDatasetConfig):
        # Lazy imports.
        from PIL import Image
        from torchvision import transforms
        self._Image = Image
        self.config = config

        root = Path(config.root)
        if not root.exists():
            raise FileNotFoundError(
                f"VisionDataset root does not exist: {root}. For M8 the "
                f"typical path is E:/data/coco/train2017."
            )
        # COCO train2017/val2017 are flat directories of .jpg files.
        self._image_paths = sorted(root.glob("*.jpg"))
        if not self._image_paths:
            raise ValueError(
                f"VisionDataset found no .jpg files in {root}."
            )
        self._n_samples = len(self._image_paths)

        if config.image_size % config.patch_size != 0:
            raise ValueError(
                f"image_size={config.image_size} not divisible by "
                f"patch_size={config.patch_size}"
            )
        per_side = config.image_size // config.patch_size
        self._tokens_per_sample = per_side * per_side

        if config.normalization == "imagenet":
            self._transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        elif config.normalization == "none":
            self._transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown normalization: {config.normalization!r}")

        # Iteration state.
        self._epoch_index = 0
        self._pos_within_epoch = 0
        self._shuffle: Optional[torch.Tensor] = None
        self._build_shuffle_for_epoch(self._epoch_index)

    # -- Public API --

    def next_batch(self) -> dict:
        bs = self.config.batch_size
        if self._shuffle is None:
            self._build_shuffle_for_epoch(self._epoch_index)
        assert self._shuffle is not None

        end_pos = self._pos_within_epoch + bs
        if end_pos > self._n_samples:
            partial = self._shuffle[self._pos_within_epoch :].tolist()
            self._epoch_index += 1
            self._build_shuffle_for_epoch(self._epoch_index)
            self._pos_within_epoch = 0
            rem = bs - len(partial)
            second = self._shuffle[:rem].tolist()
            self._pos_within_epoch = rem
            indices = partial + second
        else:
            indices = self._shuffle[self._pos_within_epoch : end_pos].tolist()
            self._pos_within_epoch = end_pos

        images = [self._load_one(self._image_paths[i]) for i in indices]
        return {"image": torch.stack(images, dim=0)}

    def batch_token_count(self, batch: dict) -> int:
        return int(batch["image"].shape[0]) * int(self._tokens_per_sample)

    def tokens_per_pass(self) -> int:
        return int(self._n_samples) * int(self._tokens_per_sample)

    def unique_sample_count(self) -> int:
        return int(self._n_samples)

    def state_dict(self) -> dict:
        return {
            "epoch_index": int(self._epoch_index),
            "pos_within_epoch": int(self._pos_within_epoch),
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch_index = int(state["epoch_index"])
        self._pos_within_epoch = int(state["pos_within_epoch"])
        self._build_shuffle_for_epoch(self._epoch_index)

    # -- Internals --

    def _load_one(self, path: Path) -> torch.Tensor:
        with self._Image.open(path) as pil:
            # Some COCO images are grayscale (mode "L"); convert to RGB
            # so the Conv2d patch embedding gets 3 channels.
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            return self._transform(pil)

    def _build_shuffle_for_epoch(self, epoch_index: int) -> None:
        seed = self.config.base_seed ^ (epoch_index * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed & 0x7FFFFFFFFFFFFFFF))
        self._shuffle = torch.randperm(self._n_samples, generator=gen)


# ---------------------------------------------------------------------------
# Multimodal loader
# ---------------------------------------------------------------------------


class MultimodalDataLoaderImpl:
    """Concrete ``MultimodalDataLoader``. Dispatches ``next_batch(modality)``
    to the appropriate per-modality dataset; combines per-modality state
    dicts into a single state dict for resume.

    Per 4.8 2026-06-06: ``state_dict``/``load_state_dict`` round-trip the
    per-modality shuffle position so the without-replacement guarantee
    survives resume.
    """

    def __init__(
        self,
        text: Optional[TextDataset] = None,
        audio: Optional[AudioDataset] = None,
        vision: Optional[VisionDataset] = None,
    ):
        if text is None and audio is None and vision is None:
            raise ValueError("At least one modality dataset must be provided")
        # Note: stubs are still "provided" if the caller wants the
        # runner to crash loudly on first audio/vision sample. To omit
        # the modality entirely, set it to None here and ensure the
        # sampler's corpus_sizes_tokens does not include it.
        self._datasets: dict[str, object] = {}
        if text is not None:
            self._datasets["text"] = text
        if audio is not None:
            self._datasets["audio"] = audio
        if vision is not None:
            self._datasets["vision"] = vision

    # -- Held-out eval support (2026-07-15; duck-typed and optional so
    #    legacy loaders and the audio/vision stubs need no changes) --

    def holdout_batch_count(self, modality: str, batch_size: int) -> int:
        ds = self._datasets.get(modality)
        if ds is None or not hasattr(ds, "holdout_batch_count"):
            return 0
        return int(ds.holdout_batch_count(batch_size))

    def holdout_batches(self, modality: str, batch_size: int):
        ds = self._datasets.get(modality)
        if ds is None or not hasattr(ds, "holdout_batches"):
            return
        yield from ds.holdout_batches(batch_size)

    # -- Protocol implementation --

    def next_batch(self, modality: str) -> dict:
        ds = self._datasets.get(modality)
        if ds is None:
            raise KeyError(
                f"Modality {modality!r} not registered with this loader. "
                f"Registered modalities: {sorted(self._datasets)}"
            )
        return ds.next_batch()  # type: ignore[attr-defined]

    def batch_token_count(self, modality: str, batch: dict) -> int:
        ds = self._datasets.get(modality)
        if ds is None:
            raise KeyError(f"Modality {modality!r} not registered")
        return int(ds.batch_token_count(batch))  # type: ignore[attr-defined]

    def state_dict(self) -> dict:
        return {
            modality: ds.state_dict()  # type: ignore[attr-defined]
            for modality, ds in self._datasets.items()
        }

    def load_state_dict(self, state: dict) -> None:
        for modality, modality_state in state.items():
            ds = self._datasets.get(modality)
            if ds is None:
                # Loader was reconfigured between checkpoint and resume;
                # skip rather than error so a partial resume is still
                # useful.
                continue
            ds.load_state_dict(modality_state)  # type: ignore[attr-defined]

    # -- Convenience for runner config --

    def corpus_sizes_tokens(self) -> dict[str, int]:
        """Returns the per-modality overlap-counted tokens-per-pass dict,
        intended as the value for ``SamplerConfig.corpus_sizes_tokens``.
        Skips modalities whose dataset cannot report a per-pass count
        (stubs).

        **Footgun warning (per 4.8 review 2026-06-06):** silently skipping
        stubs is fine for Gate-1 (audio/vision absent → text-only sampler
        is correct), but once audio/vision are stubs-not-yet-real, this
        method would auto-configure a text-only sampler without warning
        even though the run was intended to be multimodal. Track #10
        replaces the stubs with real impls; until then, prefer building
        the sampler config explicitly rather than auto-deriving it here.
        """
        sizes: dict[str, int] = {}
        for modality, ds in self._datasets.items():
            try:
                sizes[modality] = int(ds.tokens_per_pass())  # type: ignore[attr-defined]
            except NotImplementedError:
                # Stub modality -- the caller must provide the value from
                # elsewhere if they want to include this modality in the
                # sampler weighting.
                continue
        return sizes


# ---------------------------------------------------------------------------
# Filelist parsing
# ---------------------------------------------------------------------------


def _read_filelist(path: Path) -> list[Path]:
    """Read a filelist, ignoring ``#``-comments and blank lines (matches
    the loader contract documented in the M8 brief v0.5 §2)."""
    paths: list[Path] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))
    if not paths:
        raise ValueError(f"Filelist {path} contained no paths")
    return paths


# ---------------------------------------------------------------------------
# Default-config build helper
# ---------------------------------------------------------------------------


def build_text_only_loader(
    source_paths: list[Path | str],
    tokenizer_path: Path,
    batch_size: int,
    seq_len: int = 128,
    stride: int = 64,
    base_seed: int = 42,
) -> MultimodalDataLoaderImpl:
    """Builds a text-only ``MultimodalDataLoaderImpl`` for the Gate-1
    smoke test (per 4.8 build order). Audio and vision are absent; the
    runner must use a ``SamplerConfig`` with only ``text`` in
    ``corpus_sizes_tokens`` so the sampler never picks an unbuilt
    modality.

    For the v0.5 spec baseline corpus (gutenberg_4gb), pass
    ``source_paths=[Path("E:/data/gutenberg_4gb")]``. For the M7
    curriculum subset, use ``source_paths=_read_filelist(...)`` over
    ``corpus_build/m7_filelist.txt``; that is a different corpus than
    the v0.5 spec names and requires explicit decision.
    """
    text = TextDataset(TextDatasetConfig(
        source_paths=list(source_paths),
        tokenizer_path=Path(tokenizer_path),
        batch_size=batch_size,
        seq_len=seq_len,
        stride=stride,
        base_seed=base_seed,
    ))
    return MultimodalDataLoaderImpl(text=text)


def build_multimodal_loader(
    text_source_paths: list[Path | str],
    text_tokenizer_path: Path,
    text_batch_size: int,
    audio_root: Path,
    audio_batch_size: int,
    vision_root: Path,
    vision_batch_size: int,
    text_seq_len: int = 128,
    text_stride: int = 64,
    audio_max_samples: int = 80000,
    vision_image_size: int = 224,
    vision_patch_size: int = 16,
    base_seed: int = 42,
) -> MultimodalDataLoaderImpl:
    """Builds a three-modality ``MultimodalDataLoaderImpl`` for multimodal
    M8 runs and the multimodal smoke. Each per-modality dataset gets a
    slightly offset seed so the three modalities' shuffles do not
    accidentally correlate.

    Paths for the v0.5 §2 baseline (Brian's hardware as of 2026-06-07):
        text:   files / dirs under E:/data/gutenberg_4gb (or m7 subset)
        audio:  E:/data/LibriSpeech/train-clean-100
        vision: E:/data/coco/train2017
    """
    text = TextDataset(TextDatasetConfig(
        source_paths=list(text_source_paths),
        tokenizer_path=Path(text_tokenizer_path),
        batch_size=text_batch_size,
        seq_len=text_seq_len,
        stride=text_stride,
        base_seed=base_seed,
    ))
    audio = AudioDataset(AudioDatasetConfig(
        root=Path(audio_root),
        batch_size=audio_batch_size,
        max_audio_samples=audio_max_samples,
        base_seed=base_seed + 1,
    ))
    vision = VisionDataset(VisionDatasetConfig(
        root=Path(vision_root),
        batch_size=vision_batch_size,
        image_size=vision_image_size,
        patch_size=vision_patch_size,
        base_seed=base_seed + 2,
    ))
    return MultimodalDataLoaderImpl(text=text, audio=audio, vision=vision)
