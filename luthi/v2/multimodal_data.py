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
        n = self._tokens.numel()
        max_start = n - config.seq_len
        if max_start < 0:
            raise ValueError(
                f"Corpus has {n} tokens, less than seq_len={config.seq_len}"
            )
        self._n_sequences = (max_start // config.stride) + 1

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


class AudioDataset(_ModalityStub):
    """LibriSpeech waveform stream (NOT YET IMPLEMENTED).

    When implemented, will yield ``{"audio_waveform": Tensor[B, samples]}``
    of raw 16 kHz audio. The AudioEncoder in the model will run the mel
    spectrogram + patch embedding inside its forward, so the encoder
    trains. Do NOT pre-compute audio_tokens here -- bypassing the
    encoder during training is the gradient trap 4.8 flagged.
    """

    def __init__(self):
        super().__init__("audio")


class VisionDataset(_ModalityStub):
    """COCO image stream (NOT YET IMPLEMENTED).

    When implemented, will yield ``{"image": Tensor[B, 3, H, W]}`` of
    ImageNet-normalized RGB images at 224x224 (matching VisionEncoder
    defaults). The VisionEncoder will produce 196 patch tokens internally.
    Do NOT pre-compute vision_tokens here.
    """

    def __init__(self):
        super().__init__("vision")


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
