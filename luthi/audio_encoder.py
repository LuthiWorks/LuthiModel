"""Audio encoder for multimodal LuthiModel.

Converts raw audio waveforms to a sequence of d_model-dimensional tokens
suitable for the shared living weight trunk. The pipeline:

    waveform → mel spectrogram (CPU) → patch embedding → projection → d_model tokens

The mel spectrogram is always computed on CPU to avoid DirectML compatibility
issues with torch.stft. The patch embedding and projection run on the model's
device. This is safe because mel computation is a data preprocessing step,
not a bottleneck.
"""

import torch
import torch.nn as nn
import torchaudio


class AudioEncoder(nn.Module):
    """Encodes raw audio waveforms into d_model token sequences.

    Each output token represents a patch of the mel spectrogram covering
    the full frequency range x patch_frames time steps. At default settings
    (16kHz, hop=160, patch_frames=16), each token covers ~160ms of audio,
    yielding ~6.25 tokens per second.

    Args:
        d_model: Output dimension (must match the trunk model).
        sample_rate: Expected audio sample rate.
        n_mels: Number of mel frequency bins.
        hop_length: STFT hop length in samples.
        n_fft: STFT window size in samples.
        patch_frames: Number of spectrogram frames per patch token.
        max_audio_tokens: Maximum number of output tokens (for positional embedding).
    """

    def __init__(
        self,
        d_model: int,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
        n_fft: int = 400,
        patch_frames: int = 16,
        max_audio_tokens: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.patch_frames = patch_frames
        self.max_audio_tokens = max_audio_tokens

        # Mel spectrogram — always runs on CPU
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Patch embedding: treats spectrogram as 1-channel image, extracts
        # non-overlapping patches of (n_mels, patch_frames). Each patch
        # covers the full frequency range x patch_frames time steps.
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(n_mels, patch_frames),
            stride=(n_mels, patch_frames),
        )

        # Post-patch projection and normalization
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

        # Learned positional embedding for audio token positions
        self.pos_embed = nn.Embedding(max_audio_tokens, d_model)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode raw audio waveform to d_model token sequence.

        Args:
            waveform: [batch, samples] raw audio at self.sample_rate.

        Returns:
            [batch, n_tokens, d_model] audio token embeddings.
        """
        target_device = self.proj.weight.device

        # Mel spectrogram on CPU (avoids DirectML torch.stft issues)
        wav_cpu = waveform.detach().cpu().float()

        # Pad to at least n_fft samples so torch.stft doesn't error
        if wav_cpu.shape[-1] < self.n_fft:
            pad_needed = self.n_fft - wav_cpu.shape[-1]
            wav_cpu = torch.nn.functional.pad(wav_cpu, (0, pad_needed))

        mel = self.mel_spec(wav_cpu)  # [batch, n_mels, n_frames]

        # Truncate frames to a multiple of patch_frames so Conv2d divides evenly
        n_frames = mel.shape[-1]
        usable_frames = (n_frames // self.patch_frames) * self.patch_frames
        if usable_frames == 0:
            # Audio too short for even one patch — return a single zero token
            batch = waveform.shape[0]
            return torch.zeros(batch, 1, self.d_model, device=target_device)
        mel = mel[:, :, :usable_frames]

        # Move to model device for the trainable layers
        mel = mel.to(target_device)

        # Add channel dimension for Conv2d: [batch, 1, n_mels, n_frames]
        mel = mel.unsqueeze(1)

        # Patch embedding: [batch, d_model, 1, n_patches]
        patches = self.patch_embed(mel)

        # Reshape to sequence: [batch, n_patches, d_model]
        patches = patches.squeeze(2).transpose(1, 2)

        n_tokens = patches.shape[1]

        # Projection + positional embedding + layer norm
        tokens = self.proj(patches)
        positions = torch.arange(n_tokens, device=target_device).unsqueeze(0)
        tokens = tokens + self.pos_embed(positions)
        tokens = self.ln(tokens)

        return tokens

    def n_tokens_for_samples(self, n_samples: int) -> int:
        """Calculate how many tokens a given number of audio samples produces."""
        n_frames = n_samples // self.hop_length
        return n_frames // self.patch_frames
