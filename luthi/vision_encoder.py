"""Vision encoder for multimodal LuthiModel.

Converts RGB images to a sequence of d_model-dimensional tokens suitable for
the shared living weight trunk. The pipeline:

    image → patch embedding (Conv2d) → projection → d_model tokens

This follows the same design philosophy as the AudioEncoder — modality-specific
encoding into a shared representation space. At default settings (224x224 image,
16x16 patches), the encoder produces 196 tokens per image.
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Encodes RGB images into d_model token sequences.

    Each output token represents a non-overlapping patch of the image.
    At default settings (224x224, patch_size=16), each token covers a
    16x16 pixel region, yielding 14x14 = 196 tokens per image.

    Args:
        d_model: Output dimension (must match the trunk model).
        image_size: Expected input image size (square). Images should be
            resized to this before encoding.
        patch_size: Size of each square patch in pixels.
        in_channels: Number of input channels (3 for RGB).
        max_vision_tokens: Maximum number of output tokens (for positional
            embedding). Should be >= (image_size / patch_size)^2.
    """

    def __init__(
        self,
        d_model: int,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        max_vision_tokens: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_vision_tokens = max_vision_tokens

        self.n_patches = (image_size // patch_size) ** 2

        # Patch embedding: non-overlapping patches via strided convolution
        # Input: [batch, 3, H, W] → Output: [batch, d_model, H/patch, W/patch]
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

        # Post-patch projection and normalization
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

        # Learned positional embedding for patch positions
        self.pos_embed = nn.Embedding(max_vision_tokens, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to d_model token sequence.

        Args:
            images: [batch, 3, image_size, image_size] normalized RGB images.
                Values should be normalized to approximately zero mean, unit
                variance (e.g., ImageNet normalization).

        Returns:
            [batch, n_patches, d_model] vision token embeddings.
        """
        # Patch embedding: [batch, d_model, h_patches, w_patches]
        patches = self.patch_embed(images)

        # Reshape to sequence: [batch, n_patches, d_model]
        patches = patches.flatten(2).transpose(1, 2)

        n_tokens = patches.shape[1]

        # Projection + positional embedding + layer norm
        tokens = self.proj(patches)
        positions = torch.arange(n_tokens, device=images.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(positions)
        tokens = self.ln(tokens)

        return tokens

    def n_tokens_for_image(self) -> int:
        """Calculate how many tokens an image produces at current settings."""
        return self.n_patches
