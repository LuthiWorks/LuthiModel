"""COCO Captions dataset for vision-text paired training.

Handles COCO-format datasets where images are stored as JPEGs and captions
are in a JSON annotations file. Returns fixed-size (image, text) pairs
suitable for the MultimodalLuthiLM.

COCO directory structure:
    coco/
        train2017/          (118,287 images)
        val2017/            (5,000 images)
        annotations/
            captions_train2017.json
            captions_val2017.json
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class COCOCaptionDataset(Dataset):
    """Image-text paired dataset from COCO Captions.

    Each item returns a dict with:
        image: [3, image_size, image_size] normalized RGB tensor
        text_input: [max_text_tokens] encoded text tokens (for prediction)
        text_target: [max_text_tokens] shifted tokens (prediction target)

    One random caption is selected per image per call, so different epochs
    see different caption pairings.

    Args:
        image_dir: Path to image directory (e.g., coco/train2017).
        annotations_file: Path to captions JSON
            (e.g., coco/annotations/captions_train2017.json).
        tokenizer: Text tokenizer with encode() method.
        image_size: Output image size (square, images are resized).
        max_text_tokens: Fixed text sequence length for next-token prediction.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotations_file: str | Path,
        tokenizer,
        image_size: int = 224,
        max_text_tokens: int = 128,
    ):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens

        # Load annotations
        with open(annotations_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build image_id -> filename mapping
        id_to_filename = {}
        for img_info in data["images"]:
            id_to_filename[img_info["id"]] = img_info["file_name"]

        # Build image_id -> list of captions
        id_to_captions: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            caption = ann["caption"].strip().lower()
            if img_id not in id_to_captions:
                id_to_captions[img_id] = []
            id_to_captions[img_id].append(caption)

        # Build entries: (image_path, [captions]) for images that exist
        self.entries: list[tuple[Path, list[str]]] = []
        for img_id, captions in sorted(id_to_captions.items()):
            filename = id_to_filename.get(img_id)
            if filename is None:
                continue
            img_path = self.image_dir / filename
            if img_path.exists():
                self.entries.append((img_path, captions))

        if not self.entries:
            raise ValueError(
                f"No image-caption pairs found. "
                f"Image dir: {self.image_dir}, "
                f"Annotations: {annotations_file}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, captions = self.entries[idx]

        # --- Load and prepare image ---
        image = self._load_image(img_path)

        # --- Select one random caption ---
        caption = random.choice(captions)

        # --- Encode text ---
        token_ids = self.tokenizer.encode(caption)

        # Pad or truncate to max_text_tokens + 1 (need input + target)
        target_len = self.max_text_tokens + 1
        if len(token_ids) > target_len:
            token_ids = token_ids[:target_len]
        else:
            token_ids = token_ids + [0] * (target_len - len(token_ids))

        tokens = torch.tensor(token_ids, dtype=torch.long)
        text_input = tokens[:-1]
        text_target = tokens[1:]

        return {
            "image": image,
            "text_input": text_input,
            "text_target": text_target,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image, resize, convert to normalized tensor.

        Returns:
            [3, image_size, image_size] float32 tensor with ImageNet
            normalization applied.
        """
        img = Image.open(path).convert("RGB")

        # Resize to square (center crop would be better but this is simpler
        # and works fine for training)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor: [3, H, W] in [0, 1]
        tensor = torch.tensor(
            list(img.getdata()),
            dtype=torch.float32,
        ).reshape(self.image_size, self.image_size, 3)
        tensor = tensor.permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor
