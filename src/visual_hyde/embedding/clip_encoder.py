"""
CLIP / SigLIP encoder for both images and text.

Wraps `open-clip-torch` to provide a unified interface for encoding:
  - PIL images or image file paths → float32 vectors
  - text strings → float32 vectors

Supports batch encoding with configurable batch size and device.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from visual_hyde.config import EmbeddingModel, EmbeddingSettings, get_settings
from visual_hyde.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model name mapping: our enum → open_clip model/pretrained names
# ---------------------------------------------------------------------------

_MODEL_MAP: dict[EmbeddingModel, tuple[str, str]] = {
    EmbeddingModel.CLIP_VIT_B32: ("ViT-B-32", "openai"),
    EmbeddingModel.CLIP_VIT_L14: ("ViT-L-14", "openai"),
    EmbeddingModel.SIGLIP_BASE: ("ViT-B-16-SigLIP", "webli"),
    EmbeddingModel.SIGLIP_SO400M: ("ViT-SO400M-14-SigLIP", "webli"),
}


class CLIPEncoder:
    """
    Encodes images and text using CLIP or SigLIP via open-clip-torch.

    Usage:
        encoder = CLIPEncoder()
        img_vecs = encoder.encode_images([path1, path2, ...])  # shape (N, D)
        txt_vecs = encoder.encode_texts(["query 1", "query 2"])  # shape (N, D)
    """

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self._settings = settings or get_settings().embedding
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenize: Any = None

    def _load_model(self) -> None:
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "open-clip-torch is required: uv add open-clip-torch"
            )

        model_name, pretrained = _MODEL_MAP[self._settings.model]
        logger.info(f"Loading CLIP model: {model_name} ({pretrained})")

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self._settings.device,
        )
        self._model.eval()
        self._tokenize = open_clip.get_tokenizer(model_name)
        logger.info(f"CLIP model loaded on device: {self._settings.device}")

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_images(
        self,
        images: list[Path | Image.Image],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of images into normalized float32 vectors.

        Args:
            images: List of file paths (Path) or PIL Images.
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape (N, embed_dim), dtype float32.
        """
        self._load_model()
        batch_size = self._settings.batch_size
        all_vecs: list[np.ndarray] = []

        batches = [images[i: i + batch_size] for i in range(0, len(images), batch_size)]
        iterator = tqdm(batches, desc="Encoding images") if show_progress else batches

        with torch.no_grad():
            for batch in iterator:
                pil_batch = [
                    Image.open(img).convert("RGB") if isinstance(img, Path) else img.convert("RGB")
                    for img in batch
                ]
                tensors = torch.stack(
                    [self._preprocess(img) for img in pil_batch]
                ).to(self._settings.device)

                features = self._model.encode_image(tensors)

                if self._settings.normalize:
                    features = features / features.norm(dim=-1, keepdim=True)

                all_vecs.append(features.cpu().float().numpy())

        return np.concatenate(all_vecs, axis=0)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode text strings into normalized float32 vectors.

        Args:
            texts: List of query strings.
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape (N, embed_dim), dtype float32.
        """
        self._load_model()
        batch_size = self._settings.batch_size
        all_vecs: list[np.ndarray] = []

        batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
        iterator = tqdm(batches, desc="Encoding texts") if show_progress else batches

        with torch.no_grad():
            for batch in iterator:
                tokens = self._tokenize(batch).to(self._settings.device)
                features = self._model.encode_text(tokens)

                if self._settings.normalize:
                    features = features / features.norm(dim=-1, keepdim=True)

                all_vecs.append(features.cpu().float().numpy())

        return np.concatenate(all_vecs, axis=0)

    @property
    def embed_dim(self) -> int:
        """Return embedding dimension for the loaded model."""
        self._load_model()
        return self._model.visual.output_dim


from typing import Any
