"""
Pattern B: Hypothetical chart generation via Nano Banana (Google Gemini).

Uses the Gemini image generation API to produce a photorealistic chart image
directly from the text query. The generated image is then embedded via CLIP.

Falls back gracefully if the API key is not configured.
"""

from __future__ import annotations

import base64
from pathlib import Path

from visual_hyde.config import GenerationSettings, get_settings
from visual_hyde.generation.prompts import build_image_gen_prompt
from visual_hyde.logging import get_logger
from visual_hyde.types import GeneratedChart

logger = get_logger(__name__)


class NanoBananaGenerator:
    """
    Generates hypothetical chart images using Google Gemini (Nano Banana)
    image generation API.

    Requires:
      - GEMINI_API_KEY set in environment / .env (VH_GEN_GEMINI_API_KEY)
      - google-generativeai package: uv add google-generativeai

    Args:
        settings: GenerationSettings (defaults to global config).
        cache_dir: Directory for caching generated PNGs.
    """

    def __init__(
        self,
        settings: GenerationSettings | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings().generation
        cfg_paths = get_settings().paths
        self._cache_dir = cache_dir or cfg_paths.generated_charts_dir / "nano_banana"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "Install google-generativeai: uv add google-generativeai"
            )

        api_key = self._settings.gemini_api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured. "
                "Set VH_GEN_GEMINI_API_KEY in .env"
            )

        genai.configure(api_key=api_key)
        self._client = genai
        return self._client

    def generate(self, query_id: str, query: str) -> GeneratedChart:
        """
        Generate a hypothetical chart image using Nano Banana.

        Returns GeneratedChart. On failure, returns generation_ok=False with
        error details so the experiment pipeline can record and continue.
        """
        save_path = self._cache_dir / f"{query_id}.png"

        if save_path.exists():
            return GeneratedChart(
                query_id=query_id,
                image_path=save_path,
                method="nano_banana",
                generation_ok=True,
            )

        try:
            self._generate_and_save(query, save_path)
            return GeneratedChart(
                query_id=query_id,
                image_path=save_path,
                method="nano_banana",
                generation_ok=True,
            )
        except Exception as e:
            logger.warning(f"Nano Banana generation failed for '{query_id}': {e}")
            return GeneratedChart(
                query_id=query_id,
                image_path=save_path,  # may not exist
                method="nano_banana",
                generation_ok=False,
                error=str(e),
            )

    def generate_batch(
        self,
        query_ids: list[str],
        queries: list[str],
        show_progress: bool = True,
    ) -> list[GeneratedChart]:
        from tqdm import tqdm

        pairs = zip(query_ids, queries)
        if show_progress:
            pairs = tqdm(list(pairs), desc="Generating charts (Nano Banana)")

        return [self.generate(qid, q) for qid, q in pairs]

    def _generate_and_save(self, query: str, save_path: Path) -> None:
        genai = self._get_client()
        prompt = build_image_gen_prompt(query)

        model = genai.GenerativeModel(self._settings.image_gen_model)
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "image/png"},
        )

        # Extract image bytes from response
        img_bytes = self._extract_image_bytes(response)
        if img_bytes is None:
            raise ValueError("No image data returned from Gemini API")

        save_path.write_bytes(img_bytes)
        logger.debug(f"Saved Nano Banana chart: {save_path}")

    @staticmethod
    def _extract_image_bytes(response: Any) -> bytes | None:
        """
        Extract raw PNG bytes from a Gemini API response.

        The Gemini image generation API may return image data in different
        fields depending on API version. This method handles both cases.
        """
        # Attempt 1: inline_data (standard image generation response)
        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return part.inline_data.data
        except (AttributeError, IndexError):
            pass

        # Attempt 2: base64-encoded text in response
        try:
            text = response.text
            if text and len(text) > 100:
                return base64.b64decode(text)
        except Exception:
            pass

        return None


from typing import Any
