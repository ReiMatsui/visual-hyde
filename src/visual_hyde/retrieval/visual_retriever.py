"""
Visual HyDE retriever (core contribution).

Pipeline per query:
  1. Generate hypothetical chart image (matplotlib or Nano Banana)
  2. Encode the generated image with CLIP
  3. Search the corpus FAISS index using the image vector
  4. Return ranked SearchResults

The retriever is initialized with a pre-built CorpusIndex and a generator.
"""

from __future__ import annotations

from pathlib import Path

from visual_hyde.config import GenerationMethod, get_settings
from visual_hyde.embedding.clip_encoder import CLIPEncoder
from visual_hyde.embedding.corpus_index import CorpusIndex
from visual_hyde.generation import get_generator
from visual_hyde.logging import get_logger
from visual_hyde.retrieval.base import BaseRetriever
from visual_hyde.types import GeneratedChart, QueryItem, RetrievalOutput, SearchResult

logger = get_logger(__name__)


class VisualHyDERetriever(BaseRetriever):
    """
    Visual HyDE retriever: text query → hypothetical chart → image search.

    Args:
        index:             Pre-built CorpusIndex (FAISS).
        generation_method: MATPLOTLIB or NANO_BANANA.
        encoder:           CLIPEncoder instance (shared across retrievers).
        top_k:             Default number of results.
    """

    def __init__(
        self,
        index: CorpusIndex,
        generation_method: GenerationMethod = GenerationMethod.MATPLOTLIB,
        encoder: CLIPEncoder | None = None,
    ) -> None:
        self._index = index
        self._method = generation_method
        self._generator = get_generator(generation_method)
        self._encoder = encoder or CLIPEncoder()

    @property
    def name(self) -> str:
        return f"visual_hyde_{self._method.value}"

    @property
    def generation_failures(self) -> list[dict]:
        """Return recorded chart generation failures from the underlying generator."""
        return getattr(self._generator, "failures", [])

    @property
    def generation_failure_rate(self) -> float:
        return getattr(self._generator, "failure_rate", 0.0)

    def retrieve_one(self, query: QueryItem, top_k: int = 10) -> RetrievalOutput:
        # Step 1: Generate hypothetical chart
        chart: GeneratedChart = self._generator.generate(query.id, query.text)

        if not chart.generation_ok:
            logger.debug(
                f"Chart generation failed for {query.id}: {chart.error}. "
                "Using fallback image."
            )

        # Step 2: Encode the generated image
        query_vec = self._encoder.encode_images(
            [chart.image_path], show_progress=False
        )[0]

        # Step 3: Search corpus
        results = self._index.search(query_vec, top_k=top_k)

        return RetrievalOutput(query_id=query.id, results=results)

    def retrieve_batch(
        self,
        queries: list[QueryItem],
        top_k: int = 10,
        show_progress: bool = True,
    ) -> list[RetrievalOutput]:
        """
        Batch retrieval. Generates all charts first (with caching), then
        does a single batched FAISS search.
        """
        from tqdm import tqdm

        # Step 1: Generate all charts (cached results are returned instantly)
        query_ids = [q.id for q in queries]
        query_texts = [q.text for q in queries]
        charts = self._generator.generate_batch(query_ids, query_texts, show_progress)

        # Step 2: Encode all generated images in one batch
        image_paths = [c.image_path for c in charts]
        all_vecs = self._encoder.encode_images(image_paths, show_progress=show_progress)

        # Step 3: Batch FAISS search
        import numpy as np

        all_results_lists = self._index.search_batch(all_vecs, top_k=top_k)

        return [
            RetrievalOutput(query_id=q.id, results=results)
            for q, results in zip(queries, all_results_lists)
        ]
