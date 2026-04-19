"""
Text-Direct retriever baseline.

Encodes the raw text query with CLIP's text encoder and searches the
image corpus directly (cross-modal search). This is the primary baseline
that Visual HyDE aims to outperform on trend/pattern queries.
"""

from __future__ import annotations

import numpy as np

from visual_hyde.embedding.clip_encoder import CLIPEncoder
from visual_hyde.embedding.corpus_index import CorpusIndex
from visual_hyde.logging import get_logger
from visual_hyde.retrieval.base import BaseRetriever
from visual_hyde.types import QueryItem, RetrievalOutput

logger = get_logger(__name__)


class TextDirectRetriever(BaseRetriever):
    """
    Text-Direct (CLIP): raw query text → CLIP text embedding → corpus search.

    The canonical cross-modal baseline that suffers from the modality gap
    on shape/pattern queries.
    """

    def __init__(
        self,
        index: CorpusIndex,
        encoder: CLIPEncoder | None = None,
    ) -> None:
        self._index = index
        self._encoder = encoder or CLIPEncoder()

    @property
    def name(self) -> str:
        return "text_direct"

    def retrieve_one(self, query: QueryItem, top_k: int = 10) -> RetrievalOutput:
        vec = self._encoder.encode_texts([query.text], show_progress=False)[0]
        results = self._index.search(vec, top_k=top_k)
        return RetrievalOutput(query_id=query.id, results=results)

    def retrieve_batch(
        self,
        queries: list[QueryItem],
        top_k: int = 10,
        show_progress: bool = True,
    ) -> list[RetrievalOutput]:
        """Efficient batch: encode all texts at once, then batch FAISS search."""
        texts = [q.text for q in queries]
        all_vecs = self._encoder.encode_texts(texts, show_progress=show_progress)
        all_results_lists = self._index.search_batch(all_vecs, top_k=top_k)

        return [
            RetrievalOutput(query_id=q.id, results=results)
            for q, results in zip(queries, all_results_lists)
        ]
