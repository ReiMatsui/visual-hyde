"""
RRF Hybrid retriever.

Combines visual retrieval and text retrieval scores using
Reciprocal Rank Fusion (RRF):

    rrf_score(d) = α / (k + visual_rank(d)) + (1-α) / (k + text_rank(d))

where:
  α  ∈ [0, 1]  — visual weight (0 = text-only, 1 = visual-only)
  k             — RRF smoothing constant (default 60)

Reference: Cormack et al. (2009) "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods."
"""

from __future__ import annotations

from visual_hyde.logging import get_logger
from visual_hyde.retrieval.base import BaseRetriever
from visual_hyde.types import QueryItem, RetrievalOutput, SearchResult

logger = get_logger(__name__)

_DEFAULT_K = 60


class HybridRRFRetriever(BaseRetriever):
    """
    RRF Hybrid: combines visual_retriever and text_retriever results.

    Args:
        visual_retriever: Any BaseRetriever that returns image-based results.
        text_retriever:   TextDirectRetriever (or similar text-based retriever).
        alpha:            Visual weight in [0, 1].
        rrf_k:            RRF smoothing constant.
    """

    def __init__(
        self,
        visual_retriever: BaseRetriever,
        text_retriever: BaseRetriever,
        alpha: float = 0.5,
        rrf_k: int = _DEFAULT_K,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._visual = visual_retriever
        self._text = text_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k

    @property
    def name(self) -> str:
        return f"hybrid_rrf_a{self.alpha:.1f}"

    def retrieve_one(self, query: QueryItem, top_k: int = 10) -> RetrievalOutput:
        # Retrieve independently from both systems (with larger top_k for fusion)
        fetch_k = max(top_k * 3, 50)

        visual_out = self._visual.retrieve_one(query, top_k=fetch_k)
        text_out = self._text.retrieve_one(query, top_k=fetch_k)

        merged = self._fuse(visual_out.results, text_out.results, top_k)
        return RetrievalOutput(query_id=query.id, results=merged)

    def retrieve_batch(
        self,
        queries: list[QueryItem],
        top_k: int = 10,
        show_progress: bool = True,
    ) -> list[RetrievalOutput]:
        fetch_k = max(top_k * 3, 50)

        visual_outputs = self._visual.retrieve_batch(queries, top_k=fetch_k, show_progress=show_progress)
        text_outputs = self._text.retrieve_batch(queries, top_k=fetch_k, show_progress=False)

        return [
            RetrievalOutput(
                query_id=q.id,
                results=self._fuse(v.results, t.results, top_k),
            )
            for q, v, t in zip(queries, visual_outputs, text_outputs)
        ]

    def _fuse(
        self,
        visual_results: list[SearchResult],
        text_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Compute RRF scores and return top_k fused results.
        """
        scores: dict[str, float] = {}

        # Visual contribution
        for res in visual_results:
            scores[res.corpus_id] = scores.get(res.corpus_id, 0.0)
            scores[res.corpus_id] += self.alpha / (self.rrf_k + res.rank)

        # Text contribution
        for res in text_results:
            scores[res.corpus_id] = scores.get(res.corpus_id, 0.0)
            scores[res.corpus_id] += (1.0 - self.alpha) / (self.rrf_k + res.rank)

        # Sort by descending RRF score and take top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(corpus_id=cid, score=score, rank=rank)
            for rank, (cid, score) in enumerate(sorted_items, start=1)
        ]
