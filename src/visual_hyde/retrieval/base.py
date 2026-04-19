"""
Abstract base class for all retrievers.

Every retrieval strategy (text-direct, visual-hyde, hybrid, etc.) must
implement this interface so experiment scripts can swap them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from visual_hyde.types import QueryItem, RetrievalOutput


class BaseRetriever(ABC):
    """
    Protocol for all Visual HyDE retrievers.

    Subclasses must implement `retrieve_one` and can optionally override
    `retrieve_batch` for efficiency.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier string used in result tables (e.g. 'visual_hyde_matplotlib')."""
        ...

    @abstractmethod
    def retrieve_one(self, query: QueryItem, top_k: int = 10) -> RetrievalOutput:
        """
        Retrieve top-k results for a single query.

        Args:
            query:  The query to process.
            top_k:  Number of results to return.

        Returns:
            RetrievalOutput with ranked SearchResults.
        """
        ...

    def retrieve_batch(
        self,
        queries: list[QueryItem],
        top_k: int = 10,
        show_progress: bool = True,
    ) -> list[RetrievalOutput]:
        """
        Retrieve for a list of queries. Default: loops over retrieve_one.
        Override for batched efficiency if the retriever supports it.
        """
        from tqdm import tqdm

        iterator = tqdm(queries, desc=f"Retrieving [{self.name}]") if show_progress else queries
        return [self.retrieve_one(q, top_k) for q in iterator]
