"""
FAISS-backed vector index for Visual HyDE corpus.

Responsibilities:
  - Build a FAISS flat inner-product index from CorpusItem embeddings
  - Persist index + metadata to disk
  - Load existing index
  - Provide `search(query_vector, top_k)` → List[SearchResult]

Design notes:
  - We store the corpus ID list separately (FAISS only knows integer indices).
  - All vectors are assumed L2-normalized, so inner product == cosine similarity.
  - For large corpora (>1M items), consider using IVFFlat or HNSW instead of Flat.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from visual_hyde.logging import get_logger
from visual_hyde.types import CorpusItem, EmbeddingRecord, SearchResult

logger = get_logger(__name__)


class CorpusIndex:
    """
    Wraps a FAISS IndexFlatIP for Visual HyDE corpus search.

    Usage:
        index = CorpusIndex()
        index.build(embedding_records)
        index.save(path)

        # Later:
        index = CorpusIndex.load(path)
        results = index.search(query_vec, top_k=10)
    """

    def __init__(self) -> None:
        self._index: Any = None           # faiss.Index
        self._id_list: list[str] = []     # Maps FAISS integer position → corpus_id
        self._embed_dim: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, records: list[EmbeddingRecord]) -> None:
        """
        Build FAISS index from a list of EmbeddingRecords.

        Args:
            records: Corpus embeddings (all must have same dimension).
        """
        try:
            import faiss
        except ImportError:
            raise RuntimeError("Install faiss-cpu: uv add faiss-cpu")

        if not records:
            raise ValueError("Cannot build index from empty record list")

        self._embed_dim = records[0].vector.shape[0]
        vecs = np.stack([r.vector for r in records]).astype(np.float32)
        self._id_list = [r.corpus_id for r in records]

        logger.info(
            f"Building FAISS IndexFlatIP: {len(records)} vectors, dim={self._embed_dim}"
        )
        self._index = faiss.IndexFlatIP(self._embed_dim)
        self._index.add(vecs)
        logger.info("Index built successfully")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Search the index for the nearest neighbors of a query vector.

        Args:
            query_vector: 1D float32 array of shape (embed_dim,).
            top_k:        Number of results to return.

        Returns:
            List of SearchResult ordered by rank (best first).
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        qvec = query_vector.astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(qvec, top_k)

        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            results.append(
                SearchResult(
                    corpus_id=self._id_list[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    def search_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
    ) -> list[list[SearchResult]]:
        """
        Search for multiple queries by looping over individual FAISS calls.

        Using a per-query loop avoids FAISS batch-search crashes on Apple
        Silicon (aarch64) where the C-extension can segfault with large batches.

        Args:
            query_vectors: 2D float32 array of shape (N, embed_dim),
                           or list of 1D arrays.
            top_k:         Number of results per query.

        Returns:
            List of N result lists.
        """
        if self._index is None:
            raise RuntimeError("Index not built.")

        vecs = np.asarray(query_vectors, dtype=np.float32)
        return [self.search(vec, top_k=top_k) for vec in vecs]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        """
        Save index to disk.

        Writes:
          <directory>/index.faiss  — FAISS binary index
          <directory>/metadata.json — id_list and embed_dim
        """
        try:
            import faiss
        except ImportError:
            raise RuntimeError("Install faiss-cpu")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        meta = {
            "embed_dim": self._embed_dim,
            "id_list": self._id_list,
            "size": len(self._id_list),
        }
        (directory / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info(f"Index saved to {directory} ({len(self._id_list)} items)")

    @classmethod
    def load(cls, directory: Path) -> "CorpusIndex":
        """Load a previously saved index from disk."""
        try:
            import faiss
        except ImportError:
            raise RuntimeError("Install faiss-cpu")

        directory = Path(directory)
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        instance = cls()
        instance._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        instance._id_list = meta["id_list"]
        instance._embed_dim = meta["embed_dim"]

        logger.info(
            f"Index loaded from {directory}: {len(instance._id_list)} items, "
            f"dim={instance._embed_dim}"
        )
        return instance

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._id_list)

    @property
    def is_built(self) -> bool:
        return self._index is not None


from typing import Any
