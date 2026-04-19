"""
ColPali baseline retriever — stub with pre-computed result loading.

Background
----------
ColPali (Faysse et al., 2024) is a state-of-the-art document retrieval model
that combines a Vision Language Model (PaliGemma) with a late-interaction
scoring mechanism similar to ColBERT.  Running ColPali inference requires a
large GPU (≥24 GB VRAM) and the `vidore/colpali-engine` library.

Integration strategy
--------------------
Rather than running inference inline, this retriever:

1. Loads **pre-computed result files** (JSONL, one JSON object per query) from
   disk when they are available.
2. Returns an **empty RetrievalOutput** with a logged warning when no
   pre-computed results are found, so experiments can still run (just without
   ColPali scores).

To generate the pre-computed files, see `run_colpali_inference()` below and
the script at ``scripts/precompute_colpali.py``.

Pre-computed JSONL format
-------------------------
Each line is a JSON object::

    {
        "query_id": "chartqa_001",
        "results": [
            {"corpus_id": "chartqa_042", "score": 0.873, "rank": 1},
            {"corpus_id": "chartqa_017", "score": 0.801, "rank": 2},
            ...
        ]
    }
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from visual_hyde.logging import get_logger
from visual_hyde.retrieval.base import BaseRetriever
from visual_hyde.types import QueryItem, RetrievalOutput, SearchResult

logger = get_logger(__name__)


class ColPaliRetriever(BaseRetriever):
    """
    ColPali baseline — serves pre-computed results from a JSONL file.

    Args:
        results_path: Path to the pre-computed results JSONL file.
                      If ``None`` or the file does not exist, every call to
                      ``retrieve_one`` will return an empty ``RetrievalOutput``
                      with a warning.
    """

    def __init__(self, results_path: Path | str | None = None) -> None:
        self._results_path = Path(results_path) if results_path else None
        self._cache: dict[str, list[SearchResult]] | None = None

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Retriever identifier used in result tables."""
        return "colpali"

    def retrieve_one(self, query: QueryItem, top_k: int = 10) -> RetrievalOutput:
        """
        Return pre-computed ColPali results for a single query.

        If pre-computed results are unavailable (file missing or query ID not
        found), an empty ``RetrievalOutput`` is returned and a warning is
        logged.

        Args:
            query:  The query to look up.
            top_k:  Maximum number of results to return.

        Returns:
            RetrievalOutput with ranked SearchResults (may be empty).
        """
        cache = self._load_cache()

        if cache is None:
            _warn_no_results(query.id, reason="no pre-computed results file is available")
            return RetrievalOutput(query_id=query.id, results=[])

        if query.id not in cache:
            _warn_no_results(query.id, reason=f"query ID not found in {self._results_path}")
            return RetrievalOutput(query_id=query.id, results=[])

        results = cache[query.id][:top_k]
        # Re-assign ranks in case top_k is smaller than the stored list
        trimmed = [
            SearchResult(corpus_id=r.corpus_id, score=r.score, rank=i + 1)
            for i, r in enumerate(results)
        ]
        return RetrievalOutput(query_id=query.id, results=trimmed)

    # ------------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, list[SearchResult]] | None:
        """
        Load the JSONL file into memory on first access (lazy, cached).

        Returns:
            Mapping from query_id → sorted SearchResult list, or ``None`` if
            the file is unavailable.
        """
        if self._cache is not None:
            return self._cache

        if self._results_path is None or not self._results_path.exists():
            logger.warning(
                "[ColPali] Pre-computed results file not found: %s. "
                "Run `scripts/precompute_colpali.py` to generate it.",
                self._results_path,
            )
            return None

        logger.info("[ColPali] Loading pre-computed results from %s", self._results_path)
        cache: dict[str, list[SearchResult]] = {}

        with self._results_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    query_id: str = obj["query_id"]
                    results = [
                        SearchResult(
                            corpus_id=r["corpus_id"],
                            score=float(r["score"]),
                            rank=int(r["rank"]),
                        )
                        for r in obj["results"]
                    ]
                    # Sort by rank to guarantee ordering
                    results.sort(key=lambda r: r.rank)
                    cache[query_id] = results
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "[ColPali] Skipping malformed line %d in %s: %s",
                        line_no,
                        self._results_path,
                        exc,
                    )

        logger.info("[ColPali] Loaded results for %d queries", len(cache))
        self._cache = cache
        return self._cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warn_no_results(query_id: str, reason: str) -> None:
    """Emit a consistent warning when pre-computed results cannot be served."""
    msg = (
        f"[ColPali] Returning empty results for query {query_id!r}: {reason}. "
        "To include ColPali in your experiment, generate pre-computed results "
        "with `run_colpali_inference()` or `scripts/precompute_colpali.py`."
    )
    logger.warning(msg)
    warnings.warn(msg, stacklevel=3)


# ---------------------------------------------------------------------------
# Inference script helper
# ---------------------------------------------------------------------------


def run_colpali_inference(
    corpus_items: list,  # list[CorpusItem] — typed loosely to avoid circular imports
    queries: list,       # list[QueryItem]
    output_path: Path | str,
    model_name: str = "vidore/colpali-v1.3",
    batch_size: int = 4,
    top_k: int = 100,
    device: str = "cuda",
) -> None:
    """
    Generate ColPali retrieval results and write them to a JSONL file.

    This function must be called in an environment that has ``colpali-engine``
    installed and a CUDA-capable GPU with sufficient VRAM (≥24 GB recommended).

    Installation::

        uv add colpali-engine

    Usage::

        from pathlib import Path
        from visual_hyde.baselines.colpali import run_colpali_inference
        from visual_hyde.data.loaders import load_corpus, load_queries

        corpus = load_corpus("chartqa")
        queries = load_queries("chartqa", split="test")

        run_colpali_inference(
            corpus_items=corpus,
            queries=queries,
            output_path=Path("data/colpali_results_chartqa.jsonl"),
        )

    Args:
        corpus_items: List of ``CorpusItem`` objects (corpus to search).
        queries:      List of ``QueryItem`` objects (queries to evaluate).
        output_path:  Destination JSONL file path.
        model_name:   HuggingFace model ID for ColPali
                      (default: ``vidore/colpali-v1.3``).
        batch_size:   Number of images / queries per forward pass.
        top_k:        Number of top results to store per query.
        device:       Torch device string (``"cuda"``, ``"cuda:0"``, etc.).

    Output format (one JSON object per line)::

        {
            "query_id": "<query.id>",
            "results": [
                {"corpus_id": "<corpus_item.id>", "score": <float>, "rank": <int>},
                ...
            ]
        }

    Raises:
        ImportError: If ``colpali-engine`` or ``torch`` is not installed.
        RuntimeError: If no CUDA device is available and ``device`` is ``"cuda"``.
    """
    try:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "ColPali inference requires additional packages. "
            "Install them with: uv add colpali-engine torch torchvision"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={device!r} but no CUDA GPU is available. "
            "Run ColPali on a machine with a compatible GPU."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[ColPali] Loading model %s on %s", model_name, device)
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Encode corpus images
    # ------------------------------------------------------------------
    logger.info("[ColPali] Encoding %d corpus images (batch_size=%d)", len(corpus_items), batch_size)
    corpus_embeddings: list = []  # list of per-patch embedding tensors

    import torch
    from tqdm import tqdm

    for i in tqdm(range(0, len(corpus_items), batch_size), desc="Encoding corpus"):
        batch_items = corpus_items[i : i + batch_size]
        images = [Image.open(item.image_path).convert("RGB") for item in batch_items]
        with torch.no_grad():
            batch_inputs = processor.process_images(images).to(device)
            embeddings = model(**batch_inputs)
        corpus_embeddings.extend(embeddings.unbind(0))

    # ------------------------------------------------------------------
    # Encode queries and score
    # ------------------------------------------------------------------
    logger.info("[ColPali] Encoding %d queries and scoring", len(queries))

    with output_path.open("w", encoding="utf-8") as out_fh:
        for i in tqdm(range(0, len(queries), batch_size), desc="Scoring queries"):
            batch_queries = queries[i : i + batch_size]
            query_texts = [q.text for q in batch_queries]

            with torch.no_grad():
                query_inputs = processor.process_queries(query_texts).to(device)
                query_embeddings = model(**query_inputs)  # (B, seq_len, dim)

            for q_idx, (query, q_emb) in enumerate(
                zip(batch_queries, query_embeddings.unbind(0))
            ):
                # Late-interaction MaxSim scoring
                scores = [
                    float(
                        processor.score_multi_vector(
                            q_emb.unsqueeze(0),
                            c_emb.unsqueeze(0),
                        ).item()
                    )
                    for c_emb in corpus_embeddings
                ]

                # Rank by score descending
                ranked_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
                top_indices = ranked_indices[:top_k]

                results = [
                    {
                        "corpus_id": corpus_items[j].id,
                        "score": scores[j],
                        "rank": rank + 1,
                    }
                    for rank, j in enumerate(top_indices)
                ]

                out_fh.write(
                    json.dumps({"query_id": query.id, "results": results}, ensure_ascii=False)
                    + "\n"
                )

    logger.info("[ColPali] Results written to %s", output_path)


import json  # noqa: E402 — already imported above; explicit for clarity
