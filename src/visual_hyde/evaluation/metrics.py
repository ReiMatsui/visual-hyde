"""
Evaluation metrics for Visual HyDE retrieval experiments.

Implements MRR@k, Recall@k, and nDCG@k using pure Python (no numpy).
All functions operate on lists of RetrievalOutput and QueryItem objects
produced by the retrieval pipeline.
"""

from __future__ import annotations

import math
from typing import Any

from visual_hyde.config import ChartType, QueryType
from visual_hyde.logging import get_logger
from visual_hyde.types import CorpusItem, QueryItem, RetrievalOutput, SearchResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_relevant_set(query: QueryItem) -> set[str]:
    """Return the set of ground-truth corpus IDs for a query."""
    return set(query.relevant_ids)


def _output_by_query_id(outputs: list[RetrievalOutput]) -> dict[str, RetrievalOutput]:
    """Index RetrievalOutput objects by query_id for O(1) lookup."""
    return {o.query_id: o for o in outputs}


def _query_by_id(queries: list[QueryItem]) -> dict[str, QueryItem]:
    """Index QueryItem objects by id for O(1) lookup."""
    return {q.id: q for q in queries}


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def mrr_at_k(
    outputs: list[RetrievalOutput],
    queries: list[QueryItem],
    k: int,
) -> float:
    """
    Compute Mean Reciprocal Rank at cutoff k.

    For each query the reciprocal rank is 1/rank of the first relevant
    document in the top-k results (0 if none found within k).

    Args:
        outputs: Ranked retrieval results, one per query.
        queries: Ground-truth query objects.
        k:       Rank cutoff.

    Returns:
        MRR@k averaged across all queries.
    """
    query_map = _query_by_id(queries)
    output_map = _output_by_query_id(outputs)

    if not queries:
        logger.warning("mrr_at_k called with empty queries list")
        return 0.0

    total_rr = 0.0
    evaluated = 0

    for query in queries:
        output = output_map.get(query.id)
        if output is None:
            logger.warning("No RetrievalOutput found for query_id=%s; skipping", query.id)
            continue

        relevant = _build_relevant_set(query)
        rr = 0.0
        for result in output.results[:k]:
            if result.corpus_id in relevant:
                rr = 1.0 / result.rank
                break

        total_rr += rr
        evaluated += 1

    if evaluated == 0:
        return 0.0

    return total_rr / evaluated


def recall_at_k(
    outputs: list[RetrievalOutput],
    queries: list[QueryItem],
    k: int,
) -> float:
    """
    Compute mean Recall at cutoff k.

    Recall@k for a single query = |relevant ∩ top-k results| / |relevant|.
    Queries with no relevant documents contribute 0 to the average.

    Args:
        outputs: Ranked retrieval results, one per query.
        queries: Ground-truth query objects.
        k:       Rank cutoff.

    Returns:
        Mean Recall@k across all queries.
    """
    output_map = _output_by_query_id(outputs)

    if not queries:
        logger.warning("recall_at_k called with empty queries list")
        return 0.0

    total_recall = 0.0
    evaluated = 0

    for query in queries:
        output = output_map.get(query.id)
        if output is None:
            logger.warning("No RetrievalOutput found for query_id=%s; skipping", query.id)
            continue

        relevant = _build_relevant_set(query)
        if not relevant:
            evaluated += 1
            continue

        top_k_ids = {r.corpus_id for r in output.results[:k]}
        hits = len(relevant & top_k_ids)
        total_recall += hits / len(relevant)
        evaluated += 1

    if evaluated == 0:
        return 0.0

    return total_recall / evaluated


def ndcg_at_k(
    outputs: list[RetrievalOutput],
    queries: list[QueryItem],
    k: int,
) -> float:
    """
    Compute mean Normalized Discounted Cumulative Gain at cutoff k.

    Uses binary relevance (1 for relevant, 0 otherwise).
    IDCG is computed from the ideal ranking given the number of relevant
    documents (capped at k).

    Args:
        outputs: Ranked retrieval results, one per query.
        queries: Ground-truth query objects.
        k:       Rank cutoff.

    Returns:
        Mean nDCG@k across all queries.
    """
    output_map = _output_by_query_id(outputs)

    if not queries:
        logger.warning("ndcg_at_k called with empty queries list")
        return 0.0

    total_ndcg = 0.0
    evaluated = 0

    for query in queries:
        output = output_map.get(query.id)
        if output is None:
            logger.warning("No RetrievalOutput found for query_id=%s; skipping", query.id)
            continue

        relevant = _build_relevant_set(query)
        if not relevant:
            evaluated += 1
            continue

        # DCG: sum of rel_i / log2(rank_i + 1) for rank_i in 1..k
        dcg = 0.0
        for pos, result in enumerate(output.results[:k], start=1):
            if result.corpus_id in relevant:
                dcg += 1.0 / math.log2(pos + 1)

        # IDCG: ideal DCG with all relevant docs ranked first
        n_relevant_in_k = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(pos + 1) for pos in range(1, n_relevant_in_k + 1))

        ndcg = dcg / idcg if idcg > 0.0 else 0.0
        total_ndcg += ndcg
        evaluated += 1

    if evaluated == 0:
        return 0.0

    return total_ndcg / evaluated


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def compute_all_metrics(
    outputs: list[RetrievalOutput],
    queries: list[QueryItem],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute MRR, Recall, and nDCG for each value in k_values.

    Args:
        outputs:  Ranked retrieval results, one per query.
        queries:  Ground-truth query objects.
        k_values: List of rank cutoffs. Defaults to [5, 10].

    Returns:
        Flat dict with keys like ``"mrr@10"``, ``"recall@5"``, ``"ndcg@10"``.
    """
    if k_values is None:
        k_values = [5, 10]

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"mrr@{k}"] = mrr_at_k(outputs, queries, k)
        metrics[f"recall@{k}"] = recall_at_k(outputs, queries, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(outputs, queries, k)

    logger.debug("compute_all_metrics(k_values=%s) -> %s", k_values, metrics)
    return metrics


def compute_by_query_type(
    outputs: list[RetrievalOutput],
    queries: list[QueryItem],
    k_values: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Break down metrics by QueryType (TREND / COMPARISON / NUMERIC).

    Args:
        outputs:  Ranked retrieval results, one per query.
        queries:  Ground-truth query objects.
        k_values: List of rank cutoffs. Defaults to [5, 10].

    Returns:
        Dict keyed by QueryType value (e.g. ``"trend"``) mapping to a
        metrics dict as returned by :func:`compute_all_metrics`.
    """
    if k_values is None:
        k_values = [5, 10]

    output_map = _output_by_query_id(outputs)
    results: dict[str, dict[str, float]] = {}

    for qt in QueryType:
        group_queries = [q for q in queries if q.query_type == qt]
        if not group_queries:
            logger.debug("No queries found for query_type=%s; skipping", qt.value)
            continue

        group_outputs = [
            output_map[q.id]
            for q in group_queries
            if q.id in output_map
        ]
        results[qt.value] = compute_all_metrics(group_outputs, group_queries, k_values)
        logger.debug(
            "compute_by_query_type: query_type=%s n=%d metrics=%s",
            qt.value,
            len(group_queries),
            results[qt.value],
        )

    return results


def compute_by_chart_type(
    outputs: list[RetrievalOutput],
    queries: list[QueryItem],
    corpus_items: list[CorpusItem],
    k_values: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Break down metrics by ChartType (LINE / BAR / SCATTER / PIE / TABLE / UNKNOWN).

    A query is assigned to a chart type based on the chart type of its
    *first* ground-truth relevant corpus item.  Queries whose relevant set
    is empty or whose relevant items are not found in the corpus are skipped.

    Args:
        outputs:      Ranked retrieval results, one per query.
        queries:      Ground-truth query objects.
        corpus_items: All corpus items (used to look up chart_type by id).
        k_values:     List of rank cutoffs. Defaults to [5, 10].

    Returns:
        Dict keyed by ChartType value (e.g. ``"bar"``) mapping to a
        metrics dict as returned by :func:`compute_all_metrics`.
    """
    if k_values is None:
        k_values = [5, 10]

    corpus_map: dict[str, CorpusItem] = {c.id: c for c in corpus_items}
    output_map = _output_by_query_id(outputs)

    # Group queries by the chart type of their primary relevant document
    groups: dict[str, list[QueryItem]] = {ct.value: [] for ct in ChartType}

    for query in queries:
        chart_type: ChartType | None = None
        for rel_id in query.relevant_ids:
            item = corpus_map.get(rel_id)
            if item is not None:
                chart_type = item.chart_type
                break

        if chart_type is None:
            logger.debug(
                "Could not determine chart_type for query_id=%s; skipping", query.id
            )
            continue

        groups[chart_type.value].append(query)

    results: dict[str, dict[str, float]] = {}
    for ct_value, group_queries in groups.items():
        if not group_queries:
            continue

        group_outputs = [
            output_map[q.id]
            for q in group_queries
            if q.id in output_map
        ]
        results[ct_value] = compute_all_metrics(group_outputs, group_queries, k_values)
        logger.debug(
            "compute_by_chart_type: chart_type=%s n=%d metrics=%s",
            ct_value,
            len(group_queries),
            results[ct_value],
        )

    return results
