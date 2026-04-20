"""
Experiment runner for Visual HyDE retrieval evaluation.

Runs multiple retrievers against a fixed query / corpus set, collects
metrics, and serialises results to JSON for offline analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visual_hyde.config import ChartType, QueryType
from visual_hyde.evaluation.metrics import (
    compute_all_metrics,
    compute_by_chart_type,
    compute_by_query_type,
)
from visual_hyde.logging import get_logger
from visual_hyde.retrieval.base import BaseRetriever
from visual_hyde.types import CorpusItem, QueryItem, RetrievalOutput, SearchResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResults:
    """
    Container for the full output of one experiment run.

    Attributes:
        metrics:          Per-retriever flat metrics dict
                          (``retriever_name -> {"mrr@10": 0.42, ...}``).
        per_query_type:   Per-retriever per-QueryType metrics
                          (``retriever_name -> {"trend": {"mrr@10": ...}, ...}``).
        per_chart_type:   Per-retriever per-ChartType metrics
                          (``retriever_name -> {"bar": {"mrr@10": ...}, ...}``).
        raw_outputs:      Full ranked lists
                          (``retriever_name -> [RetrievalOutput, ...]``).
        metadata:         Free-form dict with dataset name, timestamp, k_values, etc.
    """

    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query_type: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    per_chart_type: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    raw_outputs: dict[str, list[RetrievalOutput]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert the results to a JSON-serialisable dictionary."""
        raw_outputs_serialised: dict[str, list[dict[str, Any]]] = {}
        for retriever_name, outputs in self.raw_outputs.items():
            raw_outputs_serialised[retriever_name] = [
                {
                    "query_id": output.query_id,
                    "results": [
                        {
                            "corpus_id": r.corpus_id,
                            "score": r.score,
                            "rank": r.rank,
                        }
                        for r in output.results
                    ],
                }
                for output in outputs
            ]

        return {
            "metrics": self.metrics,
            "per_query_type": self.per_query_type,
            "per_chart_type": self.per_chart_type,
            "raw_outputs": raw_outputs_serialised,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResults:
        """Reconstruct ExperimentResults from the dict produced by :meth:`to_dict`."""
        raw_outputs: dict[str, list[RetrievalOutput]] = {}
        for retriever_name, outputs_data in data.get("raw_outputs", {}).items():
            raw_outputs[retriever_name] = [
                RetrievalOutput(
                    query_id=o["query_id"],
                    results=[
                        SearchResult(
                            corpus_id=r["corpus_id"],
                            score=r["score"],
                            rank=r["rank"],
                        )
                        for r in o["results"]
                    ],
                )
                for o in outputs_data
            ]

        return cls(
            metrics=data.get("metrics", {}),
            per_query_type=data.get("per_query_type", {}),
            per_chart_type=data.get("per_chart_type", {}),
            raw_outputs=raw_outputs,
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """
    Orchestrates retrieval evaluation across multiple retriever implementations.

    Usage::

        runner = ExperimentRunner(retrievers, queries, corpus_items)
        results = runner.run(top_k=10)
        runner.save_results(results, Path("results/run_001"))

    Args:
        retrievers:    List of retriever instances to evaluate.
        queries:       Ground-truth query objects.
        corpus_items:  Full retrieval corpus (used for chart-type breakdown).
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        queries: list[QueryItem],
        corpus_items: list[CorpusItem],
    ) -> None:
        self._retrievers = retrievers
        self._queries = queries
        self._corpus_items = corpus_items

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        top_k: int = 10,
        show_progress: bool = True,
        k_values: list[int] | None = None,
        dataset_name: str = "",
    ) -> ExperimentResults:
        """
        Run all retrievers and compute evaluation metrics.

        Args:
            top_k:          Number of results each retriever returns per query.
            show_progress:  Whether to show tqdm progress bars during retrieval.
            k_values:       Rank cutoffs for metric computation. Defaults to [5, 10].
            dataset_name:   Optional label stored in result metadata.

        Returns:
            :class:`ExperimentResults` populated with metrics and raw outputs.
        """
        if k_values is None:
            k_values = [5, 10]

        results = ExperimentResults(
            metadata={
                "dataset": dataset_name,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "top_k": top_k,
                "k_values": k_values,
                "n_queries": len(self._queries),
                "n_corpus": len(self._corpus_items),
                "retrievers": [r.name for r in self._retrievers],
            }
        )

        for retriever in self._retrievers:
            name = retriever.name
            logger.info("Running retriever: %s", name)

            try:
                outputs = retriever.retrieve_batch(
                    self._queries,
                    top_k=top_k,
                    show_progress=show_progress,
                )
            except Exception:
                logger.exception("Retriever %s raised an exception; skipping", name)
                continue

            results.raw_outputs[name] = outputs

            logger.info("Computing metrics for retriever: %s", name)
            results.metrics[name] = compute_all_metrics(
                outputs, self._queries, k_values
            )
            results.per_query_type[name] = compute_by_query_type(
                outputs, self._queries, k_values
            )
            results.per_chart_type[name] = compute_by_chart_type(
                outputs, self._queries, self._corpus_items, k_values
            )

            # ── Chart generation failure stats ────────────────────────────
            if hasattr(retriever, "generation_failures"):
                failures = retriever.generation_failures
                failure_rate = retriever.generation_failure_rate
                n_failed = len(failures)
                results.metadata[f"{name}_generation_failures"] = n_failed
                results.metadata[f"{name}_generation_failure_rate"] = round(failure_rate, 4)
                # Summarise by error type
                from collections import Counter
                type_counts = Counter(f["error_type"] for f in failures)
                results.metadata[f"{name}_failure_types"] = dict(type_counts)
                if n_failed:
                    logger.warning(
                        "Retriever %s: %d/%d chart generations failed (%.1f%%). "
                        "Types: %s",
                        name, n_failed, len(self._queries),
                        failure_rate * 100, dict(type_counts),
                    )

            logger.info(
                "Retriever %s done. Metrics: %s", name, results.metrics[name]
            )

        return results

    def save_results(
        self,
        results: ExperimentResults,
        output_dir: Path,
        retrievers: list | None = None,
    ) -> None:
        """
        Serialise ExperimentResults to ``<output_dir>/results.json``.
        Also saves chart generation failure logs to
        ``<output_dir>/<retriever>_generation_failures.jsonl`` when available.

        Args:
            results:    The experiment results to persist.
            output_dir: Directory in which to write result files.
            retrievers: Optional list of retriever instances (used to extract
                        failure details for JSONL logs).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "results.json"
        payload = results.to_dict()

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", output_path)

        # ── Save generation failure JSONL logs ────────────────────────────
        if retrievers:
            for retriever in retrievers:
                if not hasattr(retriever, "generation_failures"):
                    continue
                failures = retriever.generation_failures
                if not failures:
                    continue
                log_path = output_dir / f"{retriever.name}_generation_failures.jsonl"
                with log_path.open("w", encoding="utf-8") as fh:
                    for record in failures:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(
                    "Generation failure log saved to %s (%d entries)",
                    log_path, len(failures),
                )

    def load_results(self, output_dir: Path) -> ExperimentResults:
        """
        Load ExperimentResults from ``<output_dir>/results.json``.

        Args:
            output_dir: Directory containing ``results.json``.

        Returns:
            Reconstructed :class:`ExperimentResults` object.

        Raises:
            FileNotFoundError: If ``results.json`` does not exist.
        """
        input_path = Path(output_dir) / "results.json"

        if not input_path.exists():
            raise FileNotFoundError(
                f"results.json not found in {output_dir}. "
                "Run an experiment and call save_results() first."
            )

        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        results = ExperimentResults.from_dict(data)
        logger.info(
            "Results loaded from %s (retrievers: %s)",
            input_path,
            list(results.metrics.keys()),
        )
        return results
