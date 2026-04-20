"""
Phase 1: Main experiment — 6-condition comparison on chart retrieval.

Conditions:
  1. text_direct          (CLIP text → corpus image search)
  2. tcd_hyde             (VLM text description → CLIP text → image search)
  3. colpali              (pre-computed, loaded from JSONL)
  4. visual_hyde_matplotlib  (VLM code → matplotlib → CLIP image → search)
  5. visual_hyde_nano_banana (Gemini image gen → CLIP image → search)
  6. hybrid_rrf           (best Visual HyDE + Text-Direct, α=0.5)

Output:
  results/<dataset>/<run_id>/results.json
  results/<dataset>/<run_id>/summary_table.txt

Usage:
  uv run python experiments/phase1_main.py --dataset chartqa --max-queries 200
  uv run python experiments/phase1_main.py --dataset figureqa --max-queries 500
  uv run python experiments/phase1_main.py --dataset vidore_v2
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add src to path for editable import (uv run handles this automatically)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visual_hyde.baselines.colpali import ColPaliRetriever
from visual_hyde.baselines.tcd_hyde import TCDHyDERetriever
from visual_hyde.config import Dataset, EmbeddingSettings, GenerationMethod, get_settings
from visual_hyde.data.loaders import load_dataset_for_retrieval
from visual_hyde.embedding.clip_encoder import CLIPEncoder
from visual_hyde.embedding.corpus_index import CorpusIndex
from visual_hyde.evaluation.runner import ExperimentRunner
from visual_hyde.logging import get_logger, setup_logging
from visual_hyde.retrieval.hybrid import HybridRRFRetriever
from visual_hyde.retrieval.text_retriever import TextDirectRetriever
from visual_hyde.retrieval.visual_retriever import VisualHyDERetriever
from visual_hyde.types import EmbeddingRecord

app = typer.Typer(help="Phase 1: Main comparison experiment")
console = Console()
logger = get_logger(__name__)


def _build_or_load_index(
    corpus_items,
    encoder: CLIPEncoder,
    index_dir: Path,
) -> CorpusIndex:
    """Build corpus FAISS index (or load if already cached)."""
    if (index_dir / "index.faiss").exists():
        logger.info(f"Loading cached index from {index_dir}")
        return CorpusIndex.load(index_dir)

    logger.info(f"Building index for {len(corpus_items)} corpus items...")
    image_paths = [item.image_path for item in corpus_items]
    vecs = encoder.encode_images(image_paths, show_progress=True)

    records = [
        EmbeddingRecord(corpus_id=item.id, vector=vec)
        for item, vec in zip(corpus_items, vecs)
    ]
    index = CorpusIndex()
    index.build(records)
    index.save(index_dir)
    return index


@app.command()
def run(
    dataset: str = typer.Option("chartqa", help="Dataset: chartqa | figureqa | vidore_v2"),
    max_queries: int = typer.Option(None, help="Limit queries for quick testing"),
    alpha: float = typer.Option(0.5, help="RRF alpha weight for hybrid search"),
    colpali_results: Path = typer.Option(None, help="Pre-computed ColPali JSONL path"),
    skip_image_gen: bool = typer.Option(False, help="Skip Nano Banana (requires Gemini API key)"),
    output_dir: Path = typer.Option(None, help="Override output directory"),
    domains: str = typer.Option(None, help="Comma-separated ViDoRe domains, e.g. 'arxivqa' or 'infovqa,arxivqa'"),
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    setup_logging(log_level)
    settings = get_settings()
    settings.ensure_ready()

    dataset_enum = Dataset(dataset)
    # Parse domains option (only used for vidore_v2)
    domain_list = [d.strip() for d in domains.split(",")] if domains else None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include domain tag in output dir name for easy identification
    domain_tag = f"_{domains.replace(',', '-')}" if domains else ""
    out_dir = output_dir or (settings.paths.results_dir / dataset / f"{run_id}{domain_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Separate FAISS index per domain combination to avoid cache collisions
    index_dir = settings.paths.indices_dir / dataset / (domains or "default")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    console.rule(f"[bold blue]Phase 1: {dataset.upper()} — Loading data")
    load_kwargs: dict = {"max_queries": max_queries}
    if domain_list:
        load_kwargs["domains"] = domain_list
    corpus_items, query_items = load_dataset_for_retrieval(dataset_enum, **load_kwargs)
    console.print(f"Corpus: {len(corpus_items)} items | Queries: {len(query_items)}")

    # ------------------------------------------------------------------
    # 2. Build corpus index
    # ------------------------------------------------------------------
    console.rule("Building/Loading FAISS index")
    encoder = CLIPEncoder()
    index = _build_or_load_index(corpus_items, encoder, index_dir)

    # ------------------------------------------------------------------
    # 3. Instantiate retrievers
    # ------------------------------------------------------------------
    console.rule("Setting up retrievers")
    text_retriever = TextDirectRetriever(index=index, encoder=encoder)
    tcd_retriever = TCDHyDERetriever(index=index, encoder=encoder)
    vh_matplotlib = VisualHyDERetriever(
        index=index,
        generation_method=GenerationMethod.MATPLOTLIB,
        encoder=encoder,
    )
    colpali = ColPaliRetriever(results_path=colpali_results)
    hybrid = HybridRRFRetriever(
        visual_retriever=vh_matplotlib,
        text_retriever=text_retriever,
        alpha=alpha,
    )

    retrievers = [text_retriever, tcd_retriever, vh_matplotlib, colpali, hybrid]

    if not skip_image_gen:
        try:
            vh_nano = VisualHyDERetriever(
                index=index,
                generation_method=GenerationMethod.NANO_BANANA,
                encoder=encoder,
            )
            retrievers.insert(3, vh_nano)
            console.print("Nano Banana generator: [green]enabled[/green]")
        except Exception as e:
            console.print(f"Nano Banana skipped: {e}", style="yellow")

    # ------------------------------------------------------------------
    # 4. Run experiment
    # ------------------------------------------------------------------
    console.rule("Running retrieval experiment")
    runner = ExperimentRunner(
        retrievers=retrievers,
        queries=query_items,
        corpus_items=corpus_items,
    )
    results = runner.run(top_k=10, dataset_name=dataset)

    # ------------------------------------------------------------------
    # 5. Save & display results
    # ------------------------------------------------------------------
    runner.save_results(results, out_dir, retrievers=retrievers)
    console.print(f"\n[green]Results saved to: {out_dir}[/green]")

    _print_summary_table(results.metrics)
    console.print("\n[bold]By Query Type:[/bold]")
    _print_nested_table(results.per_query_type)

    # ── Generation failure summary ─────────────────────────────────────
    failure_keys = [k for k in results.metadata if k.endswith("_generation_failures")]
    if failure_keys:
        console.print("\n[bold]Chart Generation Failures:[/bold]")
        for key in failure_keys:
            name = key.replace("_generation_failures", "")
            n_failed = results.metadata[key]
            rate = results.metadata.get(f"{name}_generation_failure_rate", 0)
            types = results.metadata.get(f"{name}_failure_types", {})
            console.print(
                f"  {name}: [red]{n_failed}[/red] failures "
                f"({rate*100:.1f}%) — {types}"
            )


def _print_summary_table(metrics: dict[str, dict[str, float]]) -> None:
    table = Table(title="Phase 1 Results", show_header=True)
    table.add_column("Retriever", style="cyan")
    table.add_column("MRR@10", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("NDCG@10", justify="right")

    for retriever_name, m in metrics.items():
        table.add_row(
            retriever_name,
            f"{m.get('mrr@10', 0):.4f}",
            f"{m.get('recall@5', 0):.4f}",
            f"{m.get('recall@10', 0):.4f}",
            f"{m.get('ndcg@10', 0):.4f}",
        )
    console.print(table)


def _print_nested_table(nested: dict[str, dict[str, dict[str, float]]]) -> None:
    for group_name, retriever_metrics in nested.items():
        table = Table(title=f"Query Type: {group_name}", show_header=True)
        table.add_column("Retriever", style="cyan")
        table.add_column("MRR@10", justify="right")
        table.add_column("Recall@5", justify="right")
        for retriever_name, m in retriever_metrics.items():
            table.add_row(
                retriever_name,
                f"{m.get('mrr@10', 0):.4f}",
                f"{m.get('recall@5', 0):.4f}",
            )
        console.print(table)


if __name__ == "__main__":
    app()
