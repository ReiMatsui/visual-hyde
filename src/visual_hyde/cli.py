"""
Visual HyDE CLI — top-level Typer application.

Commands:
  visual-hyde build-corpus   Build FAISS index for a dataset
  visual-hyde generate       Generate hypothetical charts for a query list
  visual-hyde retrieve       Run retrieval for a single query (demo)
  visual-hyde experiment     Run a full experiment (Phase 1/2/3)

All commands respect .env settings (see .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="visual-hyde",
    help="Visual HyDE: Hypothetical chart generation for cross-modal retrieval",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


@app.command("build-corpus")
def build_corpus(
    dataset: str = typer.Argument(help="Dataset: chartqa | figureqa | vidore_v2"),
    max_items: int = typer.Option(None, help="Limit corpus size for testing"),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Download and index a dataset for retrieval."""
    from visual_hyde.config import Dataset, get_settings
    from visual_hyde.data.loaders import load_dataset_for_retrieval
    from visual_hyde.embedding.clip_encoder import CLIPEncoder
    from visual_hyde.embedding.corpus_index import CorpusIndex
    from visual_hyde.logging import setup_logging
    from visual_hyde.types import EmbeddingRecord

    setup_logging(log_level)
    settings = get_settings()
    settings.ensure_ready()

    corpus_items, _ = load_dataset_for_retrieval(Dataset(dataset), max_queries=max_items)
    encoder = CLIPEncoder()

    index_dir = settings.paths.indices_dir / dataset
    if (index_dir / "index.faiss").exists():
        console.print(f"[yellow]Index already exists at {index_dir}. Delete to rebuild.[/yellow]")
        return

    vecs = encoder.encode_images([i.image_path for i in corpus_items])
    records = [EmbeddingRecord(corpus_id=i.id, vector=v) for i, v in zip(corpus_items, vecs)]
    index = CorpusIndex()
    index.build(records)
    index.save(index_dir)
    console.print(f"[green]Index saved: {index_dir} ({len(corpus_items)} items)[/green]")


@app.command("retrieve")
def retrieve_demo(
    query: str = typer.Argument(help="Query text"),
    dataset: str = typer.Option("chartqa", help="Dataset to search"),
    method: str = typer.Option("matplotlib", help="matplotlib | nano_banana | text"),
    top_k: int = typer.Option(5, help="Number of results"),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Demo: retrieve charts for a single query and print results."""
    from visual_hyde.config import Dataset, GenerationMethod, get_settings
    from visual_hyde.embedding.clip_encoder import CLIPEncoder
    from visual_hyde.embedding.corpus_index import CorpusIndex
    from visual_hyde.logging import setup_logging
    from visual_hyde.retrieval.hybrid import HybridRRFRetriever
    from visual_hyde.retrieval.text_retriever import TextDirectRetriever
    from visual_hyde.retrieval.visual_retriever import VisualHyDERetriever
    from visual_hyde.types import QueryItem

    setup_logging(log_level)
    settings = get_settings()

    index_dir = settings.paths.indices_dir / dataset
    if not (index_dir / "index.faiss").exists():
        console.print(
            f"[red]Index not found at {index_dir}.[/red]\n"
            f"Run: visual-hyde build-corpus {dataset}"
        )
        raise typer.Exit(1)

    encoder = CLIPEncoder()
    index = CorpusIndex.load(index_dir)
    query_item = QueryItem(id="demo_q", text=query)

    if method == "text":
        retriever = TextDirectRetriever(index, encoder)
    elif method in ("matplotlib", "nano_banana"):
        retriever = VisualHyDERetriever(
            index, GenerationMethod(method), encoder
        )
    else:
        console.print(f"[red]Unknown method: {method}[/red]")
        raise typer.Exit(1)

    output = retriever.retrieve_one(query_item, top_k=top_k)

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[bold]Method:[/bold] {method}\n")
    for res in output.results:
        console.print(f"  Rank {res.rank:2d} | score={res.score:.4f} | {res.corpus_id}")


@app.command("experiment")
def run_experiment(
    phase: int = typer.Argument(help="Phase number: 1, 2, or 3"),
    dataset: str = typer.Option("chartqa"),
    max_queries: int = typer.Option(None),
) -> None:
    """Run a full experiment phase."""
    import subprocess

    script_map = {
        1: "experiments/phase1_main.py",
        2: "experiments/phase2_ablation.py",
        3: "experiments/phase3_hybrid.py",
    }
    if phase not in script_map:
        console.print("[red]Phase must be 1, 2, or 3[/red]")
        raise typer.Exit(1)

    script = Path(__file__).parent.parent.parent / script_map[phase]
    cmd = [sys.executable, str(script), f"--dataset={dataset}"]
    if max_queries:
        cmd.append(f"--max-queries={max_queries}")
    subprocess.run(cmd)


if __name__ == "__main__":
    app()
