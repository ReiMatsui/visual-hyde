"""
ColPali inference script — generates pre-computed retrieval results for Visual HyDE experiments.

ColPali (Faysse et al., 2024) is a multi-vector visual document retrieval model based on
PaliGemma (~3B params). It produces patch-level embeddings for document images and uses
late-interaction MaxSim scoring (similar to ColBERT) against query token embeddings.

Hardware requirements:
  - CUDA GPU with ≥16GB VRAM (recommended: A100, 3090, 4090)
  - Apple Silicon Mac with ≥16GB unified memory (MPS, slower but functional)
  - CPU: feasible but very slow (~2-5 min/image)

Installation:
  uv add colpali-engine
  # Torch is already a dependency of visual-hyde

Usage:
  # ViDoRe (default, recommended)
  uv run python scripts/precompute_colpali.py --dataset vidore_v2 --device mps

  # ChartQA
  uv run python scripts/precompute_colpali.py --dataset chartqa --device mps --max-queries 200

  # Use CUDA if available
  uv run python scripts/precompute_colpali.py --dataset vidore_v2 --device cuda

Output:
  data/colpali/<dataset>_results.jsonl

  Each line:
    {"query_id": "vidore_q0", "results": [{"corpus_id": "...", "score": 0.83, "rank": 1}, ...]}

After running, pass the JSONL to phase1_main.py:
  uv run python experiments/phase1_main.py \\
    --dataset vidore_v2 \\
    --colpali-results data/colpali/vidore_v2_results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visual_hyde.config import Dataset, get_settings
from visual_hyde.data.loaders import load_dataset_for_retrieval
from visual_hyde.logging import get_logger, setup_logging

app = typer.Typer(help="Pre-compute ColPali retrieval results")
console = Console()
logger = get_logger(__name__)


@app.command()
def run(
    dataset: str = typer.Option("vidore_v2", help="Dataset: chartqa | figureqa | vidore_v2"),
    device: str = typer.Option("mps", help="Torch device: mps | cuda | cuda:0 | cpu"),
    model_name: str = typer.Option("vidore/colpali-v1.3", help="ColPali model on HF Hub"),
    batch_size: int = typer.Option(2, help="Images per forward pass (lower = less VRAM)"),
    top_k: int = typer.Option(100, help="Results to store per query"),
    max_queries: int = typer.Option(None, help="Limit queries for testing"),
    output_path: Path = typer.Option(None, help="Override output JSONL path"),
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    setup_logging(log_level)
    settings = get_settings()
    settings.ensure_ready()

    # ── Validate ColPali install ──────────────────────────────────────────
    try:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor
        from PIL import Image
    except ImportError as exc:
        console.print(
            "[red]Missing dependency:[/red] Install ColPali with:\n"
            "  [bold]uv add colpali-engine[/bold]",
            highlight=False,
        )
        raise typer.Exit(1) from exc

    # ── Device selection ──────────────────────────────────────────────────
    if device.startswith("cuda") and not torch.cuda.is_available():
        console.print(
            f"[yellow]Warning:[/yellow] CUDA not available. Falling back to CPU.\n"
            "For MPS (Apple Silicon) use: --device mps"
        )
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        console.print("[yellow]Warning:[/yellow] MPS not available. Falling back to CPU.")
        device = "cpu"

    console.print(f"Using device: [bold]{device}[/bold]")

    # ── Output path ───────────────────────────────────────────────────────
    colpali_dir = settings.paths.data_dir / "colpali"
    colpali_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path or (colpali_dir / f"{dataset}_results.jsonl")
    console.print(f"Output: [cyan]{out_path}[/cyan]")

    # ── Load dataset ──────────────────────────────────────────────────────
    console.rule(f"Loading dataset: {dataset.upper()}")
    dataset_enum = Dataset(dataset)
    corpus_items, query_items = load_dataset_for_retrieval(
        dataset_enum, max_queries=max_queries
    )
    console.print(f"Corpus: {len(corpus_items)} items | Queries: {len(query_items)}")

    # ── Load ColPali model ────────────────────────────────────────────────
    console.rule(f"Loading {model_name}")
    dtype = torch.bfloat16

    # MPS doesn't support bfloat16 uniformly — use float16 or float32
    if device == "mps":
        dtype = torch.float16
    elif device == "cpu":
        dtype = torch.float32

    try:
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device != "mps" else None,
        )
        if device == "mps":
            model = model.to(device)
        model.eval()
        processor = ColPaliProcessor.from_pretrained(model_name)
        console.print(f"[green]Model loaded.[/green] dtype={dtype}, device={device}")
    except Exception as exc:
        console.print(f"[red]Failed to load model:[/red] {exc}")
        raise typer.Exit(1)

    # ── Encode corpus images ──────────────────────────────────────────────
    console.rule("Encoding corpus images")
    from tqdm import tqdm

    corpus_embeddings: list = []
    n_failed_corpus = 0

    for i in tqdm(range(0, len(corpus_items), batch_size), desc="Corpus"):
        batch = corpus_items[i : i + batch_size]
        images: list = []
        valid_indices: list[int] = []

        for j, item in enumerate(batch):
            try:
                img = Image.open(item.image_path).convert("RGB")
                images.append(img)
                valid_indices.append(i + j)
            except Exception as e:
                logger.warning("Could not load image %s: %s", item.image_path, e)
                n_failed_corpus += 1
                corpus_embeddings.append(None)  # placeholder
                continue

        if not images:
            continue

        try:
            with torch.no_grad():
                batch_input = processor.process_images(images)
                batch_input = {k: v.to(device) for k, v in batch_input.items()}
                embeddings = model(**batch_input)  # (B, patches, dim)
            # Fill in the valid positions
            emb_iter = iter(embeddings.unbind(0))
            for j, item in enumerate(batch):
                if i + j in valid_indices:
                    corpus_embeddings.append(next(emb_iter).cpu())
        except Exception as e:
            logger.warning("Encoding failed for batch %d: %s", i, e)
            for _ in batch:
                corpus_embeddings.append(None)

    console.print(
        f"Corpus encoded: {len(corpus_embeddings) - n_failed_corpus} ok, "
        f"{n_failed_corpus} failed"
    )

    # Filter out None embeddings for scoring
    valid_corpus = [
        (item, emb)
        for item, emb in zip(corpus_items, corpus_embeddings)
        if emb is not None
    ]
    valid_corpus_items = [c for c, _ in valid_corpus]
    valid_corpus_embs = [e for _, e in valid_corpus]

    # ── Encode queries and score ──────────────────────────────────────────
    console.rule("Encoding queries and scoring")
    n_written = 0
    n_failed_query = 0

    with out_path.open("w", encoding="utf-8") as out_fh:
        for i in tqdm(range(0, len(query_items), batch_size), desc="Queries"):
            batch_queries = query_items[i : i + batch_size]
            query_texts = [q.text for q in batch_queries]

            try:
                with torch.no_grad():
                    q_inputs = processor.process_queries(query_texts)
                    q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
                    query_embeddings = model(**q_inputs)  # (B, seq_len, dim)

                for q, q_emb in zip(batch_queries, query_embeddings.unbind(0)):
                    q_emb_cpu = q_emb.unsqueeze(0).cpu()

                    # Late-interaction MaxSim scoring against all valid corpus items
                    scores: list[float] = []
                    for c_emb in valid_corpus_embs:
                        score = processor.score_multi_vector(
                            q_emb_cpu, c_emb.unsqueeze(0)
                        )
                        scores.append(float(score.item()))

                    # Rank descending
                    ranked = sorted(
                        range(len(scores)), key=lambda j: scores[j], reverse=True
                    )
                    top = ranked[:top_k]

                    results = [
                        {
                            "corpus_id": valid_corpus_items[j].id,
                            "score": scores[j],
                            "rank": rank + 1,
                        }
                        for rank, j in enumerate(top)
                    ]

                    out_fh.write(
                        json.dumps(
                            {"query_id": q.id, "results": results},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_written += 1

            except Exception as e:
                logger.warning("Query batch %d failed: %s", i, e)
                n_failed_query += len(batch_queries)
                # Write empty results so query IDs still appear in the file
                for q in batch_queries:
                    out_fh.write(
                        json.dumps({"query_id": q.id, "results": []}, ensure_ascii=False) + "\n"
                    )

    console.print(
        f"\n[green]Done.[/green] Written: {n_written} queries, "
        f"failed: {n_failed_query} queries.\n"
        f"Output: [cyan]{out_path}[/cyan]\n\n"
        f"Run the experiment with:\n"
        f"  [bold]uv run python experiments/phase1_main.py "
        f"--dataset {dataset} --colpali-results {out_path}[/bold]"
    )


if __name__ == "__main__":
    app()
