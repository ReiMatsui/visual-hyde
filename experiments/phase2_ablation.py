"""
Phase 2: Destructive ablation study — which chart elements drive retrieval accuracy?

Uses FigureQA synthetic charts (fully controllable structure).
For each query, generates a "reference" hypothetical chart and then applies
systematic corruptions to measure their impact on retrieval metrics.

Corruption conditions (Table from Section 5.3 of research plan):
  0. baseline      — no corruption (upper bound)
  1. color_change  — keep shape/type, change colors randomly
  2. label_change  — keep shape, replace axis labels with random domain
  3. trend_reverse — reverse trend direction (up→down, down→up)
  4. type_change   — change chart type (line→bar) but keep trend
  5. full_random   — everything randomized (lower bound)

Output: results/phase2/<run_id>/ablation_results.json
        results/phase2/<run_id>/ablation_plot.png

Usage:
  uv run python experiments/phase2_ablation.py --max-queries 200
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visual_hyde.config import Dataset, get_settings
from visual_hyde.data.loaders import load_dataset_for_retrieval
from visual_hyde.embedding.clip_encoder import CLIPEncoder
from visual_hyde.embedding.corpus_index import CorpusIndex
from visual_hyde.evaluation.metrics import compute_all_metrics
from visual_hyde.generation.matplotlib_gen import MatplotlibChartGenerator
from visual_hyde.logging import get_logger, setup_logging
from visual_hyde.retrieval.visual_retriever import VisualHyDERetriever
from visual_hyde.types import EmbeddingRecord, GeneratedChart, QueryItem, RetrievalOutput

app = typer.Typer(help="Phase 2: Ablation study on chart element importance")
console = Console()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Corruption implementations
# ---------------------------------------------------------------------------


def _corrupt_color(chart_path: Path, output_path: Path) -> None:
    """Redraw chart with colors shuffled (keep everything else)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # Load original, convert to array, randomly shift hue (simple approach:
    # swap RGB channels)
    img = np.array(Image.open(chart_path).convert("RGB"))
    perm = np.random.permutation([0, 1, 2])
    corrupted = img[:, :, perm]
    Image.fromarray(corrupted).save(output_path)


def _corrupt_labels(code_template: str) -> str:
    """Replace axis label strings with unrelated domain labels."""
    import re

    random_labels = [
        ("Temperature (°C)", "Month"),
        ("Humidity (%)", "Season"),
        ("Altitude (m)", "Region"),
        ("Population (M)", "Country"),
        ("Distance (km)", "Day"),
    ]
    import random
    y_label, x_label = random.choice(random_labels)

    # Replace common label patterns with random ones
    code = re.sub(r"xlabel\(['\"].*?['\"]\)", f"xlabel('{x_label}')", code_template)
    code = re.sub(r"ylabel\(['\"].*?['\"]\)", f"ylabel('{y_label}')", code_template)
    code = re.sub(r"title\(['\"].*?['\"]\)", "title('Chart')", code)
    return code


def _make_reversed_trend_prompt(original_query: str) -> str:
    """Build a query prompt that requests the opposite trend."""
    replacements = [
        ("増加", "減少"), ("上昇", "下降"), ("増える", "減る"),
        ("increasing", "decreasing"), ("rising", "falling"),
        ("upward", "downward"), ("growth", "decline"),
    ]
    result = original_query
    for src, dst in replacements:
        result = result.replace(src, dst)
    # If no keyword replaced, append explicit reversal instruction
    if result == original_query:
        result = f"Reversed trend version of: {original_query}"
    return result


def _make_type_changed_prompt(original_query: str) -> str:
    """Build a query prompt that uses a different chart type."""
    # Line → bar, bar → line, etc.
    replacements = [
        ("折れ線", "棒グラフ"), ("棒グラフ", "折れ線"),
        ("line chart", "bar chart"), ("bar chart", "line chart"),
        ("scatter", "line"), ("pie", "bar"),
    ]
    result = original_query
    for src, dst in replacements:
        result = result.replace(src, dst)
    if result == original_query:
        result = f"Show the same data as a bar chart: {original_query}"
    return result


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


CORRUPTION_CONDITIONS = [
    "baseline",
    "color_change",
    "label_change",
    "trend_reverse",
    "type_change",
    "full_random",
]


class AblationStudy:
    """Runs the destructive ablation experiment on FigureQA."""

    def __init__(
        self,
        index: CorpusIndex,
        encoder: CLIPEncoder,
        generator: MatplotlibChartGenerator,
        ablation_cache_dir: Path,
    ) -> None:
        self._index = index
        self._encoder = encoder
        self._generator = generator
        self._cache_dir = ablation_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        queries: list[QueryItem],
        top_k: int = 10,
    ) -> dict[str, dict[str, float]]:
        """
        Run all corruption conditions.

        Returns dict: condition_name → {metric: value}
        """
        condition_results: dict[str, list[RetrievalOutput]] = {
            c: [] for c in CORRUPTION_CONDITIONS
        }

        from tqdm import tqdm

        for query in tqdm(queries, desc="Ablation queries"):
            for condition in CORRUPTION_CONDITIONS:
                output = self._retrieve_with_corruption(query, condition, top_k)
                condition_results[condition].append(output)

        # Compute metrics for each condition
        metrics: dict[str, dict[str, float]] = {}
        for condition, outputs in condition_results.items():
            metrics[condition] = compute_all_metrics(outputs, queries, k_values=[5, 10])

        return metrics

    def _retrieve_with_corruption(
        self,
        query: QueryItem,
        condition: str,
        top_k: int,
    ) -> RetrievalOutput:
        corrupted_id = f"{query.id}__{condition}"
        save_path = self._cache_dir / f"{corrupted_id}.png"

        if not save_path.exists():
            self._generate_corrupted(query, condition, save_path)

        if save_path.exists():
            vec = self._encoder.encode_images([save_path], show_progress=False)[0]
        else:
            # Fallback: use random noise vector
            import numpy as np
            vec = np.random.randn(self._encoder.embed_dim).astype("float32")
            vec /= np.linalg.norm(vec)

        results = self._index.search(vec, top_k=top_k)
        return RetrievalOutput(query_id=query.id, results=results)

    def _generate_corrupted(
        self,
        query: QueryItem,
        condition: str,
        save_path: Path,
    ) -> None:
        """Generate a corrupted hypothetical chart."""
        if condition == "baseline":
            chart = self._generator.generate(query.id, query.text)
            if chart.image_path.exists() and chart.image_path != save_path:
                import shutil
                shutil.copy(chart.image_path, save_path)

        elif condition == "color_change":
            baseline_chart = self._generator.generate(query.id, query.text)
            if baseline_chart.image_path.exists():
                _corrupt_color(baseline_chart.image_path, save_path)

        elif condition == "label_change":
            baseline_chart = self._generator.generate(query.id, query.text)
            if baseline_chart.code:
                corrupted_code = _corrupt_labels(baseline_chart.code)
                self._generator._execute_code(corrupted_code, save_path)

        elif condition == "trend_reverse":
            reversed_query = _make_reversed_trend_prompt(query.text)
            chart = self._generator.generate(corrupted_id_hack(query.id, condition), reversed_query)
            if chart.image_path.exists() and chart.image_path != save_path:
                import shutil
                shutil.copy(chart.image_path, save_path)

        elif condition == "type_change":
            type_changed_query = _make_type_changed_prompt(query.text)
            chart = self._generator.generate(corrupted_id_hack(query.id, condition), type_changed_query)
            if chart.image_path.exists() and chart.image_path != save_path:
                import shutil
                shutil.copy(chart.image_path, save_path)

        elif condition == "full_random":
            # Generate completely random noise image
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            x = np.random.randn(50)
            y = np.random.randn(50)
            ax.scatter(x, y, c=np.random.rand(50, 3))
            ax.set_title("Random")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close(fig)


def corrupted_id_hack(query_id: str, condition: str) -> str:
    """Return a temp generator cache key that won't collide with baseline."""
    return f"{query_id}__{condition}_tmp"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def run(
    max_queries: int = typer.Option(200, help="Number of FigureQA queries to use"),
    output_dir: Path = typer.Option(None, help="Override output directory"),
    log_level: str = typer.Option("INFO"),
) -> None:
    setup_logging(log_level)
    settings = get_settings()
    settings.ensure_ready()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_dir or (settings.paths.results_dir / "phase2" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load FigureQA
    console.rule("[bold blue]Phase 2: FigureQA Ablation Study")
    corpus_items, query_items = load_dataset_for_retrieval(
        Dataset.FIGURE_QA,
        max_queries=max_queries,
        split="validation1",
    )
    query_items = query_items[:max_queries]
    console.print(f"Corpus: {len(corpus_items)} | Queries: {len(query_items)}")

    # Build index
    encoder = CLIPEncoder()
    index_dir = settings.paths.indices_dir / "figureqa"

    if (index_dir / "index.faiss").exists():
        index = CorpusIndex.load(index_dir)
    else:
        vecs = encoder.encode_images([i.image_path for i in corpus_items])
        records = [EmbeddingRecord(corpus_id=i.id, vector=v) for i, v in zip(corpus_items, vecs)]
        index = CorpusIndex()
        index.build(records)
        index.save(index_dir)

    generator = MatplotlibChartGenerator()
    ablation_cache = settings.paths.generated_charts_dir / "ablation"

    study = AblationStudy(index, encoder, generator, ablation_cache)
    metrics = study.run(query_items, top_k=10)

    # Save results
    result_file = out_dir / "ablation_results.json"
    result_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    console.print(f"\n[green]Results saved: {result_file}[/green]")

    # Display table
    table = Table(title="Phase 2: Ablation Results (MRR@10)")
    table.add_column("Condition", style="cyan")
    table.add_column("MRR@10", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("NDCG@10", justify="right")

    for condition in CORRUPTION_CONDITIONS:
        m = metrics.get(condition, {})
        table.add_row(
            condition,
            f"{m.get('mrr@10', 0):.4f}",
            f"{m.get('recall@5', 0):.4f}",
            f"{m.get('ndcg@10', 0):.4f}",
        )
    console.print(table)

    _plot_ablation(metrics, out_dir / "ablation_plot.png")
    console.print(f"[green]Plot saved: {out_dir / 'ablation_plot.png'}[/green]")


def _plot_ablation(metrics: dict[str, dict[str, float]], save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [c for c in CORRUPTION_CONDITIONS if c in metrics]
    mrr_values = [metrics[c].get("mrr@10", 0) for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(conditions, mrr_values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("MRR@10")
    ax.set_title("Phase 2: Ablation Study — Impact of Chart Element Corruption on MRR@10")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=mrr_values[0], color="green", linestyle="--", alpha=0.5, label="Baseline")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    app()
