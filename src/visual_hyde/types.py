"""
Shared domain types for Visual HyDE.

All data flowing through the pipeline is typed using these dataclasses /
TypedDicts so that module boundaries are explicit and mypy-checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visual_hyde.config import ChartType, QueryType


# ---------------------------------------------------------------------------
# Corpus items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusItem:
    """A single chart image in the retrieval corpus."""

    id: str                    # Unique identifier (e.g. "chartqa_001")
    image_path: Path           # Absolute path to the PNG/JPG
    chart_type: ChartType = ChartType.UNKNOWN
    source_dataset: str = ""   # "chartqa" | "figureqa" | "vidore_v2"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryItem:
    """A single retrieval query."""

    id: str
    text: str
    query_type: QueryType = QueryType.TREND
    relevant_ids: list[str] = field(default_factory=list)  # Ground-truth corpus IDs
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single ranked result from any retriever."""

    corpus_id: str
    score: float
    rank: int


@dataclass
class RetrievalOutput:
    """Full ranked list for one query."""

    query_id: str
    results: list[SearchResult]  # Ordered by rank (ascending)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingRecord:
    """An embedding vector paired with its corpus item ID."""

    corpus_id: str
    vector: np.ndarray  # shape: (embed_dim,), dtype float32

    def __post_init__(self) -> None:
        if self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)


# ---------------------------------------------------------------------------
# Generated charts
# ---------------------------------------------------------------------------


@dataclass
class GeneratedChart:
    """Output of the hypothetical chart generation step."""

    query_id: str
    image_path: Path
    method: str           # "matplotlib" | "nano_banana"
    generation_ok: bool   # False if generation failed / fallback used
    code: str = ""        # matplotlib source code (if applicable)
    error: str = ""       # Error message if generation_ok is False


# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------


@dataclass
class ExperimentCondition:
    """Defines one row in the comparison table."""

    name: str             # e.g. "visual_hyde_matplotlib"
    category: str         # "proposed" | "baseline" | "existing_sota"
    description: str


EXPERIMENT_CONDITIONS = [
    ExperimentCondition(
        name="text_direct",
        category="baseline",
        description="CLIPテキストエンコーダで直接クエリを埋め込んで検索",
    ),
    ExperimentCondition(
        name="tcd_hyde",
        category="baseline",
        description="VLMがチャートをテキスト記述→テキスト埋め込みで検索",
    ),
    ExperimentCondition(
        name="colpali",
        category="existing_sota",
        description="ページ画像パッチ分割+VLM埋め込み+Late Interaction",
    ),
    ExperimentCondition(
        name="visual_hyde_matplotlib",
        category="proposed",
        description="matplotlibコード生成→PNG→CLIP画像埋め込み検索",
    ),
    ExperimentCondition(
        name="visual_hyde_nano_banana",
        category="proposed",
        description="Nano Banana画像生成→CLIP画像埋め込み検索",
    ),
    ExperimentCondition(
        name="visual_hyde_hybrid",
        category="proposed",
        description="最良Visual HyDE条件 + Text-Direct の RRF統合",
    ),
]
