"""
Configuration management for Visual HyDE.

All settings can be overridden via environment variables or a .env file.
Uses pydantic-settings for typed, validated configuration.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EmbeddingModel(str, Enum):
    """Supported CLIP/SigLIP variants."""

    CLIP_VIT_B32 = "openai/clip-vit-base-patch32"
    CLIP_VIT_L14 = "openai/clip-vit-large-patch14"
    SIGLIP_BASE = "google/siglip-base-patch16-224"
    SIGLIP_SO400M = "google/siglip-so400m-patch14-384"


class GenerationMethod(str, Enum):
    """Chart generation strategy."""

    MATPLOTLIB = "matplotlib"   # Pattern A: VLM → matplotlib code → PNG
    NANO_BANANA = "nano_banana" # Pattern B: Gemini image generation


class LLMProvider(str, Enum):
    """LLM provider for VLM-based code/text generation."""

    ANTHROPIC = "anthropic"  # Claude (claude-opus-4-6 etc.)
    OPENAI = "openai"        # GPT-4o, GPT-4-turbo etc.


class Dataset(str, Enum):
    """Supported datasets."""

    CHART_QA = "chartqa"
    FIGURE_QA = "figureqa"
    VIDORE_V2 = "vidore_v2"


class ChartType(str, Enum):
    """Chart taxonomy used for cross-analysis."""

    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    TABLE = "table"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    """Query taxonomy (Section 5.2)."""

    TREND = "trend"       # Type A: tendency/shape queries
    COMPARISON = "comparison"  # Type B: comparative queries
    NUMERIC = "numeric"   # Type C: specific value queries


# ---------------------------------------------------------------------------
# Settings classes
# ---------------------------------------------------------------------------


class HuggingFaceSettings(BaseSettings):
    """HuggingFace credentials. Used for gated datasets (e.g. ViDoRe V2)."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    hf_token: str = ""

    def apply(self) -> None:
        """Set HF_TOKEN so datasets / huggingface_hub pick it up automatically."""
        import os
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            try:
                from huggingface_hub import login
                login(token=self.hf_token, add_to_git_credential=False)
            except Exception:
                pass  # HF_TOKEN env var alone is sufficient for most datasets


class PathSettings(BaseSettings):
    """File-system paths. Override via env vars prefixed with VH_."""

    model_config = SettingsConfigDict(env_prefix="VH_", env_file=".env", extra="ignore")

    # config.py は src/visual_hyde/config.py にある
    # .parent x3 → src/visual_hyde → src → project root
    project_root: Path = Field(default=Path(__file__).parent.parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def indices_dir(self) -> Path:
        return self.data_dir / "indices"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def generated_charts_dir(self) -> Path:
        return self.data_dir / "generated_charts"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [
            self.raw_dir,
            self.processed_dir,
            self.indices_dir,
            self.results_dir,
            self.generated_charts_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="VH_EMBED_", env_file=".env", extra="ignore")

    model: EmbeddingModel = EmbeddingModel.CLIP_VIT_L14
    batch_size: int = 32
    device: str = "cpu"  # "cuda" | "mps" | "cpu"
    normalize: bool = True  # L2-normalize embeddings before storage


class GenerationSettings(BaseSettings):
    """Hypothetical chart generation configuration."""

    model_config = SettingsConfigDict(env_prefix="VH_GEN_", env_file=".env", extra="ignore")

    method: GenerationMethod = GenerationMethod.MATPLOTLIB

    # ── LLM provider selection ───────────────────────────────────────────────
    # Set VH_GEN_LLM_PROVIDER=openai to switch from Anthropic to OpenAI.
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC

    # Anthropic API
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-6"

    # OpenAI API
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = ""  # Override for Azure / local proxies (empty = default)

    # Shared VLM settings
    max_code_tokens: int = 1024
    generation_timeout_s: int = 30

    # Image generation (Nano Banana / Gemini)
    gemini_api_key: str = ""
    image_gen_model: str = "gemini-2.0-flash-preview-image-generation"

    # Matplotlib render settings
    chart_width_px: int = 512
    chart_height_px: int = 384
    chart_dpi: int = 100

    @property
    def active_model(self) -> str:
        """Return the model name for the currently selected provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_model
        return self.anthropic_model


class RetrievalSettings(BaseSettings):
    """Retrieval and hybrid search configuration."""

    model_config = SettingsConfigDict(env_prefix="VH_RETR_", env_file=".env", extra="ignore")

    top_k: int = 10
    # RRF hybrid weight: 0=text-only, 1=visual-only
    alpha: float = 0.5
    rrf_k: int = 60  # RRF smoothing constant


class EvaluationSettings(BaseSettings):
    """Evaluation configuration."""

    model_config = SettingsConfigDict(env_prefix="VH_EVAL_", env_file=".env", extra="ignore")

    mrr_k: int = 10
    recall_k_values: list[int] = Field(default=[5, 10])
    ndcg_k: int = 10


# ---------------------------------------------------------------------------
# Unified settings accessor
# ---------------------------------------------------------------------------


class Settings:
    """Top-level settings container. Use as a singleton via `get_settings()`."""

    def __init__(self) -> None:
        self.huggingface = HuggingFaceSettings()
        self.huggingface.apply()  # HF_TOKEN を環境変数に反映
        self.paths = PathSettings()
        self.embedding = EmbeddingSettings()
        self.generation = GenerationSettings()
        self.retrieval = RetrievalSettings()
        self.evaluation = EvaluationSettings()

    def ensure_ready(self) -> None:
        """Create directories and validate critical settings."""
        self.paths.ensure_dirs()


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
