"""
TCD-HyDE baseline retriever: Textual Chart Description HyDE.

Strategy
--------
1. A VLM (claude-opus-4-6) is prompted to produce a *textual description of a
   hypothetical chart* that would answer the query — i.e., the VLM operates
   entirely in text space, never generating a real image.
2. The resulting description is encoded with CLIP's **text encoder**.
3. The embedding is used to search the image-embedding corpus index.

This contrasts with Visual HyDE, which generates an *actual chart image* and
encodes it with CLIP's **image encoder**.  TCD-HyDE therefore tests whether
the HyDE benefit can be captured without cross-modal generation.
"""

from __future__ import annotations

from visual_hyde.config import GenerationSettings, get_settings
from visual_hyde.embedding.clip_encoder import CLIPEncoder
from visual_hyde.embedding.corpus_index import CorpusIndex
from visual_hyde.llm_client import BaseLLMClient, LLMClient
from visual_hyde.logging import get_logger
from visual_hyde.retrieval.base import BaseRetriever
from visual_hyde.types import QueryItem, RetrievalOutput

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert data visualization analyst. "
    "When given a retrieval query about a chart, you write a concise textual "
    "description of the hypothetical chart that would best answer that query. "
    "Describe the chart as if you are seeing it: mention the chart type, axes, "
    "data trends, key values, and any notable visual patterns. "
    "Be specific and factual. Do not hedge or say 'I imagine'; write as though "
    "the chart exists in front of you. Keep the description under 120 words."
)

_USER_TEMPLATE = (
    "Retrieval query: {query_text}\n\n"
    "Write a textual description of the chart that would best answer this query."
)


class TCDHyDERetriever(BaseRetriever):
    """
    Textual Chart Description HyDE (TCD-HyDE) baseline.

    Uses a VLM to generate a hypothetical chart *description* (text only),
    then encodes it with CLIP's text encoder to search the image corpus.

    Supports both Anthropic (Claude) and OpenAI (GPT-4o) as the VLM backend.
    Set VH_GEN_LLM_PROVIDER=openai in .env to switch providers.

    Args:
        corpus_index:  Pre-built CorpusIndex containing image embeddings.
        clip_encoder:  CLIPEncoder instance (text encoder will be used).
        gen_settings:  GenerationSettings (provider, api key, model, etc.)
        llm:           Pre-built LLM client override (useful for testing).
    """

    def __init__(
        self,
        index: CorpusIndex,
        encoder: CLIPEncoder | None = None,
        gen_settings: GenerationSettings | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self._index = index
        self._encoder = encoder or CLIPEncoder()
        self._gen = gen_settings or get_settings().generation
        self._llm: BaseLLMClient = llm or LLMClient(self._gen)

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Retriever identifier used in result tables."""
        return "tcd_hyde"

    def retrieve_one(self, query: QueryItem, top_k: int = 10) -> RetrievalOutput:
        """
        Retrieve top-k results for a single query via textual HyDE.

        Steps:
            1. Ask the VLM to describe a hypothetical chart for the query.
            2. Encode the description with CLIP text encoder.
            3. Search the corpus index and return ranked results.

        Args:
            query:  The query to process.
            top_k:  Number of results to return.

        Returns:
            RetrievalOutput with ranked SearchResults.
        """
        logger.debug(f"[TCD-HyDE] Generating chart description for query: {query.id!r}")

        description = self._generate_description(query.text)
        logger.debug(f"[TCD-HyDE] Description ({len(description)} chars): {description[:80]}...")

        # Encode the description with CLIP's text encoder
        vec = self._encoder.encode_texts([description], show_progress=False)[0]

        # Search the corpus index (image embeddings)
        results = self._index.search(vec, top_k=top_k)

        return RetrievalOutput(query_id=query.id, results=results)

    # ------------------------------------------------------------------
    # VLM generation
    # ------------------------------------------------------------------

    def _generate_description(self, query_text: str) -> str:
        """
        Call the VLM to generate a hypothetical chart description.

        Uses the configured LLM provider (Anthropic or OpenAI).

        Args:
            query_text: The raw query string.

        Returns:
            Textual chart description as a plain string.
        """
        description = self._llm.generate(
            system=_SYSTEM_PROMPT,
            user=_USER_TEMPLATE.format(query_text=query_text),
        )
        if not description:
            logger.warning("[TCD-HyDE] VLM returned empty description")
        return description.strip()
