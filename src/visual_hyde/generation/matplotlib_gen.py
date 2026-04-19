"""
Pattern A: Hypothetical chart generation via matplotlib code synthesis.

Pipeline:
  1. Send query to VLM (Claude) → get Python matplotlib code
  2. Execute the code in a sandboxed subprocess
  3. Return path to the generated PNG

Security note: The generated code is executed in a subprocess with a
restricted environment. Only matplotlib and numpy are available.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from visual_hyde.config import GenerationSettings, get_settings
from visual_hyde.generation.prompts import build_matplotlib_prompt
from visual_hyde.llm_client import BaseLLMClient, LLMClient
from visual_hyde.logging import get_logger
from visual_hyde.types import GeneratedChart

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------


def _extract_code_block(text: str) -> str | None:
    """Extract Python code from a markdown code block."""
    # Match ```python ... ``` or ``` ... ```
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    # No block found — return raw text if it looks like code
    if "import matplotlib" in text or "plt." in text:
        return text.strip()
    return None


def _make_safe_script(code: str, output_path: str) -> str:
    """
    Wrap user code with safety preamble.

    Sets output_path as a variable so the generated code can reference it.
    Strips any plt.show() calls to prevent blocking.
    """
    code = re.sub(r"plt\.show\(\)", "# plt.show() removed", code)
    preamble = textwrap.dedent(f"""\
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        output_path = {output_path!r}
    """)
    return preamble + "\n" + code


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class MatplotlibChartGenerator:
    """
    Generates hypothetical chart images by asking a VLM to write matplotlib
    code and then executing it in a subprocess.

    Supports both Anthropic (Claude) and OpenAI (GPT-4o) as the VLM backend.
    Set VH_GEN_LLM_PROVIDER=openai in .env to switch providers.

    Args:
        settings:  GenerationSettings override (defaults to global config).
        cache_dir: Directory to cache generated chart PNGs.
        llm:       Pre-built LLM client override (useful for testing).
    """

    def __init__(
        self,
        settings: GenerationSettings | None = None,
        cache_dir: Path | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings().generation
        cfg_paths = get_settings().paths
        self._cache_dir = cache_dir or cfg_paths.generated_charts_dir / "matplotlib"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._vlm: BaseLLMClient = llm or LLMClient(self._settings)

    def generate(self, query_id: str, query: str) -> GeneratedChart:
        """
        Generate a hypothetical chart for the given query.

        Returns a GeneratedChart. If generation fails, `generation_ok=False`
        and a fallback blank image is returned so the pipeline can continue.
        """
        save_path = self._cache_dir / f"{query_id}.png"

        # Return cached result if available
        if save_path.exists():
            return GeneratedChart(
                query_id=query_id,
                image_path=save_path,
                method="matplotlib",
                generation_ok=True,
            )

        try:
            code = self._get_matplotlib_code(query)
            self._execute_code(code, save_path)
            return GeneratedChart(
                query_id=query_id,
                image_path=save_path,
                method="matplotlib",
                generation_ok=True,
                code=code,
            )
        except Exception as e:
            logger.warning(f"Chart generation failed for query '{query_id}': {e}")
            fallback = self._make_fallback_chart(save_path, query)
            return GeneratedChart(
                query_id=query_id,
                image_path=fallback,
                method="matplotlib",
                generation_ok=False,
                error=str(e),
            )

    def generate_batch(
        self,
        query_ids: list[str],
        queries: list[str],
        show_progress: bool = True,
    ) -> list[GeneratedChart]:
        """Generate charts for a batch of queries."""
        from tqdm import tqdm

        pairs = zip(query_ids, queries)
        if show_progress:
            pairs = tqdm(list(pairs), desc="Generating charts (matplotlib)")

        return [self.generate(qid, q) for qid, q in pairs]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_matplotlib_code(self, query: str) -> str:
        system, user = build_matplotlib_prompt(query)
        raw_response = self._vlm.generate(system, user)
        code = _extract_code_block(raw_response)
        if code is None:
            raise ValueError(f"No code block found in VLM response: {raw_response[:200]}")
        return code

    def _execute_code(self, code: str, save_path: Path) -> None:
        """Execute generated code in a subprocess with timeout."""
        safe_code = _make_safe_script(code, str(save_path))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="vh_chart_"
        ) as f:
            f.write(safe_code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self._settings.generation_timeout_s,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Chart script failed (exit {result.returncode}):\n{result.stderr[:500]}"
                )
            if not save_path.exists():
                raise RuntimeError(f"Script ran but {save_path} was not created")
        finally:
            Path(script_path).unlink(missing_ok=True)

    def _make_fallback_chart(self, save_path: Path, query: str) -> Path:
        """
        Generate a generic 'unknown' placeholder chart as fallback.
        This ensures the pipeline can always retrieve a vector.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.text(
            0.5,
            0.5,
            f"[Generation failed]\n{query[:80]}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        ax.set_axis_off()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return save_path


from typing import Any
