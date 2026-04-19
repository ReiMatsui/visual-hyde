"""
Prompt templates for hypothetical chart generation.

Two sets of prompts:
  1. MATPLOTLIB_SYSTEM / MATPLOTLIB_USER — instruct a VLM to generate
     executable matplotlib Python code that visually represents the query.
  2. IMAGE_GEN_PROMPT — plain-language prompts for Nano Banana / FLUX.1.

Design principle: The generated chart does NOT need accurate values.
It only needs to capture chart type + structural pattern (trend shape).
"""

from __future__ import annotations

MATPLOTLIB_SYSTEM = """\
You are an expert data visualization assistant.
Your task is to write Python matplotlib code that generates a chart image \
representing the VISUAL PATTERN described by the user's query.

Rules:
- Output ONLY valid Python code wrapped in triple backticks (```python ... ```)
- Import only: matplotlib, numpy (already installed)
- Use plt.savefig(output_path, dpi=100, bbox_inches='tight')
  where output_path is provided as a variable in scope
- DO NOT show, display, or plt.show() the figure
- Use dummy/synthetic data — exact values do NOT matter
- Focus on reproducing the chart TYPE and TREND SHAPE (e.g., upward line,
  grouped bars, pie slices) described in the query
- Keep the chart clean: add a title, axis labels, and a legend if needed
- Figure size: figsize=(6.4, 4.8)
- Use a professional color palette (e.g., tab10 or seaborn colors)
"""

MATPLOTLIB_USER = """\
Query: {query}

Generate matplotlib Python code that creates a chart visually representing
the pattern/structure described in this query.
Focus on:
1. Correct chart type (line, bar, scatter, pie, etc.)
2. Correct trend direction/shape (rising, falling, grouped, etc.)
3. Realistic but dummy axis labels

Variable `output_path` is already defined in the execution scope.
"""

IMAGE_GEN_PROMPT = """\
A clean, professional data visualization chart that shows: {query}

Style: white background, clear axis labels, professional color scheme,
data journalism style. No decorative elements. Chart type should match
the described pattern. High quality PNG.
"""


def build_matplotlib_prompt(query: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for matplotlib code generation."""
    return MATPLOTLIB_SYSTEM, MATPLOTLIB_USER.format(query=query)


def build_image_gen_prompt(query: str) -> str:
    """Return a text-to-image prompt for Nano Banana / FLUX.1 / SD3."""
    return IMAGE_GEN_PROMPT.format(query=query)
