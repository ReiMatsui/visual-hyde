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
You are a data visualization code generator for a retrieval system.
Your output is executed automatically — any error causes pipeline failure.

══ MANDATORY OUTPUT FORMAT ══
Wrap ALL code in a single ```python ... ``` block. Nothing outside the block.

══ EXECUTION ENVIRONMENT ══
- Available imports: matplotlib, numpy ONLY (no pandas, seaborn, scipy, etc.)
- The variable `output_path` (str) is pre-defined in scope — DO NOT redefine it
- matplotlib backend is already set to 'Agg' — DO NOT call matplotlib.use()

══ REQUIRED CALLS (both must appear) ══
1. plt.savefig(output_path, dpi=100, bbox_inches='tight')
2. plt.close()   ← prevents memory leaks across batch runs

══ FORBIDDEN ══
- plt.show()  — blocks the process
- Any file I/O, network calls, or imports beyond matplotlib/numpy
- Interactive widgets or animations
- Redefining output_path

══ GOAL ══
Reproduce the VISUAL STRUCTURE implied by the query using synthetic data.
- Choose the right chart type (line, bar, pie, scatter, heatmap, etc.)
- Match the general trend/pattern (increasing, decreasing, grouped, proportional)
- Use realistic but dummy axis labels and a descriptive title
- Figure: figsize=(6.4, 4.8), tab10 color palette
- Keep it clean and readable

Exact data values do NOT matter — only visual structure matters.
"""

MATPLOTLIB_USER = """\
Query: {query}

Write matplotlib Python code (inside ```python ... ```) that creates a chart
capturing the visual structure implied by this query.

Checklist before outputting:
✓ Code is inside ```python ... ```
✓ Only matplotlib and numpy are imported
✓ plt.savefig(output_path, dpi=100, bbox_inches='tight') is called
✓ plt.close() is called at the end
✓ plt.show() does NOT appear
✓ output_path is NOT redefined
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
