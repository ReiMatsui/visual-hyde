"""
Chart generation subpackage.

Provides two generation strategies (Pattern A / B) with a unified factory.
"""

from visual_hyde.config import GenerationMethod
from visual_hyde.generation.image_gen import NanoBananaGenerator
from visual_hyde.generation.matplotlib_gen import MatplotlibChartGenerator
from visual_hyde.types import GeneratedChart


def get_generator(method=None):
    """Factory: returns the appropriate generator based on config or argument."""
    from visual_hyde.config import get_settings
    if method is None:
        method = get_settings().generation.method
    method = GenerationMethod(method)
    if method == GenerationMethod.MATPLOTLIB:
        return MatplotlibChartGenerator()
    elif method == GenerationMethod.NANO_BANANA:
        return NanoBananaGenerator()
    raise ValueError(f"Unknown generation method: {method}")


__all__ = ["MatplotlibChartGenerator", "NanoBananaGenerator", "get_generator", "GeneratedChart"]
