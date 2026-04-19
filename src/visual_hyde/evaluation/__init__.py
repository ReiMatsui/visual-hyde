"""
Evaluation package for Visual HyDE.

Public API::

    from visual_hyde.evaluation import ExperimentRunner, ExperimentResults, compute_all_metrics
"""

from __future__ import annotations

from visual_hyde.evaluation.metrics import compute_all_metrics
from visual_hyde.evaluation.runner import ExperimentResults, ExperimentRunner

__all__ = [
    "ExperimentRunner",
    "ExperimentResults",
    "compute_all_metrics",
]
