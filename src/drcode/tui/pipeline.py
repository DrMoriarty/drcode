"""Compatibility facade of the pipeline.

The pipeline layers live UI-neutral in :mod:`drcode.core.pipeline`
(they reach the frontend only through the host seam,
:mod:`drcode.core.host`). This module re-exports the composed
:class:`PipelineLayer` — the one name `drcode.tui.pipeline` has always
exported (the app's MRO plugs it in between `ActionsLayer` and `App`) —
along with the sub-layer classes under their historical import paths.
"""

from __future__ import annotations

from ..core.pipeline import (
    PipelineAgentRunLayer,
    PipelineCompactionLayer,
    PipelineCoreLayer,
    PipelineDepsLayer,
    PipelineHooksLayer,
    PipelineLayer,
    PipelineNetLayer,
    PipelineResolverLayer,
    PipelineTesterLayer,
)

__all__ = [
    "PipelineAgentRunLayer",
    "PipelineCompactionLayer",
    "PipelineCoreLayer",
    "PipelineDepsLayer",
    "PipelineHooksLayer",
    "PipelineLayer",
    "PipelineNetLayer",
    "PipelineResolverLayer",
    "PipelineTesterLayer",
]
