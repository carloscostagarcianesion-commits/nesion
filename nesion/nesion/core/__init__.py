"""Core eviction algorithms and attention hooks.

This subpackage contains the low-level building blocks used by
:class:`~nesion.engine.NesionEngine`:

- :mod:`~nesion.core.h2o_eviction` — H2O (Heavy Hitter Oracle) implementation
- :mod:`~nesion.core.attention_hook` — Forward hook for capturing attention weights
- :mod:`~nesion.core.utils` — Tensor manipulation helpers
"""

from nesion.core.attention_hook import AttentionHook
from nesion.core.h2o_eviction import H2OEvictor

__all__ = ["AttentionHook", "H2OEvictor"]
