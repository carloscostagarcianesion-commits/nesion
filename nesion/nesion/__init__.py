"""Nesion: Production-Grade KV-Cache Eviction Engine for Hugging Face LLMs.

Optimises generative inference by seamlessly hooking into Attention layers 
and mathematically purging low-relevance KV tokens on the fly.
"""

from __future__ import annotations

# Core API Re-exports
from nesion.config import NesionConfig
from nesion.core.h2o_eviction import H2OEvictor
from nesion.engine import NesionEngine

__version__ = "0.1.0"
__all__ = ["H2OEvictor", "NesionConfig", "NesionEngine"]
