"""Configuration dataclasses for the Nesion eviction engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NesionConfig:
    """Production configuration parameters for Nesion KV-Cache Eviction.

    Attributes
    ----------
    sink_tokens : int
        Number of Anchor tokens (e.g. system prompt, BOS) protected from eviction.
    cache_budget : float
        Fraction of the context length preserved in the KV-Cache.
    eviction_threshold : float
        Minimum cumulative attention required to not be purged immediately.
    heavy_hitter_ratio : float
        Fraction of the dynamic budget allocated explicitly for top-attention tokens.
    update_interval : int
        Controls layer/step frequency for recalculating budget limits (default per step).
    verbose : bool
        Activate debug logs tracking VRAM purges dynamically.
    """

    sink_tokens: int = 4
    cache_budget: float = 0.3
    eviction_threshold: float = 0.01
    heavy_hitter_ratio: float = 0.1
    update_interval: int = 1
    verbose: bool = False

    def __post_init__(self) -> None:
        """Called automatically after instantiation to ensure integrity."""
        self.validate()

    def validate(self) -> None:
        """Validate parameter ranges suitable for production.

        Raises
        ------
        ValueError
            If variables break algebraic constraints.
        """
        if self.sink_tokens < 0:
            raise ValueError(f"sink_tokens must be >= 0. Got: {self.sink_tokens}")
            
        if not (0.0 < self.cache_budget <= 1.0):
            raise ValueError(f"cache_budget must be in (0, 1]. Got: {self.cache_budget}")
            
        if self.eviction_threshold < 0.0:
            raise ValueError(f"eviction_threshold must be >= 0. Got: {self.eviction_threshold}")
            
        if not (0.0 <= self.heavy_hitter_ratio <= 1.0):
            msg = f"heavy_hitter_ratio must be in [0, 1]. Got: {self.heavy_hitter_ratio}"
            raise ValueError(msg)
            
        if self.update_interval < 1:
            raise ValueError(f"update_interval must be >= 1. Got: {self.update_interval}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NesionConfig:
        """Instantiate configuration dynamically from a dictionary map."""
        # Filter strictly valid dataclass fields ignoring noise inputs
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_dict)

    def __repr__(self) -> str:
        """Clean representation for logging frameworks."""
        return (
            f"NesionConfig("
            f"sink={self.sink_tokens}, "
            f"budget={self.cache_budget:.1%}, "
            f"hh_ratio={self.heavy_hitter_ratio:.1%}, "
            f"interval={self.update_interval})"
        )
