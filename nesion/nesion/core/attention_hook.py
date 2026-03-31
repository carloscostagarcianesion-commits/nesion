"""Forward hooks for capturing attention weights from HF Transformers models.

This module provides :class:`AttentionHook`, which registers ``forward`` hooks
on attention modules to intercept the attention weight matrices required by the
H2O eviction algorithm.

Compatible with:
  • ``LlamaAttention``, ``MistralAttention``, ``GPTNeoXAttention``
  • Any HF attention module that returns ``(attn_output, attn_weights, ...)``
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Known HF attention class names (used for auto-detection)
_KNOWN_ATTENTION_CLASSES = {
    "LlamaAttention",
    "LlamaSdpaAttention",
    "LlamaFlashAttention2",
    "MistralAttention",
    "MistralSdpaAttention",
    "MistralFlashAttention2",
    "GPTNeoXAttention",
    "GPT2Attention",
    "BloomAttention",
    "FalconAttention",
    "GemmaAttention",
    "GemmaSdpaAttention",
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Phi3Attention",
    "Phi3SdpaAttention",
}


class AttentionHook:
    """Manages forward hooks on attention layers to capture attention weights.

    Parameters
    ----------
    model : nn.Module
        The Hugging Face model to hook into.
    callback : callable
        Function called with ``(layer_idx, attention_weights)`` every time
        an attention forward pass completes.
    skip_layers : list[int] | None
        Layer indices to skip (no hook will be registered).
    """

    def __init__(
        self,
        model: nn.Module,
        callback: Callable[[int, torch.Tensor], None],
        skip_layers: list[int] | None = None,
    ) -> None:
        self._model = model
        self._callback = callback
        self._skip_layers = set(skip_layers) if skip_layers else set()
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._layer_map: dict[int, str] = {}

        self._register_hooks()

    # ── Hook Registration ────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Auto-detect attention modules and register forward hooks."""
        attention_modules = self._find_attention_modules()

        if not attention_modules:
            raise RuntimeError(
                "Could not find any attention modules in the model. "
                "Supported architectures include: LLaMA, Mistral, GPT-NeoX, "
                "GPT-2, BLOOM, Falcon, Gemma, Qwen2, Phi-3. "
                "Please check that the model is a valid HF CausalLM."
            )

        for layer_idx, (name, module) in enumerate(attention_modules):
            if layer_idx in self._skip_layers:
                logger.debug(f"Skipping layer {layer_idx} ({name})")
                continue

            hook = module.register_forward_hook(
                self._make_hook_fn(layer_idx)
            )
            self._hooks.append(hook)
            self._layer_map[layer_idx] = name

        logger.info(
            f"Registered {len(self._hooks)} attention hooks "
            f"(skipped {len(self._skip_layers)} layers)"
        )

    def _find_attention_modules(self) -> list[tuple[str, nn.Module]]:
        """Find all attention modules in the model by class name matching."""
        attention_modules: list[tuple[str, nn.Module]] = []

        for name, module in self._model.named_modules():
            class_name = type(module).__name__
            if class_name in _KNOWN_ATTENTION_CLASSES:
                attention_modules.append((name, module))

        if not attention_modules:
            # Fallback: try to find by attribute patterns
            attention_modules = self._find_by_structure()

        return attention_modules

    def _find_by_structure(self) -> list[tuple[str, nn.Module]]:
        """Fallback detection: look for modules with attention-like attributes."""
        results: list[tuple[str, nn.Module]] = []

        for name, module in self._model.named_modules():
            has_projections = (
                hasattr(module, "q_proj")
                or hasattr(module, "query")
                or hasattr(module, "q_attn")
            )
            has_kv = (
                hasattr(module, "k_proj")
                or hasattr(module, "key")
                or hasattr(module, "kv_attn")
            )
            if has_projections and has_kv:
                results.append((name, module))

        return results

    def _make_hook_fn(
        self, layer_idx: int
    ) -> Callable[..., None]:
        """Create a hook function bound to a specific layer index.

        The hook intercepts the module output and extracts the attention
        weights tensor. HF attention modules typically return:
            ``(attn_output, attn_weights, past_key_value)``
        or just ``(attn_output,)`` when ``output_attentions=False``.
        """

        def hook_fn(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> None:
            # Extract attention weights from output tuple
            attn_weights = self._extract_attention_weights(output)
            if attn_weights is not None:
                self._callback(layer_idx, attn_weights)

        return hook_fn

    @staticmethod
    def _extract_attention_weights(output: Any) -> torch.Tensor | None:
        """Extract attention weights from a module's output.

        Handles multiple output formats:
          • Tuple with attn_weights at index 1
          • Single tensor (attn_output only) — returns None
          • Dataclass-style outputs with ``.attentions`` attribute
        """
        EXPECTED_DIM = 4
        min_out_len = 3
        if isinstance(output, tuple) and len(output) >= min_out_len:
            candidate = output[1]
            if isinstance(candidate, torch.Tensor) and candidate.dim() == EXPECTED_DIM:
                return candidate
        elif hasattr(output, "attentions") and output.attentions is not None:
            return output.attentions

        return None

    # ── Lifecycle ────────────────────────────────────────────────────

    def remove(self) -> None:
        """Remove all registered hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._layer_map.clear()
        logger.info("All attention hooks removed")

    @property
    def num_hooked_layers(self) -> int:
        """Number of layers currently hooked."""
        return len(self._hooks)

    def __repr__(self) -> str:
        return (
            f"AttentionHook(hooked_layers={self.num_hooked_layers}, "
            f"skipped={len(self._skip_layers)})"
        )

    def __del__(self) -> None:
        self.remove()
