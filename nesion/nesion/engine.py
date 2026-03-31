"""NesionEngine — High-level interface for KV-Cache Eviction.

Orchestrates the interception of attention layers and applies the 
H2OEvictor module to manage memory dynamically during generation.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from torch import nn

from nesion.config import NesionConfig
from nesion.core.h2o_eviction import H2OEvictor
from nesion.core.utils import get_device, get_num_heads, get_num_layers

logger = logging.getLogger(__name__)

# Classes representing attention in modern HF architectures
_DEFAULT_ATTENTION_CLASSES = {
    "LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2",
    "MistralAttention", "MistralSdpaAttention", "MistralFlashAttention2",
    "Qwen2Attention", "Qwen2SdpaAttention", "GemmaAttention", "GemmaSdpaAttention",
    "Phi3Attention", "Phi3SdpaAttention", "GPTNeoXAttention", "GPT2Attention",
}


class NesionEngine:
    """Production-grade KV-Cache interception engine.

    Intercepts the model's forward passes and injects the H2OEvictor
    to manage memory footprint without retraining or architectural changes.

    Parameters
    ----------
    model : nn.Module
        HF Transformer model (e.g. LlamaForCausalLM).
    cache_budget : float
        Retention fraction (0.3 = 30% kept).
    config : NesionConfig, optional
        Custom config override.
    """

    def __init__(
        self,
        model: nn.Module,
        cache_budget: float = 0.3,
        config: NesionConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or NesionConfig(cache_budget=cache_budget)
        
        self._device = get_device(model)
        self._num_layers = get_num_layers(model)
        self._num_heads = get_num_heads(model)
        
        # Estimate head_dim from configuration
        self._head_dim = getattr(model.config, "head_dim", 
                                getattr(model.config, "hidden_size", 4096) // self._num_heads)

        # Evictors registry (one per layer)
        self.evictors: list[H2OEvictor] = [
            H2OEvictor(self.config, i, self._num_heads, self._head_dim).to(self._device)
            for i in range(self._num_layers)
        ]

        self._is_applied = False
        self._originals: dict[str, Callable] = {}

    def apply(self) -> None:
        """Apply H2O interception to all detectable attention modules."""
        if self._is_applied:
            return

        logger.info(f"Applying Nesion Engine (budget={self.config.cache_budget:.1%})")
        
        layer_idx = 0
        for name, module in self.model.named_modules():
            class_name = type(module).__name__
            if class_name in _DEFAULT_ATTENTION_CLASSES:
                self._patch_module(name, module, layer_idx)
                layer_idx += 1
        
        # Patch model.generate to reset state
        self._patch_generate()
        self._is_applied = True

    def remove(self) -> None:
        """Restore original forward methods and clean up memory."""
        if not self._is_applied:
            return

        for name, module in self.model.named_modules():
            if name in self._originals:
                module.forward = self._originals[name]
        
        if "generate" in self._originals:
            self.model.generate = self._originals["generate"]
            
        self._originals.clear()
        self._is_applied = False
        
        # Cleanup
        for evictor in self.evictors:
            evictor.reset_state()
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _patch_module(self, name: str, module: nn.Module, layer_idx: int) -> None:
        """Replace the forward method with a H2O-aware wrapper."""
        original_forward = module.forward
        self._originals[name] = original_forward
        evictor = self.evictors[layer_idx]

        @wraps(original_forward)
        def nesion_forward(*args: Any, **kwargs: Any) -> Any:
            # We enforce use_cache=True because we intercept past_key_value
            kwargs["use_cache"] = True
            
            # Exec original forward
            outputs = original_forward(*args, **kwargs)
            
            # HF attention typically: (attn_output, attn_weights, past_key_value)
            min_out_len = 3
            if not isinstance(outputs, tuple) or len(outputs) < min_out_len:
                return outputs
                
            attn_out = outputs[0]
            attn_weights = outputs[1]
            past_kv = outputs[2]
            
            if past_kv is not None and attn_weights is not None:
                # Use our evictor logic for pruning
                # Note: evictor.forward computes attention again or we pass weights
                # Spec says forward(q, k, v) computes attention, but here we already have it.
                # To maintain spec consistency, we use evictor inner methods.
                
                k, v = past_kv
                evictor._compute_token_importance(attn_weights.detach())
                evictor.step_count += 1
                
                if evictor.step_count % self.config.update_interval == 0:
                    mask = evictor._select_tokens_to_keep(evictor.accumulated_scores, k.shape[-2])
                    new_k, new_v = evictor._apply_eviction(k, v, mask)
                    
                    # Sync scores
                    if evictor.accumulated_scores is not None:
                        new_acc = torch.zeros(
                            (
                                evictor.accumulated_scores.shape[0], 
                                evictor.accumulated_scores.shape[1], 
                                new_k.shape[-2]
                            ),
                            dtype=evictor.accumulated_scores.dtype,
                            device=evictor.accumulated_scores.device
                        )
                        for i in range(mask.shape[0]):
                            new_acc[i] = evictor.accumulated_scores[i, :, mask[i]]
                        evictor.accumulated_scores = new_acc
                        
                    return (attn_out, attn_weights, (new_k, new_v), *outputs[3:])
            
            return outputs

        module.forward = nesion_forward

    def _patch_generate(self) -> None:
        """Ensure generate() starts with a fresh eviction state."""
        original_generate = self.model.generate
        self._originals["generate"] = original_generate

        @wraps(original_generate)
        def nesion_generate(*args: Any, **kwargs: Any) -> Any:
            for evictor in self.evictors:
                evictor.reset_state()
            kwargs["output_attentions"] = True
            return original_generate(*args, **kwargs)

        self.model.generate = nesion_generate

    def __enter__(self) -> NesionEngine:
        self.apply()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats across all layers."""
        total_evicted = 0
        total_vram_saved = 0
        ratios = []
        
        for ev in self.evictors:
            s = ev.get_stats()
            total_evicted += s["tokens_evicted"]
            total_vram_saved += s["vram_saved_mb"]
            ratios.append(s["compression_ratio"])
            
        return {
            "total_tokens_evicted": total_evicted,
            "total_vram_saved_mb": total_vram_saved,
            "avg_compression_ratio": sum(ratios) / len(ratios) if ratios else 0
        }
