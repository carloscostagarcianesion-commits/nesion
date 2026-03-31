"""H2O: Heavy-Hitter Oracle for Efficient Generative Inference.

This module implements the core KV-Cache eviction algorithm based on the
papers "H2O: Heavy-Hitter Oracle for Efficient Generative Inference"
(Zhang et al. 2023) and "StreamingLLM" (Xiao et al. 2023).

The `H2OEvictor` is a standard PyTorch `nn.Module` designed to wrap around
or intercept an existing attention layer, dynamically pruning the Key-Value
tensors during the generative forward pass.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as f_torch
from torch import nn

from nesion.config import NesionConfig


class H2OEvictor(nn.Module):
    """Heavy Hitter Oracle Eviction Kernel.

    Intercepts attention operations to apply dynamic KV-Cache pruning.
    Tokens are classified into three categories:
    1. Anchor tokens (Attention Sinks): The first `sink_tokens` are always kept.
    2. Heavy Hitters (HH): Tokens with the highest cumulative attention scores.
    3. Eviction Candidates: Tokens with low attention scores that are purged.

    Parameters
    ----------
    config : NesionConfig
        Configuration object containing hyperparameters like `cache_budget`,
        `sink_tokens`, and `update_interval`.
    layer_idx : int
        The index of the attention layer this evictor is attached to.
    num_heads : int
        Number of attention heads in the layer.
    head_dim : int
        Dimensionality of each attention head.

    Notes
    -----
    This module is device-agnostic and compatible with `torch.compile`.
    
    Examples
    --------
    >>> from nesion.config import NesionConfig
    >>> config = NesionConfig(cache_budget=1.0, sink_tokens=2)
    >>> evictor = H2OEvictor(config, layer_idx=0, num_heads=4, head_dim=64)
    >>> q = torch.randn(1, 4, 1, 64)
    >>> k = torch.randn(1, 4, 10, 64)
    >>> v = torch.randn(1, 4, 10, 64)
    >>> out, (new_k, new_v) = evictor(q, k, v)
    >>> new_k.shape[2] == 10  # cache_budget=1.0 keeps all tokens
    True
    
    >>> config = NesionConfig(cache_budget=0.5, sink_tokens=2)
    >>> evictor = H2OEvictor(config, layer_idx=0, num_heads=4, head_dim=64)
    >>> # Compression ratio must be between 0 and 1
    >>> stats = evictor.get_stats()
    >>> 0 <= stats['compression_ratio'] <= 1
    True
    """

    def __init__(
        self,
        config: NesionConfig,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim

        # State tensors (Device agnostic, initialized dynamically)
        self.register_buffer("accumulated_scores", None, persistent=False)
        self.register_buffer("eviction_mask", None, persistent=False)
        
        # Tracking variables
        self.step_count: int = 0
        self.tokens_total: int = 0
        self.tokens_evicted: int = 0

    def reset_state(self) -> None:
        """Clear the cumulative state between distinct generation requests."""
        self.accumulated_scores = None
        self.eviction_mask = None
        self.step_count = 0
        self.tokens_total = 0
        self.tokens_evicted = 0

    def _compute_token_importance(
        self, 
        attn_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute exponential moving average of attention scores.

        Parameters
        ----------
        attn_weights : torch.Tensor
            Raw attention weights of shape `[batch, heads, q_len, k_len]`.

        Returns
        -------
        torch.Tensor
            Updated importance scores of shape `[batch, heads, k_len]`.

        Notes
        -----
        Math formulation: 
            S(t) = (1 - alpha) * S(t-1) + alpha * sum(A, dim=-2)
        where `alpha = 0.1` weights recent context.
        """
        # Sum attention across query dimension (eje=-2)
        current_step_scores = attn_weights.sum(dim=-2)
        
        batch_sz, heads_sz, k_len = current_step_scores.shape

        if self.accumulated_scores is None:
            self.accumulated_scores = current_step_scores.clone()
        else:
            prev_scores = self.accumulated_scores
            prev_len = prev_scores.shape[-1]

            # Pad history if sequence grew
            if k_len > prev_len:
                pad_size = k_len - prev_len
                pad_tensor = torch.zeros(
                    (batch_sz, heads_sz, pad_size),
                    dtype=prev_scores.dtype,
                    device=prev_scores.device
                )
                prev_scores = torch.cat([prev_scores, pad_tensor], dim=-1)

            # alpha = 0.1 decay
            alpha = 0.1  
            self.accumulated_scores = (1.0 - alpha) * prev_scores + alpha * current_step_scores

        return self.accumulated_scores

    def _select_tokens_to_keep(
        self, 
        scores: torch.Tensor, 
        current_seq_len: int
    ) -> torch.Tensor:
        """Select tokens using H2O logic (Anchors, HH, Evicted).

        Returns
        -------
        BoolTensor [batch, seq_len] where True = vivo
        """
        batch_sz = scores.shape[0]
        device = scores.device
        sink_tokens = self.config.sink_tokens
        
        # Calculate budget k
        # k = int(cache_budget * current_seq_len) - sink_tokens
        limit = int(self.config.cache_budget * current_seq_len)
        k = max(0, limit - sink_tokens)

        # Baseline: if budget covers everything
        if current_seq_len <= limit:
            return torch.ones((batch_sz, current_seq_len), dtype=torch.bool, device=device)

        # Average scores across heads [batch, k_len]
        head_avg_scores = scores.mean(dim=1)

        # Initialize mask
        mask = torch.zeros((batch_sz, current_seq_len), dtype=torch.bool, device=device)

        # 1. Anchor tokens (always kept)
        mask[:, :sink_tokens] = True
        
        # 2. Heavy Hitters (top-k remaining)
        # Avoid re-selecting anchors
        if sink_tokens < current_seq_len:
            candidate_scores = head_avg_scores[:, sink_tokens:]
            # Ensure topk doesn't exceed candidates
            actual_k = min(k, candidate_scores.shape[-1])
            if actual_k > 0:
                # Filter by eviction_threshold if requested
                valid_mask = candidate_scores >= self.config.eviction_threshold
                filtered_scores = candidate_scores * valid_mask.float()
                
                _, top_indices = torch.topk(filtered_scores, k=actual_k, dim=-1)
                # Correct indices back to global frame
                mask.scatter_(dim=1, index=top_indices + sink_tokens, value=True)

        return mask

    def _apply_eviction(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter KV using boolean mask handling batch > 1."""
        batch_sz, heads, seq_len, dim = key_states.shape
        num_kept = int(mask[0].sum().item())
        
        if num_kept == seq_len:
            return key_states, value_states
            
        # Functional indexing for batch-compatibility
        # We assume uniform mask across batch for standard generation behavior
        res_k = torch.zeros(
            (batch_sz, heads, num_kept, dim), 
            dtype=key_states.dtype, 
            device=key_states.device
        )
        res_v = torch.zeros(
            (batch_sz, heads, num_kept, dim), 
            dtype=value_states.dtype, 
            device=value_states.device
        )

        for i in range(batch_sz):
            b_mask = mask[i]
            res_k[i] = key_states[i, :, b_mask, :]
            res_v[i] = value_states[i, :, b_mask, :]
            
            # Update stats
            self.tokens_total += seq_len
            self.tokens_evicted += (seq_len - num_kept)
            
        return res_k, res_v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """1. Manual Attention -> 2. Importance -> 3. Evict -> 4. Return"""
        d_k = query.size(-1)
        scale = 1.0 / math.sqrt(d_k)
        
        # Q * K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attn_weights = f_torch.softmax(scores, dim=-1)
        
        # Weights * V
        attn_output = torch.matmul(attn_weights, value)

        # Token importance (detach to avoid graph growth)
        self._compute_token_importance(attn_weights.detach())
        self.step_count += 1

        new_key, new_value = key, value
        if self.step_count % self.config.update_interval == 0:
            current_seq_len = key.shape[-2]
            mask = self._select_tokens_to_keep(self.accumulated_scores, current_seq_len)
            new_key, new_value = self._apply_eviction(key, value, mask)
            
            # Correct accumulated_scores to match compaction
            if self.accumulated_scores is not None:
                new_acc = torch.zeros(
                    (
                        self.accumulated_scores.shape[0], 
                        self.accumulated_scores.shape[1], 
                        new_key.shape[-2]
                    ),
                    dtype=self.accumulated_scores.dtype,
                    device=self.accumulated_scores.device
                )
                for i in range(mask.shape[0]):
                    new_acc[i] = self.accumulated_scores[i, :, mask[i]]
                self.accumulated_scores = new_acc

        return attn_output, (new_key, new_value)

    def get_stats(self) -> dict[str, Any]:
        """tokens_total, tokens_evicted, compression_ratio, vram_saved_mb."""
        total = max(1, self.tokens_total)
        kept = total - self.tokens_evicted
        
        # 2 * heads * head_dim * 2 bytes (fp16) * 2 (K+V)
        bytes_per_token = 2 * self.num_heads * self.head_dim * 2
        saved_mb = (self.tokens_evicted * bytes_per_token) / (1024 * 1024)
        
        ratio = 1.0 - (kept / total)
        
        return {
            "tokens_total": self.tokens_total,
            "tokens_evicted": self.tokens_evicted,
            "compression_ratio": float(ratio),
            "vram_saved_mb": float(saved_mb)
        }

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"H2OEvictor(layer={self.layer_idx}, "
            f"budget={self.config.cache_budget:.1%}, "
            f"sinks={self.config.sink_tokens}, "
            f"ratio={s['compression_ratio']:.2f}, "
            f"saved={s['vram_saved_mb']:.1f}MB)"
        )
