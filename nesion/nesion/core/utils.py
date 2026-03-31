"""Tensor utilities for KV-Cache manipulation.

Helper functions for gathering, scattering, and reshaping key-value cache
tensors during eviction operations.
"""

from __future__ import annotations

import torch


def gather_kv(
    tensor: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather KV entries corresponding to ``True`` positions in *keep_mask*.

    Parameters
    ----------
    tensor : torch.Tensor
        Key or value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
    keep_mask : torch.Tensor
        Boolean mask of shape ``(seq_len,)`` indicating positions to keep.

    Returns
    -------
    torch.Tensor
        Pruned tensor of shape ``(batch, num_heads, kept_len, head_dim)``.
    """
    # keep_mask: (seq_len,) → indices of kept positions
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]  # (kept_len,)

    # Index into the seq_len dimension (dim=2)
    return tensor.index_select(dim=2, index=keep_indices)


def scatter_kv(
    pruned: torch.Tensor,
    keep_mask: torch.Tensor,
    original_len: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Scatter pruned KV entries back to their original positions.

    Useful for debugging/visualisation. Not used in the hot path.

    Parameters
    ----------
    pruned : torch.Tensor
        Pruned tensor of shape ``(batch, num_heads, kept_len, head_dim)``.
    keep_mask : torch.Tensor
        Boolean mask of shape ``(original_len,)``.
    original_len : int
        Original sequence length before eviction.
    fill_value : float
        Value for evicted positions. Default ``0.0``.

    Returns
    -------
    torch.Tensor
        Restored tensor of shape ``(batch, num_heads, original_len, head_dim)``.
    """
    batch, num_heads, _, head_dim = pruned.shape
    result = torch.full(
        (batch, num_heads, original_len, head_dim),
        fill_value=fill_value,
        dtype=pruned.dtype,
        device=pruned.device,
    )

    keep_indices = keep_mask.nonzero(as_tuple=True)[0]
    result[:, :, keep_indices, :] = pruned

    return result


def compute_cache_size_bytes(
    past_key_values: tuple,
) -> int:
    """Compute total memory usage of a ``past_key_values`` tuple in bytes.

    Parameters
    ----------
    past_key_values : tuple
        Standard HF ``past_key_values`` — a tuple of ``(key, value)`` pairs.

    Returns
    -------
    int
        Total memory in bytes across all layers.
    """
    total = 0
    for layer_kv in past_key_values:
        if layer_kv is not None:
            for tensor in layer_kv:
                if isinstance(tensor, torch.Tensor):
                    total += tensor.nelement() * tensor.element_size()
    return total


def compute_cache_size_mb(past_key_values: tuple) -> float:
    """Compute total memory of ``past_key_values`` in megabytes."""
    return compute_cache_size_bytes(past_key_values) / (1024 * 1024)


def get_num_layers(model: torch.nn.Module) -> int:
    """Detect the number of transformer layers in a HF model.

    Tries common attribute patterns: ``model.layers``, ``model.h``,
    ``transformer.h``, ``model.decoder.layers``, etc.
    """
    # Try direct attributes
    for attr_path in [
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
        ("gpt_neox", "layers"),
    ]:
        obj = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return len(obj)

    # Fallback: count attention modules
    count = 0
    for name, module in model.named_modules():
        if "attention" in name.lower() and hasattr(module, "q_proj"):
            count += 1
    if count > 0:
        return count

    raise RuntimeError(
        "Could not determine number of layers in the model. "
        "Please ensure this is a valid Hugging Face CausalLM."
    )


def get_num_heads(model: torch.nn.Module) -> int:
    """Detect the number of attention heads from the model config."""
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("Model does not have a .config attribute")

    for attr in ["num_attention_heads", "n_head", "num_heads"]:
        if hasattr(config, attr):
            return getattr(config, attr)

    raise RuntimeError(
        f"Could not find num_heads in model config. "
        f"Available: {list(vars(config).keys())}"
    )


def get_device(model: torch.nn.Module) -> torch.device:
    """Get the device of the model's first parameter."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def format_bytes(num_bytes: int) -> str:
    """Format a byte count into a human-readable string."""
    KB_FACTOR = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < KB_FACTOR:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= KB_FACTOR  # type: ignore[assignment]
    return f"{num_bytes:.1f} PB"
