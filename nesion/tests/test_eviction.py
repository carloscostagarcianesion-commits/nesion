"""Unit tests for the H2OEvictor core logical rules."""

import pytest
import torch

from nesion.config import NesionConfig
from nesion.core.h2o_eviction import H2OEvictor


def test_anchor_tokens_never_evicted(tiny_config: NesionConfig, mock_attention_weights: torch.Tensor, generic_kv_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Verifies sink_tokens are always kept regardless of their attention scores."""
    k, v = generic_kv_tensors
    seq_len = k.shape[-2]
    
    # We test with different configs to ensure the logic holds
    configs = [
        NesionConfig(sink_tokens=2, cache_budget=0.5),
        NesionConfig(sink_tokens=4, cache_budget=0.8),
        NesionConfig(sink_tokens=1, cache_budget=0.2),
    ]
    
    for cfg in configs:
        evictor = H2OEvictor(cfg, layer_idx=0, num_heads=4, head_dim=64)
        
        # Manually set accumulated_scores with low values for the sink tokens
        evictor.accumulated_scores = torch.ones((1, 4, seq_len)) * 100.0
        evictor.accumulated_scores[..., :cfg.sink_tokens] = 0.0001
        
        mask = evictor._select_tokens_to_keep(evictor.accumulated_scores, seq_len)
        
        # Check that the first `sink_tokens` indices are True
        assert mask[0, :cfg.sink_tokens].all().item() is True, f"Sink tokens evicted with {cfg}"


def test_heavy_hitters_selection(tiny_config: NesionConfig, generic_kv_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Verifies that the top-k scored tokens are selected as heavy hitters."""
    k, v = generic_kv_tensors
    seq_len = 20
    
    evictor = H2OEvictor(tiny_config, layer_idx=0, num_heads=4, head_dim=64)
    
    # Create scores where indices 5, 10, 15 have huge scores
    scores = torch.rand(1, 4, seq_len) * 0.1
    scores[:, :, 5] = 10.0
    scores[:, :, 10] = 9.0
    scores[:, :, 15] = 8.0
    
    mask = evictor._select_tokens_to_keep(scores, seq_len)
    
    # budget = 0.5 * 20 = 10 tokens. Sinks = 2. Heavy Hitters = 8.
    expected_budget = int(tiny_config.cache_budget * seq_len)
    
    assert mask[0].sum().item() == expected_budget
    assert mask[0, 5].item() is True
    assert mask[0, 10].item() is True
    assert mask[0, 15].item() is True


@pytest.mark.parametrize("budget, sinks, seq_len", [
    (1.0, 2, 10), (0.1, 4, 100), (0.5, 0, 50),
    (0.3, 10, 20), (0.01, 1, 1000), (0.99, 5, 128)
])
def test_compression_ratio_bounds(budget: float, sinks: int, seq_len: int) -> None:
    """Validates compression_ratio is always strictly bounded in [0, 1]."""
    cfg = NesionConfig(cache_budget=budget, sink_tokens=sinks)
    evictor = H2OEvictor(cfg, layer_idx=0, num_heads=4, head_dim=64)
    
    # Mock some eviction 
    evictor.tokens_total = seq_len
    evictor.tokens_evicted = seq_len - max(sinks, int(budget * seq_len))
    
    stats = evictor.get_stats()
    assert 0.0 <= stats["compression_ratio"] <= 1.0


def test_full_budget_no_eviction(generic_kv_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Verifies that cache_budget=1.0 keeps all tokens seamlessly."""
    cfg = NesionConfig(cache_budget=1.0, sink_tokens=2)
    evictor = H2OEvictor(cfg, layer_idx=0, num_heads=4, head_dim=64)
    k, v = generic_kv_tensors
    seq_len = k.shape[-2]
    
    scores = torch.rand(1, 4, seq_len)
    mask = evictor._select_tokens_to_keep(scores, seq_len)
    
    assert mask.all().item() is True


def test_accumulated_scores_update() -> None:
    """Verifies exponential moving average decay logic of 0.9*old + 0.1*new."""
    cfg = NesionConfig(cache_budget=0.5)
    evictor = H2OEvictor(cfg, layer_idx=0, num_heads=1, head_dim=64)
    
    # Sim 10 steps
    # Weights sum(dim=-2) => [batch, heads, k_len]
    attn_1 = torch.ones(1, 1, 1, 5) # sum = 1.0 everywhere
    evictor._compute_token_importance(attn_1)
    
    assert evictor.accumulated_scores[0, 0, 0].item() == pytest.approx(1.0)
    
    attn_2 = torch.ones(1, 1, 1, 5) * 2.0 # sum = 2.0 everywhere
    evictor._compute_token_importance(attn_2)
    
    # 0.9 * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
    assert evictor.accumulated_scores[0, 0, 0].item() == pytest.approx(1.1, rel=1e-5)


@pytest.mark.parametrize("seq_len", [100, 500, 1000, 2000])
def test_vram_estimation_positive(tiny_config: NesionConfig, seq_len: int) -> None:
    """Validates VRAM saved estimation is properly non-negative and scales."""
    evictor = H2OEvictor(tiny_config, layer_idx=0, num_heads=4, head_dim=64)
    evictor.tokens_total = seq_len
    evictor.tokens_evicted = int(seq_len * 0.5)
    
    stats = evictor.get_stats()
    assert stats["vram_saved_mb"] >= 0.0


def test_reset_clears_state(tiny_config: NesionConfig) -> None:
    """Verifies that resetting state properly blanks dynamic tensors manually."""
    evictor = H2OEvictor(tiny_config, layer_idx=0, num_heads=4, head_dim=64)
    
    # Mutate
    evictor.accumulated_scores = torch.ones(1)
    evictor.eviction_mask = torch.ones(1)
    evictor.step_count = 20
    
    evictor.reset_state()
    
    assert evictor.accumulated_scores is None
    assert evictor.eviction_mask is None
    assert evictor.step_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_device_agnostic_cuda() -> None:
    """Verifies evictor tensor allocations function natively on CUDA without moves."""
    cfg = NesionConfig(cache_budget=0.5)
    evictor = H2OEvictor(cfg, layer_idx=0, num_heads=1, head_dim=64).cuda()
    
    attn = torch.ones(1, 1, 1, 5, device="cuda")
    scores = evictor._compute_token_importance(attn)
    
    assert scores.device.type == "cuda"

def test_device_agnostic_cpu() -> None:
    """Verifies evictor tensor allocations function natively on CPU."""
    cfg = NesionConfig(cache_budget=0.5)
    evictor = H2OEvictor(cfg, layer_idx=0, num_heads=1, head_dim=64)
    
    attn = torch.ones(1, 1, 1, 5, device="cpu")
    scores = evictor._compute_token_importance(attn)
    
    assert scores.device.type == "cpu"
