"""Shared fixtures for the Nesion test suite.

Provides deterministic configurations, mock attention weights, and a dummy
LLM to ensure tests are reproducible and fast without requiring GPU/Internet.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from nesion.config import NesionConfig


@pytest.fixture(scope="session")
def tiny_config() -> NesionConfig:
    """NesionConfig with minimal values for fast testing."""
    return NesionConfig(
        sink_tokens=2,
        cache_budget=0.5,
        eviction_threshold=0.01,
        heavy_hitter_ratio=0.1,  # Kept for compatibility if used
        update_interval=1,
        verbose=False,
    )


@pytest.fixture(scope="session")
def mock_attention_weights() -> torch.Tensor:
    """Mock attention tensor [1, 4, 16, 16] with specific distribution.
    
    Tokens 0-1: high scores (0.8+)
    Tokens 2-5: medium scores (0.3-0.5)
    Tokens 6-15: low scores (0.01-0.05)
    """
    # Use manual seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(42)

    batch_size, num_heads, q_len, k_len = 1, 4, 16, 16
    weights = torch.empty((batch_size, num_heads, q_len, k_len))

    # First 2 tokens: High scores uniformly in [0.8, 1.0)
    weights[..., :, 0:2] = 0.8 + 0.2 * torch.rand((batch_size, num_heads, q_len, 2), generator=generator)
    
    # Tokens 2-5: Medium scores uniformly in [0.3, 0.5)
    weights[..., :, 2:6] = 0.3 + 0.2 * torch.rand((batch_size, num_heads, q_len, 4), generator=generator)
    
    # Tokens 6-15: Low scores uniformly in [0.01, 0.05)
    weights[..., :, 6:16] = 0.01 + 0.04 * torch.rand((batch_size, num_heads, q_len, 10), generator=generator)

    # Normalize across k_len so it behaves somewhat like softmax
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


@pytest.fixture(scope="session")
def tiny_model_and_tokenizer() -> tuple[nn.Module, Any]:
    """Provides a tiny dummy LLaMA model and a mock tokenizer for fast offline CI testing."""
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    class MockTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        
        def __call__(self, text: str, return_tensors: str = "pt", **kwargs) -> dict:
            # Just create a dummy tensor of length roughly proportional to words
            words = text.split()
            length = max(1, len(words))
            return {"input_ids": torch.randint(1, 1000, (1, length))}
            
        def decode(self, *args, **kwargs) -> str:
            return "Mock decoded text"
            
    return model, MockTokenizer()


@pytest.fixture(scope="session")
def sample_prompt() -> str:
    """String of ~50 tokens for generation tests."""
    return (
        "In the quiet expanse of the cosmos, the first stars began to ignite. "
        "Their light traveled across immense distances, carrying the ancient "
        "secrets of the universe. We watched with wonder, knowing that each "
        "flicker contained the history of a billion years and the promise of tomorrow."
    )


@pytest.fixture
def device() -> torch.device:
    """Best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def generic_kv_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generic k, v tensors of shape [1, 4, 16, 64]."""
    k = torch.randn(1, 4, 16, 64, device=device)
    v = torch.randn(1, 4, 16, 64, device=device)
    return k, v
