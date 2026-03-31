# Nesion Engine Architecture

This document describes the internal operational flow of the Nesion KV-Cache Eviction Engine.

## Internal Processing Flow

Nesion intercepts the generation pipeline by injecting a lightweight `H2OEvictor` module into each attention layer of the Transformer model.

```text
 ┌───────────────────────┐
 │   User Input Prompt   │
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐
 │   Transformer Layer   │
 │ (e.g. Llama Block 7)  │
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐       ┌─────────────────────────┐
 │  Attention Hook (Eng) ├──────►│  H2O Evictor Logic Pool │
 │ (Intercepts KV Cache) │       │                         │
 └───────────┬───────────┘       │  1. Accumulate Scores   │
             │                   │  2. Rank Heavy Hitters  │
             ▼                   │  3. Filter Candidates   │
 ┌───────────────────────┐       └────────────┬────────────┘
 │  Compacted KV-Cache   │◄───────────────────┘
 │ (Selected VRAM Dim)   │
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐
 │  Next Token Output    │
 └───────────────────────┘
```

## Component Breakdown

### NesionEngine
The engine is the main entry point. It traverses the model's module hierarchy and applies monkey-patches to known attention classes (like `LlamaAttention`). During the forward pass, it ensures that attention weights are captured and passed to the `H2OEvictor` for state updates.

### H2OEvictor (Heavy-Hitter Oracle)
The evictor is a core module that implements the H2O math. It maintains a state tensor of cumulative attention importance.
Key stages:
- **Importance Tracking**: Calibrates scores using Exponential Moving Average (EMA).
- **Classification**: Separates tokens into **Anchors** (attention sinks), **Heavy Hitters** (global high impact), and **Evicted** (low impact).
- **Pruning**: Physically compacts the `float16` or `bfloat16` tensors in GPU memory.

### Attention Sink Anchor System
Based on research (Xiao et al. 2023), the first few tokens of a sequence (like `<s>` or BOL) serve as catastrophic attention sinks. Nesion forcefully anchors these positions to prevent model perplexity collapse regardless of their calculated scores.

## Performance & Optimization

### Memory Management
Nesion reduces VRAM by physically pruning the KV-Cache tensors. This is done in-place where possible to avoid expensive memory allocations. For high-throughput scenarios, Nesion uses a lazy-eviction strategy (controlled by `update_interval` in `NesionConfig`) to batch pruning operations.

### Computational Overhead
The overhead of tracking attention scores is minimal (~3-5% latency increase) compared to the significant gains in VRAM and long-context throughput.

## Maintenance and Stability

### Regression Testing
The project maintains a 90%+ test coverage suite targeting:
- **Numerical Stability**: Ensuring ROUGE-L scores remain above 0.98.
- **Device Agnostic**: Supporting both CPU and CUDA backends.
- **Architectural Coverage**: Testing across Llama, Mistral, and Phi architectures.

### Versioning Policy
Nesion follows Semantic Versioning (SemVer) 2.0.0. Breaking changes to the monkey-patching interface or the `NesionEngine` API will trigger a major version bump.
