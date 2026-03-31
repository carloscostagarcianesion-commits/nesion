# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-31
### Added
- Initial release of Nesion KV-Cache Eviction Engine.
- `H2OEvictor` module for memory compression tracking and pruning.
- `NesionEngine` context manager for seamless Hugging Face model patch.
- Comprehensive test suite with numerical stability and regression checks.
- Benchmark scripts for throughput and VRAM scaling analysis.
- Premium web dashboard with interactive VRAM calculator and model comparisons.
- Support for Llama 3, Mistral, Qwen, Gemma, Phi, and more.

### Fixed
- Fixed `ModuleNotFoundError` in benchmark scripts when `matplotlib` is missing.
- Enhanced robustness across cross-platform CUDA/CPU deployments.
