"""CLI entry points for Nesion benchmarks and utilities."""

from __future__ import annotations

import argparse
import sys


def run_benchmark() -> None:
    """Entry point for ``nesion-benchmark`` CLI command."""
    parser = argparse.ArgumentParser(
        prog="nesion-benchmark",
        description="Run Nesion KV-Cache eviction benchmarks",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model name or path (default: gpt2)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Cache budget sizes to benchmark (default: 256 512 1024)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of tokens to generate per run (default: 200)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV (optional)",
    )

    args = parser.parse_args()

    # Lazy imports so CLI --help is fast
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as e:
        print(f"Error: Missing dependency — {e}")
        print("Install with: pip install nesion[benchmark]")
        sys.exit(1)

    from nesion import NesionConfig, NesionEngine  # noqa: PLC0415

    print("=" * 60)
    print("  Nesion Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Budgets: {args.budget}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"{'='*60}\n")

    # Load model
    device_map = args.device if args.device != "auto" else "auto"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    results = []

    for budget in args.budget:
        print(f"\n--- Budget: {budget} tokens ---")

        config = NesionConfig(
            max_cache_size=budget,
            heavy_hitter_ratio=0.5,
            recent_ratio=0.5,
            track_vram=True,
            verbose=True,
        )

        engine = NesionEngine(model=model, config=config)

        import time  # noqa: PLC0415
        start = time.perf_counter()
        output_ids = engine.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.perf_counter() - start

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        stats = engine.get_stats()
        tokens_gen = output_ids.shape[1] - inputs["input_ids"].shape[1]

        result = {
            "budget": budget,
            "tokens_generated": tokens_gen,
            "time_s": round(elapsed, 2),
            "tokens_per_sec": round(tokens_gen / elapsed, 1),
            "total_evictions": stats["total_evictions"],
            "total_tokens_evicted": stats["total_tokens_evicted"],
        }
        results.append(result)

        print(f"  Generated: {tokens_gen} tokens in {elapsed:.2f}s")
        print(f"  Speed: {tokens_gen / elapsed:.1f} tok/s")
        print(f"  Evictions: {stats['total_evictions']}")
        print(f"  Preview: {generated_text[:100]}...")

        engine.remove_hooks()

    # Print summary table
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"{'Budget':>8} {'Tokens':>8} {'Time(s)':>8} {'Tok/s':>8} {'Evictions':>10}")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in results:
        print(
            f"{r['budget']:>8} {r['tokens_generated']:>8} "
            f"{r['time_s']:>8} {r['tokens_per_sec']:>8} "
            f"{r['total_evictions']:>10}"
        )

    # Save CSV if requested
    if args.output:
        try:
            import pandas as pd  # noqa: PLC0415
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
        except ImportError:
            print("\nInstall pandas to save CSV: pip install pandas")


if __name__ == "__main__":
    run_benchmark()
