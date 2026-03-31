#!/usr/bin/env python3
"""🏃 Nesion Performance Benchmark & VRAM Analysis 🚀

This script measures the real-world performance benefits of the Nesion
KV-Cache Eviction Engine compared to standard Hugging Face inference.

It measures and plots:
- VRAM Usage (MB) vs Context Length
- Throughput (Tokens per second)
- Time-To-First-Token (TTFT)
- Output Coherence (ROUGE-L Score vs Baseline)

Usage:
    python benchmark.py --model microsoft/phi-2 --budget 0.3
    python benchmark.py --model meta-llama/Llama-3.2-1B

Results are saved as a CSV and a corresponding PNG chart.
"""

import argparse
import csv
import logging
import os
import time
from typing import Any

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from nesion import NesionEngine

# Suppress HF warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==============================================================================
# Helper Classes
# ==============================================================================

class TTFTLogger(LogitsProcessor):
    """LogitsProcessor to capture the exact Time-To-First-Token (TTFT)."""
    
    def __init__(self) -> None:
        self.start_time: float = time.perf_counter()
        self.ttft: float | None = None
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.ttft is None:
            self.ttft = (time.perf_counter() - self.start_time) * 1000  # ms
        return scores


def calculate_rouge_l(reference: str, hypothesis: str) -> float:
    """Calculate ROUGE-L score between texts to check coherence.
    Falls back to simple word overlap if rouge-score is not installed.
    """
    try:
        from rouge_score import rouge_scorer  # noqa: PLC0415
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    except ImportError:
        # Fallback to IoU overlap logic 
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        if not ref_words or not hyp_words:
            return 0.0
        intersection = ref_words.intersection(hyp_words)
        return len(intersection) / float(len(ref_words.union(hyp_words)))

# ==============================================================================
# Benchmarking Core
# ==============================================================================

def run_single_benchmark(
    model: Any, 
    tokenizer: Any, 
    prompt: str, 
    max_new_tokens: int,
    use_nesion: bool = False,
    budget: float = 0.3
) -> dict[str, Any]:
    """Run a single performance benchmark cycle and collect metrics."""
    # Move inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    # Pre-execution cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Setup tools
    ttft_logger = TTFTLogger()
    engine = None
    
    if use_nesion:
        engine = NesionEngine(model, cache_budget=budget)
        engine.apply()
        
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[ttft_logger],
            use_cache=True,
            # Force greedy decoding to ensure deterministic-like generation
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    
    # Extract results
    gen_tokens = outputs.shape[1] - input_len
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    if engine:
        engine.remove()

    return {
        "text": text,
        "peak_vram_mb": peak_vram,
        "ttft_ms": ttft_logger.ttft or 0.0,
        "total_time_s": total_time,
        "tps": (gen_tokens - 1) / (total_time - ((ttft_logger.ttft or 0.0)/1000)),
        "gen_tokens": gen_tokens,
        "context_len": input_len + gen_tokens
    }

# ==============================================================================
# Main Runner
# ==============================================================================

def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Nesion VRAM & Perf Benchmark")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="HF Model ID")
    parser.add_argument("--budget", type=float, default=0.3, help="Nesion cache budget (0-1)")
    parser.add_argument("--prompt-size", type=int, default=128, help="Initial prompt length")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        msg = "⚠️  WARNING: Running on CPU. VRAM metrics reflect RAM; throughput will be low."
        print(msg)
        
    print(f"\n🚀 Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()
    
    # Create deterministic seed prompt
    line = "The theory of general relativity, published by Albert Einstein in 1915, "
    line += "is a geometric theory of gravitation that "
    prompt = line * (args.prompt_size // 15 + 1)
    
    # -------------------------------------------------------------------------
    # 1. Performance Face-off (Metrics Table)
    # -------------------------------------------------------------------------
    print("\n⚔️  Phase 1: Performance Face-off (Max Context ≈ 500)")
    bench_tokens = 300
    
    # Run Baseline
    print("   -> Running Baseline (Full KV-Cache)...")
    base_res = run_single_benchmark(model, tokenizer, prompt, bench_tokens, use_nesion=False)
    
    # Run Nesion
    print(f"   -> Running Nesion (Budget = {args.budget*100:.0f}%)...")
    nesion_res = run_single_benchmark(
        model, tokenizer, prompt, bench_tokens, use_nesion=True, budget=args.budget
    )
    
    # Calculate Coherence
    rouge_score = calculate_rouge_l(base_res["text"], nesion_res["text"])
    
    # Display Table
    print("\n" + "="*80)
    print(f"{'Metric':<25} | {'Baseline (No Eviction)':<22} | {'Nesion (H2O)':<22}")
    print("-" * 80)
    print(f"{'Peak VRAM (MB)':<25} | {base_res['peak_vram_mb']:>19.1f} MB | "
          f"{nesion_res['peak_vram_mb']:>19.1f} MB "
          f"({(1 - nesion_res['peak_vram_mb']/base_res['peak_vram_mb'])*-100:+.1f}%)")
    print(f"{'Throughput (Tokens/s)':<25} | {base_res['tps']:>19.1f} T/s | "
          f"{nesion_res['tps']:>19.1f} T/s "
          f"({(nesion_res['tps']/base_res['tps'] - 1)*100:+.1f}%)")
    print(f"{'TTFT (Prefill Latency)':<25} | {base_res['ttft_ms']:>19.1f} ms | "
          f"{nesion_res['ttft_ms']:>19.1f} ms")
    print(f"{'ROUGE-L Coherence':<25} | {'1.000 (Reference)':>22} | {rouge_score:>22.3f}")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 2. VRAM Scaling Analysis (Graph Generation)
    # -------------------------------------------------------------------------
    print("\n📈 Phase 2: VRAM Scaling Analysis")
    ctx_lengths = [100, 300, 600, 1000]
    base_vram = []
    nesion_vram = []
    
    for tokens in ctx_lengths:
        print(f"   -> Profiling context length: {args.prompt_size + tokens}...")
        
        # Baseline
        res = run_single_benchmark(model, tokenizer, prompt, tokens, use_nesion=False)
        base_vram.append(res['peak_vram_mb'])
        
        # Nesion
        res_ns = run_single_benchmark(
            model, tokenizer, prompt, tokens, use_nesion=True, budget=args.budget
        )
        nesion_vram.append(res_ns['peak_vram_mb'])
        
    # Generate Matplotlib chart
    if not HAS_MATPLOTLIB:
        print("\n⚠️  WARNING: matplotlib not found. Skipping plot generation.")
        print("   -> Tip: Install with 'pip install matplotlib' to see visual trends.")
    else:
        plt.figure(figsize=(10, 6), dpi=150)
    
    true_ctx = [args.prompt_size + t for t in ctx_lengths]
    plt.plot(
        true_ctx, base_vram, marker='o', label="Baseline (Full KV-Cache)", 
        linewidth=2, color="#E53935"
    )
    plt.plot(
        true_ctx, nesion_vram, marker='s', label=f"Nesion (Budget {args.budget*100:.0f}%)", 
        linewidth=2, color="#43A047"
    )
    
    plt.title(
        f"KV-Cache VRAM Scaling: Baseline vs Nesion\nModel: {args.model}", 
        fontsize=14, pad=15
    )
    plt.xlabel("Total Sequences Length (Tokens)", fontsize=12)
    plt.ylabel("Peak VRAM Usage (MB)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.fill_between(true_ctx, base_vram, nesion_vram, color='#A5D6A7', alpha=0.3)
    
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)
    
    # Save CSV
    sanitize_model = args.model.replace("/", "_")
    csv_path = f"results/benchmark_{sanitize_model}.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Context Length", "Baseline VRAM (MB)", "Nesion VRAM (MB)", "Ahorro %"])
        for i in range(len(true_ctx)):
            sav = (1 - (nesion_vram[i]/base_vram[i])) * 100
            writer.writerow([true_ctx[i], base_vram[i], nesion_vram[i], f"{sav:.1f}%"])

    # Save PNG
    if HAS_MATPLOTLIB:
        png_path = f"results/vram_scaling_{sanitize_model}.png"
        plt.savefig(png_path, bbox_inches="tight")
        print(f"   - Plot: {png_path}\n")
    else:
        print("\n")

    print("💾 Results exported successfully!")
    print(f"   - CSV:  {csv_path}")


if __name__ == "__main__":
    main()
