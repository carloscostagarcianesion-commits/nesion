#!/usr/bin/env python3
"""🏃 Nesion VRAM Profiler 🚀

This script provides a step-by-step memory profile of an LLM generation
with and without Nesion. It generates a CSV log of VRAM usage per token step.

Usage:
    python scripts/profile_memory.py --model microsoft/phi-2 --steps 100
"""

import argparse
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from nesion import NesionConfig, NesionEngine

# Suppress HF warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

class VRAMLogger(LogitsProcessor):
    """LogitsProcessor to capture VRAM usage at every generation step."""
    def __init__(self):
        self.history: list[float] = []
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Measure allocated memory in MB
        vram = torch.cuda.memory_allocated() / (1024 * 1024)
        self.history.append(vram)
        return scores

def profile(model_id: str, max_steps: int, budget: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📦 Loading {model_id} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    prompt = "In the deep silence of the cosmos, the first stars began to "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 1. Profile Baseline
    print("📉 Profiling Baseline (No Eviction)...")
    logger_base = VRAMLogger()
    with torch.no_grad():
        model.generate(
            **inputs, 
            max_new_tokens=max_steps, 
            logits_processor=[logger_base],
            use_cache=True,
            do_sample=False
        )

    # 2. Profile Nesion
    print(f"🚀 Profiling Nesion (Budget={budget:.1%})...")
    logger_nesion = VRAMLogger()
    config = NesionConfig(cache_budget=budget)
    with NesionEngine(model, config=config):
        with torch.no_grad():
            model.generate(
                **inputs, 
                max_new_tokens=max_steps, 
                logits_processor=[logger_nesion],
                use_cache=True,
                do_sample=False
            )

    # Export Results
    os.makedirs("results", exist_ok=True)
    filename = f"results/vram_profile_{model_id.replace('/', '_')}.csv"
    with open(filename, "w") as f:
        f.write("step,vram_baseline_mb,vram_nesion_mb,diff_mb\n")
        for i in range(min(len(logger_base.history), len(logger_nesion.history))):
            b = logger_base.history[i]
            n = logger_nesion.history[i]
            f.write(f"{i},{b:.2f},{n:.2f},{b-n:.2f}\n")
            
    print(f"✅ Profiling complete. Results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/phi-2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--budget", type=float, default=0.3)
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Profiling results will be based on CPU RAM.")
        
    profile(args.model, args.steps, args.budget)
