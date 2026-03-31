#!/usr/bin/env python3
"""🏋️ Nesion Industrial-Strength Stress & Endurance Test 🏗️

This script pushes Nesion to the limit by simulating high-concurrency 
inference and long-sequence generation loops while monitoring VRAM stability.
"""

import argparse
import logging
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nesion import NesionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StressTester:
    def __init__(self, model_id: str, budget: float, duration_mins: int, concurrency: int):
        self.model_id = model_id
        self.budget = budget
        self.duration_secs = duration_mins * 60
        self.concurrency = concurrency
        self.stop_event = threading.Event()
        
        logger.info(
            f"🚀 Initializing Stress Test: {model_id} | Budget: {budget} | "
            f"Concurrency: {concurrency}"
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True
        )
        self.model.eval()
        
    def _worker(self, worker_id: int):
        """Worker thread simulating continuous generation."""
        prompt = (
            "Explain the importance of KV-Cache eviction in distributed LLM training "
            "systems with high latency."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        cycle = 0
        while not self.stop_event.is_set():
            cycle += 1
            try:
                with torch.no_grad():
                    # We simulate high context by generating many tokens
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                if worker_id == 0 and cycle % 5 == 0:
                    vram_mb = 0.0
                    if self.device == "cuda":
                        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    logger.info(f"  [Monitor] Peak VRAM: {vram_mb:.1f} MB | Cycles: {cycle}")
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} failed in cycle {cycle}: {e}")
                
    def run(self):
        """Orchestrate the stress test."""
        logger.info(f"🔥 Starting load for {self.duration_secs/60:.1f} minutes...")
        
        with NesionEngine(self.model, cache_budget=self.budget):
            threads: list[threading.Thread] = []
            for i in range(self.concurrency):
                t = threading.Thread(target=self._worker, args=(i,), daemon=True)
                t.start()
                threads.append(t)
                
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.duration_secs:
                time.sleep(1)
                
            self.stop_event.set()
            for t in threads:
                t.join(timeout=5)
                
        logger.info("✅ Stress test completed successfully.")
        
        if self.device == "cuda":
            final_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            logger.info(f"📊 Final Peak VRAM recorded: {final_vram:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Nesion High-Load Stress Test")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="HF Model ID")
    parser.add_argument("--budget", type=float, default=0.3, help="Nesion budget (0-1)")
    parser.add_argument("--mins", type=int, default=2, help="Test duration in minutes")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Number of concurrent threads"
    )
    args = parser.parse_args()
    
    tester = StressTester(args.model, args.budget, args.mins, args.concurrency)
    tester.run()

if __name__ == "__main__":
    main()
