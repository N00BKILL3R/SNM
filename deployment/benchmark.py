from __future__ import annotations

import argparse
import os
import time

import psutil
import torch

from models.semantic_comm_model import SemanticCommModel
from utils.config import load_config


def model_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024 if path and os.path.exists(path) else 0.0


def run_bench(model, x, warmup=10, iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000 / iters
    throughput = x.size(0) / (latency_ms / 1000)
    return latency_ms, throughput


def main():
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    c = cfg["model"]["in_channels"]
    s = 32 if c == 3 else 28
    x = torch.randn(32, c, s, s)

    model = SemanticCommModel(cfg["model"]).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)

    lat, th = run_bench(model, x, cfg["deployment"]["benchmark_warmup"], cfg["deployment"]["benchmark_iters"])
    params = sum(p.numel() for p in model.parameters())
    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024

    print({"latency_ms": lat, "throughput": th, "params": params, "model_size_mb": model_size_mb(args.ckpt), "cpu_mem_mb": mem_mb})


if __name__ == "__main__":
    main()
