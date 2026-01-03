#!/usr/bin/env python3
"""
Benchmark throughput across hyperparameter configurations.
Uses the optimized render kernel with loop unrolling + __ldg.
"""

import torch
import time
import sys
sys.path.insert(0, '/home/infatoshi/cuda/mine.cu')

from minecu import MineEnv

# Benchmark configurations
CONFIGS = [
    # (world_size, resolution, batch_sizes)
    (16, 32, [4096, 8192, 16384, 32768]),
    (16, 64, [4096, 8192, 16384, 32768]),
    (16, 128, [4096, 8192, 16384, 32768]),
    (32, 32, [4096, 8192, 16384, 32768]),
    (32, 64, [4096, 8192, 16384, 32768]),
    (32, 128, [4096, 8192, 16384, 32768]),
    (48, 32, [4096, 8192, 16384]),
    (48, 64, [4096, 8192, 16384]),
    (48, 128, [4096, 8192, 16384]),
]

WARMUP_STEPS = 20
BENCH_STEPS = 100


def benchmark_config(world_size: int, resolution: int, batch_size: int) -> float:
    """Benchmark a single configuration and return steps/sec."""
    try:
        env = MineEnv(
            batch_size=batch_size,
            world_size=world_size,
            render_width=resolution,
            render_height=resolution,
            device="cuda",
        )

        obs = env.reset()
        actions = torch.randint(0, env.action_dim, (batch_size,), device="cuda")

        # Warmup
        for _ in range(WARMUP_STEPS):
            obs, rewards, dones = env.step(actions)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(BENCH_STEPS):
            obs, rewards, dones = env.step(actions)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        steps_per_sec = (batch_size * BENCH_STEPS) / elapsed

        del env
        torch.cuda.empty_cache()

        return steps_per_sec

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return 0.0


def main():
    print("=" * 70)
    print("THROUGHPUT BENCHMARK - Optimized Kernel (unroll + __ldg, block=128)")
    print("=" * 70)
    print(f"Warmup: {WARMUP_STEPS} steps, Benchmark: {BENCH_STEPS} steps")
    print()

    results = {}

    for world_size, resolution, batch_sizes in CONFIGS:
        for batch_size in batch_sizes:
            print(f"Testing world={world_size}, res={resolution}x{resolution}, batch={batch_size}...", end=" ", flush=True)

            throughput = benchmark_config(world_size, resolution, batch_size)

            if throughput > 0:
                results[(world_size, resolution, batch_size)] = throughput
                print(f"{throughput/1e6:.2f}M steps/sec")
            else:
                print("OOM")

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Print as Python dict for copy-paste into plot script
    print("data = {")
    for (ws, res, bs), throughput in sorted(results.items()):
        print(f"    ({ws}, {res}, {bs}): {throughput/1e6:.2f},")
    print("}")

    # Find peak
    if results:
        peak_config, peak_throughput = max(results.items(), key=lambda x: x[1])
        print()
        print(f"PEAK: {peak_throughput/1e6:.2f}M steps/sec at world={peak_config[0]}, res={peak_config[1]}, batch={peak_config[2]}")


if __name__ == "__main__":
    main()
