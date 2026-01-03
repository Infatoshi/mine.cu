#!/usr/bin/env python3
"""
Benchmark script for comparing render kernel variants.

Usage:
    cd /home/infatoshi/cuda/mine.cu
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH uv pip install -e .
    uv run python scripts/benchmark_kernels.py

Configuration:
    - Batch size: 16384 environments
    - World size: 32x32x32 voxels
    - Resolution: 64x64 pixels
    - Warmup: 20 iterations
    - Benchmark: 100 iterations
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple
import sys


@dataclass
class BenchmarkConfig:
    batch_size: int = 16384
    world_size: int = 32
    resolution: int = 64
    max_steps: int = 64
    view_distance: float = 32.0
    fov_degrees: float = 70.0
    warmup_iters: int = 20
    benchmark_iters: int = 100
    block_sizes: List[int] = None

    def __post_init__(self):
        if self.block_sizes is None:
            self.block_sizes = [128, 256, 512]


@dataclass
class BenchmarkResult:
    name: str
    block_size: int
    times_us: List[float]  # microseconds

    @property
    def min_us(self) -> float:
        return min(self.times_us)

    @property
    def max_us(self) -> float:
        return max(self.times_us)

    @property
    def mean_us(self) -> float:
        return np.mean(self.times_us)

    @property
    def std_us(self) -> float:
        return np.std(self.times_us)


def create_test_data(config: BenchmarkConfig, device: str = "cuda"):
    """Create test voxel world and camera data."""
    B = config.batch_size
    W = config.world_size
    H = config.resolution

    # Voxel world: [B, Y, X, Z] - mostly air with ground
    voxels = torch.full((B, W, W, W), -1, dtype=torch.int8, device=device)

    # Fill ground (y < 8) with dirt/grass
    ground_height = W // 4
    for y in range(ground_height):
        if y == 0:
            voxels[:, y, :, :] = 12  # Bedrock
        elif y < ground_height - 1:
            voxels[:, y, :, :] = 1   # Dirt
        else:
            voxels[:, y, :, :] = 0   # Grass

    # Camera positions: [B, 5] = (x, y, z, yaw, pitch)
    cameras = torch.zeros((B, 5), dtype=torch.float32, device=device)
    cameras[:, 0] = W / 2.0      # x: center
    cameras[:, 1] = ground_height + 2.0  # y: above ground
    cameras[:, 2] = W / 2.0      # z: center
    cameras[:, 3] = 0.0          # yaw
    cameras[:, 4] = -0.2         # pitch: slightly down

    # Output buffer
    output = torch.zeros((B, H, H, 3), dtype=torch.float32, device=device)

    return voxels, cameras, output


def benchmark_kernel(
    render_fn: Callable,
    voxels: torch.Tensor,
    cameras: torch.Tensor,
    output: torch.Tensor,
    config: BenchmarkConfig,
    block_size: int = 256,
    name: str = "kernel"
) -> BenchmarkResult:
    """Benchmark a single kernel variant."""
    # Warmup
    for _ in range(config.warmup_iters):
        render_fn(
            voxels, cameras, output,
            config.max_steps, config.view_distance, config.fov_degrees,
            block_size
        )
    torch.cuda.synchronize()

    # Benchmark with CUDA events for precise timing
    times = []
    for _ in range(config.benchmark_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        render_fn(
            voxels, cameras, output,
            config.max_steps, config.view_distance, config.fov_degrees,
            block_size
        )
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # Convert ms to us

    return BenchmarkResult(name=name, block_size=block_size, times_us=times)


def benchmark_baseline(
    voxels: torch.Tensor,
    cameras: torch.Tensor,
    output: torch.Tensor,
    config: BenchmarkConfig
) -> BenchmarkResult:
    """Benchmark the baseline kernel from minecu._C"""
    from minecu import _C

    # Baseline doesn't have block_size parameter
    def baseline_render(voxels, cameras, output, max_steps, view_distance, fov_degrees, block_size):
        _C.render(voxels, cameras, output, max_steps, view_distance, fov_degrees)

    # Warmup
    for _ in range(config.warmup_iters):
        _C.render(voxels, cameras, output, config.max_steps, config.view_distance, config.fov_degrees)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config.benchmark_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _C.render(voxels, cameras, output, config.max_steps, config.view_distance, config.fov_degrees)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)

    return BenchmarkResult(name="baseline", block_size=256, times_us=times)


def verify_correctness(
    baseline_output: torch.Tensor,
    optimized_output: torch.Tensor,
    name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> Tuple[bool, float]:
    """Verify that optimized kernel produces same output as baseline."""
    is_close = torch.allclose(baseline_output, optimized_output, atol=atol, rtol=rtol)
    max_diff = (baseline_output - optimized_output).abs().max().item()
    return is_close, max_diff


def print_results_table(results: List[BenchmarkResult], baseline_mean: float):
    """Print formatted results table."""
    print("\n" + "=" * 85)
    print(f"{'Variant':<30} {'Block':<6} {'Min (us)':<12} {'Max (us)':<12} {'Mean (us)':<12} {'Speedup':<10}")
    print("=" * 85)

    for r in results:
        speedup = baseline_mean / r.mean_us if r.mean_us > 0 else 0
        speedup_str = f"{speedup:.2f}x" if r.name != "baseline" else "-"
        print(f"{r.name:<30} {r.block_size:<6} {r.min_us:<12.2f} {r.max_us:<12.2f} {r.mean_us:<12.2f} {speedup_str:<10}")

    print("=" * 85)


def main():
    print("=" * 60)
    print("CUDA Render Kernel Benchmark")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Configuration
    config = BenchmarkConfig()
    print(f"\nConfiguration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  World size: {config.world_size}x{config.world_size}x{config.world_size}")
    print(f"  Resolution: {config.resolution}x{config.resolution}")
    print(f"  Warmup: {config.warmup_iters} iterations")
    print(f"  Benchmark: {config.benchmark_iters} iterations")
    print(f"  Block sizes: {config.block_sizes}")

    # Create test data
    print("\nCreating test data...")
    voxels, cameras, output = create_test_data(config, device)
    baseline_output = torch.zeros_like(output)
    optimized_output = torch.zeros_like(output)

    total_pixels = config.batch_size * config.resolution * config.resolution
    print(f"Total pixels per frame: {total_pixels:,}")

    # Import modules
    print("\nLoading CUDA modules...")
    try:
        from minecu import _C
        print("  - minecu._C loaded")
    except ImportError as e:
        print(f"ERROR: Failed to import minecu._C: {e}")
        print("Run: cd /home/infatoshi/cuda/mine.cu && CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH uv pip install -e .")
        sys.exit(1)

    try:
        from minecu import _C_opt
        print("  - minecu._C_opt loaded")
    except ImportError as e:
        print(f"ERROR: Failed to import minecu._C_opt: {e}")
        print("Run: cd /home/infatoshi/cuda/mine.cu && CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH uv pip install -e .")
        sys.exit(1)

    results = []

    # Benchmark baseline
    print("\n" + "-" * 60)
    print("Benchmarking baseline kernel...")
    baseline_result = benchmark_baseline(voxels, cameras, baseline_output, config)
    results.append(baseline_result)
    print(f"  Baseline: {baseline_result.mean_us:.2f} us (min: {baseline_result.min_us:.2f}, max: {baseline_result.max_us:.2f})")

    # Kernel variants to test
    variants = [
        ("v1_shared_colors", _C_opt.render_v1_shared_colors),
        ("v2_ldg", _C_opt.render_v2_ldg),
        ("v3_branchless", _C_opt.render_v3_branchless),
        ("v4_unrolled", _C_opt.render_v4_unrolled),
        ("v5_combined", _C_opt.render_v5_combined),
        ("v6_fast_math", _C_opt.render_v6_fast_math),
        ("v7_full", _C_opt.render_v7_full),
        ("v8_unrolled_ldg", _C_opt.render_v8_unrolled_ldg),
        ("v9_best", _C_opt.render_v9_best),
    ]

    # Benchmark each variant at each block size
    print("\n" + "-" * 60)
    print("Benchmarking optimized variants...")

    correctness_results = {}

    for variant_name, variant_fn in variants:
        for block_size in config.block_sizes:
            name = f"{variant_name}_bs{block_size}"
            print(f"  Testing {name}...", end=" ", flush=True)

            result = benchmark_kernel(
                variant_fn, voxels, cameras, optimized_output,
                config, block_size, name
            )
            results.append(result)

            # Verify correctness against baseline
            is_correct, max_diff = verify_correctness(baseline_output, optimized_output, name)
            correctness_results[name] = (is_correct, max_diff)

            status = "OK" if is_correct else f"DIFF (max={max_diff:.2e})"
            speedup = baseline_result.mean_us / result.mean_us
            print(f"{result.mean_us:.2f} us ({speedup:.2f}x) [{status}]")

    # Print results table
    print_results_table(results, baseline_result.mean_us)

    # Correctness summary
    print("\nCorrectness Verification:")
    print("-" * 60)
    all_correct = True
    for name, (is_correct, max_diff) in correctness_results.items():
        status = "PASS" if is_correct else "FAIL"
        print(f"  {name:<30}: {status} (max diff: {max_diff:.2e})")
        if not is_correct:
            all_correct = False

    if all_correct:
        print("\nAll optimized kernels produce outputs matching baseline (atol=1e-5, rtol=1e-5)")
    else:
        print("\nWARNING: Some kernels have numerical differences from baseline")

    # Find best variant
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Exclude baseline from best search
    optimized_results = [r for r in results if r.name != "baseline"]
    if optimized_results:
        best = min(optimized_results, key=lambda r: r.mean_us)
        speedup = baseline_result.mean_us / best.mean_us
        print(f"Best variant: {best.name}")
        print(f"  Mean time: {best.mean_us:.2f} us")
        print(f"  Speedup: {speedup:.2f}x vs baseline")

        # Calculate throughput
        baseline_fps = 1e6 / baseline_result.mean_us
        best_fps = 1e6 / best.mean_us
        baseline_steps_per_sec = baseline_fps * config.batch_size
        best_steps_per_sec = best_fps * config.batch_size

        print(f"\nThroughput:")
        print(f"  Baseline: {baseline_steps_per_sec/1e6:.2f}M steps/sec")
        print(f"  Best:     {best_steps_per_sec/1e6:.2f}M steps/sec")

    # Recommendations
    print("\n" + "-" * 60)
    print("Optimization Analysis:")
    print("-" * 60)

    # Group results by variant (ignoring block size)
    variant_bests = {}
    for r in optimized_results:
        variant = r.name.rsplit("_bs", 1)[0]
        if variant not in variant_bests or r.mean_us < variant_bests[variant].mean_us:
            variant_bests[variant] = r

    print("\nBest block size per variant:")
    for variant, r in sorted(variant_bests.items(), key=lambda x: x[1].mean_us):
        speedup = baseline_result.mean_us / r.mean_us
        print(f"  {variant:<25}: block_size={r.block_size}, {r.mean_us:.2f} us ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
