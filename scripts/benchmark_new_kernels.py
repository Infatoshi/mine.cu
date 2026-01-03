#!/usr/bin/env python3
"""
Benchmark script for experimental render kernel optimizations.

Tests:
- baseline: Original render kernel (copy)
- warp_vote: Warp voting for uniform execution paths
- early_term: Early ray termination for rays missing world bounds
- combined: Both optimizations together

Configuration:
- Batch size: 16384
- World size: 32x32x32
- Resolution: 64x64
- Warmup: 20 iterations
- Benchmark: 100 iterations
"""

import torch
import time
import numpy as np

# Import experimental kernels
try:
    from minecu import _C_experimental as exp
except ImportError as e:
    print(f"ERROR: Could not import experimental kernels: {e}")
    print("Make sure to build with: CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH uv pip install -e .")
    exit(1)

# Import original kernel for reference
try:
    from minecu import _C as orig
except ImportError:
    orig = None
    print("WARNING: Original kernels not available, skipping original comparison")


def create_test_world(batch_size: int, world_size: int, device: torch.device) -> torch.Tensor:
    """Create a flat world with ground at height world_size/4."""
    voxels = torch.full(
        (batch_size, world_size, world_size, world_size),
        -1,  # AIR
        dtype=torch.int8,
        device=device
    )
    ground_height = world_size // 4

    # Fill ground layers
    voxels[:, :ground_height-1, :, :] = 1  # DIRT
    voxels[:, ground_height-1, :, :] = 0   # GRASS
    voxels[:, 0, :, :] = 12  # BEDROCK

    return voxels


def create_cameras(batch_size: int, world_size: int, device: torch.device) -> torch.Tensor:
    """Create camera positions looking at the world."""
    cameras = torch.zeros((batch_size, 5), dtype=torch.float32, device=device)

    # Position cameras in the middle of the world, above ground
    cameras[:, 0] = world_size / 2  # x
    cameras[:, 1] = world_size / 4 + 2  # y (above ground)
    cameras[:, 2] = world_size / 2  # z

    # Vary yaw slightly across batch for diversity
    cameras[:, 3] = torch.linspace(0, 0.5, batch_size, device=device)  # yaw
    cameras[:, 4] = -0.2  # pitch (looking slightly down)

    return cameras


def benchmark_kernel(
    render_fn,
    voxels: torch.Tensor,
    cameras: torch.Tensor,
    output: torch.Tensor,
    warmup_iters: int = 20,
    bench_iters: int = 100,
    max_steps: int = 64,
    view_distance: float = 32.0,
    fov_degrees: float = 70.0
) -> dict:
    """Benchmark a render kernel and return timing statistics."""

    # Warmup
    for _ in range(warmup_iters):
        render_fn(voxels, cameras, output, max_steps, view_distance, fov_degrees)
    torch.cuda.synchronize()

    # Benchmark
    times_us = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        render_fn(voxels, cameras, output, max_steps, view_distance, fov_degrees)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_us.append((end - start) * 1e6)

    times_us = np.array(times_us)
    return {
        "min_us": float(np.min(times_us)),
        "max_us": float(np.max(times_us)),
        "mean_us": float(np.mean(times_us)),
        "std_us": float(np.std(times_us)),
        "median_us": float(np.median(times_us)),
    }


def verify_correctness(
    ref_fn,
    test_fn,
    voxels: torch.Tensor,
    cameras: torch.Tensor,
    name: str,
    max_steps: int = 64,
    view_distance: float = 32.0,
    fov_degrees: float = 70.0,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> bool:
    """Verify that test_fn produces the same output as ref_fn."""
    batch_size = voxels.size(0)
    height = 64
    width = 64
    device = voxels.device

    ref_output = torch.zeros((batch_size, height, width, 3), dtype=torch.float32, device=device)
    test_output = torch.zeros((batch_size, height, width, 3), dtype=torch.float32, device=device)

    ref_fn(voxels, cameras, ref_output, max_steps, view_distance, fov_degrees)
    test_fn(voxels, cameras, test_output, max_steps, view_distance, fov_degrees)
    torch.cuda.synchronize()

    match = torch.allclose(ref_output, test_output, atol=atol, rtol=rtol)

    if not match:
        diff = (ref_output - test_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        num_mismatch = (diff > atol).sum().item()
        total_elements = ref_output.numel()
        print(f"  {name}: MISMATCH - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
              f"mismatches={num_mismatch}/{total_elements} ({100*num_mismatch/total_elements:.2f}%)")
    else:
        print(f"  {name}: PASS")

    return match


def main():
    print("=" * 70)
    print("Experimental Render Kernel Benchmark")
    print("=" * 70)

    # Configuration
    batch_size = 16384
    world_size = 32
    width = 64
    height = 64
    max_steps = 64
    view_distance = 32.0
    fov_degrees = 70.0
    warmup_iters = 20
    bench_iters = 100

    device = torch.device("cuda")

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  World size: {world_size}x{world_size}x{world_size}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Max steps: {max_steps}")
    print(f"  View distance: {view_distance}")
    print(f"  FOV: {fov_degrees} degrees")
    print(f"  Warmup iterations: {warmup_iters}")
    print(f"  Benchmark iterations: {bench_iters}")

    # Create test data
    print("\nCreating test world...")
    voxels = create_test_world(batch_size, world_size, device)
    cameras = create_cameras(batch_size, world_size, device)
    output = torch.zeros((batch_size, height, width, 3), dtype=torch.float32, device=device)

    total_pixels = batch_size * height * width
    print(f"  Total pixels per frame: {total_pixels:,}")

    # Define kernels to benchmark
    kernels = {
        "baseline": exp.render_baseline,
        "warp_vote": exp.render_warp_vote,
        "early_term": exp.render_early_term,
        "combined": exp.render_combined,
    }

    # Add original kernel if available
    if orig is not None:
        kernels["original"] = orig.render

    # Correctness verification
    print("\n" + "-" * 70)
    print("Correctness Verification (vs baseline)")
    print("-" * 70)

    ref_fn = exp.render_baseline
    all_correct = True
    for name, fn in kernels.items():
        if name == "baseline":
            print(f"  baseline: REFERENCE")
            continue
        correct = verify_correctness(ref_fn, fn, voxels, cameras, name,
                                     max_steps, view_distance, fov_degrees)
        all_correct = all_correct and correct

    if not all_correct:
        print("\nWARNING: Some kernels produced different results!")

    # Benchmarking
    print("\n" + "-" * 70)
    print("Performance Benchmarks")
    print("-" * 70)

    results = {}
    for name, fn in kernels.items():
        print(f"  Benchmarking {name}...")
        results[name] = benchmark_kernel(
            fn, voxels, cameras, output,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            max_steps=max_steps,
            view_distance=view_distance,
            fov_degrees=fov_degrees
        )

    # Results table
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    baseline_mean = results["baseline"]["mean_us"]

    print(f"\n{'Kernel':<15} {'Mean (us)':<12} {'Std (us)':<10} {'Min (us)':<10} {'Max (us)':<10} {'Speedup':<10}")
    print("-" * 70)

    for name, stats in results.items():
        speedup = baseline_mean / stats["mean_us"]
        print(f"{name:<15} {stats['mean_us']:<12.1f} {stats['std_us']:<10.1f} "
              f"{stats['min_us']:<10.1f} {stats['max_us']:<10.1f} {speedup:<10.3f}x")

    # Calculate throughput
    print("\n" + "-" * 70)
    print("Throughput Analysis")
    print("-" * 70)

    for name, stats in results.items():
        steps_per_sec = total_pixels / (stats["mean_us"] / 1e6)
        print(f"  {name}: {steps_per_sec/1e6:.2f}M pixels/sec")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Find best performing
    best_name = min(results.keys(), key=lambda k: results[k]["mean_us"])
    best_speedup = baseline_mean / results[best_name]["mean_us"]

    print(f"\nBest performer: {best_name} ({best_speedup:.3f}x vs baseline)")

    # Individual optimization analysis
    warp_speedup = baseline_mean / results["warp_vote"]["mean_us"]
    early_speedup = baseline_mean / results["early_term"]["mean_us"]
    combined_speedup = baseline_mean / results["combined"]["mean_us"]

    print(f"\nOptimization Impact:")
    print(f"  Warp voting only:      {warp_speedup:.3f}x")
    print(f"  Early termination only: {early_speedup:.3f}x")
    print(f"  Combined:              {combined_speedup:.3f}x")

    # Warp voting analysis
    warp_benefit = "beneficial" if warp_speedup > 1.0 else "not beneficial"
    print(f"\nWarp voting is {warp_benefit} for this workload (speedup: {warp_speedup:.3f}x)")

    if warp_speedup < 1.0:
        print("  Note: Warp voting overhead may exceed benefit when rays are highly divergent")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
