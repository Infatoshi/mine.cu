# LLM Navigation Guide

If you're an LLM helping with this codebase, start here.

## Quick Orientation

**mine.cu** is a high-performance batched voxel RL environment. All game logic runs on GPU via custom CUDA kernels. Peak throughput: 49.6M environment steps/second on RTX 3090.

## Key Files

| File | Purpose | When to read |
|------|---------|--------------|
| `src/kernels.cu` | All CUDA kernels | Optimizing rendering, adding features |
| `minecu/__init__.py` | Python MineEnv class | Modifying the gym-like interface |
| `minecu/config.py` | Hyperparameter configs | Adding/modifying presets |
| `src/bindings.cpp` | pybind11 bindings | Adding new kernel functions |

## Documentation

| Doc | Contents |
|-----|----------|
| [environment.md](environment.md) | Action space, block types, env parameters |
| [benchmarks.md](benchmarks.md) | Throughput data, memory usage, how to benchmark |
| [architecture.md](architecture.md) | Project structure, kernel details, data layouts |

## Common Tasks

### Adding a new action
1. Update action handling in `minecu/__init__.py` (step method)
2. Add any new kernel logic in `src/kernels.cu`
3. Update action space docs in `docs/environment.md`

### Optimizing kernels
1. Read current implementation in `src/kernels.cu`
2. Key optimizations already applied: AABB early termination, loop unrolling, `__ldg()`, block_size=128
3. Benchmark with `scripts/plot_throughput.py` data format
4. Update `docs/benchmarks.md` with new results

### Adding a new block type
1. Add constant in `src/kernels.cu` (after line ~22)
2. Add color in `BLOCK_COLORS` array
3. Update `docs/environment.md` block table

### Modifying world generation
1. Edit `generate_world_kernel` or `place_tree_kernel` in `src/kernels.cu`
2. Consider adding new generation kernels for complex terrain

## Build & Test

```bash
# Rebuild after kernel changes
CUDA_HOME=/usr/local/cuda-12.8 uv pip install -e .

# Quick test
uv run python -c "from minecu import MineEnv; e = MineEnv(batch_size=1024); print(e.reset().shape)"

# Run training example
uv run python examples/train_wood.py
```

## Code Conventions

- Use `uv` for Python package management
- CUDA kernels use `__restrict__` and `__ldg()` for read-only data
- Block size 128 for render kernel, 256 for others
- Benchmark before/after any kernel changes

## Gotchas

- Multiple CUDA versions: Always set `CUDA_HOME=/usr/local/cuda-12.8` when building
- Voxel layout is `[B, Y, X, Z]` - Y is vertical (height)
- Block ID -1 is air, 0+ are solid blocks
- Cameras store `[x, y, z, yaw, pitch]` - eye height offset applied in kernel
