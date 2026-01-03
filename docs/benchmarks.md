# Benchmarks

All benchmarks measured on RTX 3090.

## Peak Throughput

**49.6M steps/second** with:
- `batch_size=32768`
- `world_size=16`
- `render_resolution=32x32`

## Throughput Scaling

![Throughput Surface](../assets/surface.png)

Throughput depends on three hyperparameters: batch size, world size, and render resolution.

### Hyperparameter Impact

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `batch_size` | Parallel environment count | Higher is better until GPU saturates (sweet spot: 16K-32K) |
| `world_size` | Voxel world dimensions | Smaller worlds are faster (16 vs 32 vs 48) |
| `render_width/height` | Observation resolution | Major impact: 32px is 4x faster than 64px |
| `max_steps` | Raymarching iterations | Lower is faster but may cause visual artifacts |
| `view_distance` | Ray cutoff distance | Lower reduces wasted ray steps |

## Full Results

![Throughput Surface](../assets/throughput_heatmap.png)

| World | Resolution | Batch 4K | Batch 8K | Batch 16K | Batch 32K |
|-------|------------|----------|----------|-----------|-----------|
| 16x16x16 | 32x32 | 24.5M | 34.1M | 42.4M | **49.6M** |
| 16x16x16 | 64x64 | 11.1M | 12.6M | 13.4M | 13.7M |
| 16x16x16 | 128x128 | 3.3M | 3.4M | 3.5M | 3.5M |
| 32x32x32 | 32x32 | 8.1M | 9.2M | 9.9M | 10.2M |
| 32x32x32 | 64x64 | 2.8M | 2.8M | 2.9M | 2.9M |
| 32x32x32 | 128x128 | 0.8M | 0.8M | 0.8M | 0.8M |
| 48x48x48 | 32x32 | 5.5M | 5.9M | 6.3M | - |
| 48x48x48 | 64x64 | 1.8M | 1.9M | 1.9M | - |

## Memory Usage

Approximate GPU memory per configuration:

| Config | Voxels | Obs Buffer | Total |
|--------|--------|------------|-------|
| 4K batch, 32 world, 64px | 128 MB | 192 MB | ~400 MB |
| 16K batch, 32 world, 64px | 512 MB | 768 MB | ~1.5 GB |
| 32K batch, 16 world, 32px | 128 MB | 384 MB | ~800 MB |

## Running Benchmarks

```bash
# Generate throughput plots
python scripts/plot_throughput.py

# Quick benchmark
python -c "
from minecu import MineEnv
import torch, time

env = MineEnv(batch_size=16384, world_size=32, render_width=64, render_height=64)
obs = env.reset()
actions = torch.randint(0, 11, (16384,), device='cuda')

# Warmup
for _ in range(20):
    obs, r, d = env.step(actions)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(100):
    obs, r, d = env.step(actions)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f'{16384 * 100 / elapsed / 1e6:.2f}M steps/sec')
"
```
