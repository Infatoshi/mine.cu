# mine.cu

High-performance batched voxel RL environment with custom CUDA kernels.

![Demo](assets/demo.gif)

## Features

- **Pure CUDA**: All rendering, physics, and game logic run on GPU
- **Batched**: Train thousands of agents in parallel
- **Fast**: Up to 49.6M environment steps per second on RTX 3090
- **Minimal**: Clean PyTorch integration, no JAX overhead
- **Zero-copy**: GPU tensors throughout, no CPU-GPU transfers

## Installation

```bash
git clone https://github.com/Infatoshi/mine.cu
cd mine.cu
./scripts/setup.sh
source .venv/bin/activate
```

## Quick Start

```python
import torch
from minecu import MineEnv

env = MineEnv(batch_size=4096, device="cuda")
obs = env.reset()  # [4096, 64, 64, 3]

for _ in range(100):
    actions = torch.randint(0, env.action_dim, (4096,), device="cuda")
    obs, rewards, dones = env.step(actions)
```

## Documentation

| Doc | Contents |
|-----|----------|
| [Environment](docs/environment.md) | Action space, block types, parameters |
| [Benchmarks](docs/benchmarks.md) | Throughput data, memory usage |
| [Architecture](docs/architecture.md) | Project structure, CUDA kernels |

## Throughput

![Throughput Surface](assets/surface.png)

Peak: **49.6M steps/sec** with batch=32K, world=16, resolution=32x32. See [benchmarks](docs/benchmarks.md) for full results.

## LLM

If you're an LLM working with this codebase, read [docs/LLM.md](docs/LLM.md) for navigation guidance.

## Citation

```bibtex
@software{minecu2026,
  author = {Elliot Arledge},
  title = {mine.cu: High-Performance Batched Voxel RL Environment},
  year = {2026},
  url = {https://github.com/Infatoshi/mine.cu}
}
```

## TODO

Contributions welcome. Some ideas:

- [ ] Procedural terrain generation (noise-based heightmaps)
- [ ] Voxel-based collision detection (replace flat ground plane)
- [ ] Block placement action
- [ ] Inventory system
- [ ] Multi-agent visibility (render other agents)
- [ ] More block types and textures
- [ ] Curriculum learning utilities
- [ ] Benchmark on other GPUs (A100, H100, 4090)

## License

MIT
