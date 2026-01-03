# Architecture

## Project Structure

```
mine.cu/
    src/
        kernels.cu       # CUDA kernels (render, physics, raycast)
        bindings.cpp     # PyTorch/pybind11 bindings
    minecu/
        __init__.py      # Python wrapper (MineEnv class)
        config.py        # Hyperparameter configuration
    examples/
        train_wood.py    # Training example
        visualize.py     # Visualization script
    scripts/
        setup.sh         # One-command setup
        plot_throughput.py  # Generate benchmark graphs
    docs/
        environment.md   # Environment API reference
        benchmarks.md    # Performance data
        architecture.md  # This file
        LLM.md           # LLM navigation guide
```

## CUDA Kernels

All kernels are in `src/kernels.cu`.

### render_kernel (line ~85)

The main rendering kernel using DDA (Digital Differential Analyzer) raymarching.

**Optimizations applied:**
- AABB early termination: Skip rays that miss the world bounding box
- Loop unrolling: `#pragma unroll 4` on DDA steps
- `__ldg()` intrinsics: Read-only texture cache path for voxel/camera data
- Block size tuning: 128 threads per block

**Key functions:**
- `ray_intersects_aabb()` (line ~49): Fast ray-box intersection test
- DDA_STEP macro: Unrolled raymarching step

### physics_kernel (line ~295)

Handles player movement, gravity, and collision.

- Updates position based on velocity and input
- Applies gravity
- Ground collision at y = world_size/4
- World boundary clamping

### raycast_break_kernel (line ~370)

Block breaking with reward calculation.

- Raycasts from player eye position
- Breaks first solid block hit within reach (4 blocks)
- Returns reward if target block type is broken

### reset_kernel (line ~440)

Episode reset - respawns player at spawn position.

### generate_world_kernel / place_tree_kernel (line ~470+)

World generation:
- Flat terrain with grass/dirt layers
- Bedrock at y=0
- Tree placement (oak log + leaves)

## Python Bindings

`src/bindings.cpp` exposes kernels to Python via pybind11:

```cpp
PYBIND11_MODULE(_minecu, m) {
    m.def("render", &launch_render);
    m.def("physics", &launch_physics);
    m.def("raycast_break", &launch_raycast_break);
    m.def("reset", &launch_reset);
    m.def("generate_world", &launch_generate_world);
    m.def("place_tree", &launch_place_tree);
}
```

## MineEnv Class

`minecu/__init__.py` wraps the CUDA kernels in a gym-like interface:

```python
class MineEnv:
    def __init__(self, batch_size, world_size, ...):
        # Allocate GPU tensors
        self.voxels = torch.zeros(..., dtype=torch.int8, device=device)
        self.cameras = torch.zeros(..., dtype=torch.float32, device=device)
        # ...

    def reset(self) -> torch.Tensor:
        # Reset environments, return observations

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Execute action, return (obs, rewards, dones)
```

## Data Layout

### Voxels
- Shape: `[batch_size, world_size, world_size, world_size]`
- Type: `int8` (block IDs, -1 = air)
- Layout: `[B, Y, X, Z]` (Y is vertical)

### Cameras
- Shape: `[batch_size, 5]`
- Type: `float32`
- Layout: `[x, y, z, yaw, pitch]`

### Observations
- Shape: `[batch_size, height, width, 3]`
- Type: `float32`
- Layout: RGB normalized to [0, 1]

## Build System

`setup.py` uses PyTorch's CUDAExtension:

```python
ext_modules = [
    CUDAExtension(
        name='minecu._minecu',
        sources=['src/bindings.cpp', 'src/kernels.cu'],
        extra_compile_args={
            'nvcc': ['-O3', '--use_fast_math']
        }
    )
]
```

Build with: `CUDA_HOME=/usr/local/cuda-12.8 pip install -e .`
