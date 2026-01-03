# Environment Reference

## Action Space

| Action | Description |
|--------|-------------|
| 0 | No-op |
| 1 | Move forward |
| 2 | Move backward |
| 3 | Strafe left |
| 4 | Strafe right |
| 5 | Jump |
| 6 | Break block |
| 7 | Look left |
| 8 | Look right |
| 9 | Look up |
| 10 | Look down |

## Block Types

| ID | Block |
|----|-------|
| -1 | Air |
| 0 | Grass |
| 1 | Dirt |
| 2 | Stone |
| 3 | Oak Log |
| 4 | Leaves |
| 5 | Sand |
| 6 | Water |
| 7 | Glass |
| 8 | Brick |
| 9 | Cobblestone |
| 10 | Planks |
| 11 | Snow |
| 12 | Bedrock |

## Environment Parameters

```python
env = MineEnv(
    batch_size=4096,       # Number of parallel environments
    world_size=32,         # Cubic world size (32x32x32 blocks)
    render_width=64,       # Observation width
    render_height=64,      # Observation height
    episode_length=60,     # Steps per episode (None for infinite)
    target_block=3,        # Block type for reward (3=wood log)
    reward_value=1.0,      # Reward for breaking target block
    device="cuda",
)
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 4096 | Number of parallel environment instances |
| `world_size` | int | 32 | Cubic voxel world dimensions (world_size^3 blocks) |
| `render_width` | int | 64 | Observation image width in pixels |
| `render_height` | int | 64 | Observation image height in pixels |
| `episode_length` | int/None | 60 | Max steps per episode, None for infinite |
| `target_block` | int | 3 | Block ID that gives reward when broken |
| `reward_value` | float | 1.0 | Reward magnitude for breaking target block |
| `device` | str | "cuda" | PyTorch device |

## Observation Space

- Shape: `[batch_size, render_height, render_width, 3]`
- Type: `torch.float32`
- Range: `[0, 1]` (normalized RGB)

## Using Configs

```python
from minecu import MineEnv, EnvConfig
from minecu.config import FAST_ENV

# Predefined fast config
env = MineEnv.from_config(FAST_ENV)

# Custom config
config = EnvConfig(
    batch_size=32768,
    world_size=16,
    render_width=32,
    render_height=32,
    max_steps=32,
)
env = MineEnv.from_config(config)
```
