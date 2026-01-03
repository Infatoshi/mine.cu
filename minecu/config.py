"""
mine.cu configuration

All tunable hyperparameters in one place. Modify these before launching training.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    """Environment configuration."""

    # Scale
    batch_size: int = 4096
    world_size: int = 32          # Cubic world: world_size^3 voxels

    # Rendering
    render_width: int = 64
    render_height: int = 64
    max_steps: int = 64           # Raymarching iterations
    view_distance: float = 32.0   # Max ray distance
    fov_degrees: float = 70.0

    # Physics
    dt: float = 0.05              # Physics timestep (20 Hz)
    gravity: float = -20.0
    walk_speed: float = 4.0       # Blocks per second
    jump_vel: float = 8.0

    # World generation
    ground_height: int = 8
    seed: int = 42

    # Rewards
    target_block: int = 3         # OAKLOG
    reward_value: float = 1.0

    # Episode
    episode_length: Optional[int] = 60  # None for infinite

    # Device
    device: str = "cuda"

    def memory_bytes(self) -> int:
        """Estimate GPU memory usage in bytes."""
        B = self.batch_size
        W = self.world_size
        H, WW = self.render_height, self.render_width

        voxels = B * W * W * W * 1          # int8
        positions = B * 3 * 4               # float32
        velocities = B * 3 * 4
        yaws = B * 4
        pitches = B * 4
        on_ground = B * 1
        obs_buffer = B * H * WW * 3 * 4     # float32 RGB
        cameras = B * 5 * 4
        action_buffers = B * 4 * 5 + B * 2  # floats + bools

        return voxels + positions + velocities + yaws + pitches + on_ground + obs_buffer + cameras + action_buffers

    def memory_mb(self) -> float:
        """Estimate GPU memory usage in MB."""
        return self.memory_bytes() / (1024 * 1024)


@dataclass
class TrainConfig:
    """Training configuration."""

    # Optimization
    lr: float = 3e-4
    gamma: float = 0.99

    # Training
    total_steps: int = 10_000_000
    log_interval: int = 100_000

    # Policy network
    hidden_dim: int = 256


# Default configurations
DEFAULT_ENV = EnvConfig()

FAST_ENV = EnvConfig(
    batch_size=32768,
    render_width=32,
    render_height=32,
    max_steps=32,
)

QUALITY_ENV = EnvConfig(
    batch_size=4096,
    render_width=128,
    render_height=128,
    max_steps=128,
    view_distance=64.0,
)
