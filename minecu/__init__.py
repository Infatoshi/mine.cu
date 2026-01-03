"""
mine.cu - High-performance batched voxel RL environment with custom CUDA kernels

A minimal Minecraft-like environment designed for reinforcement learning research.
All environment logic runs on GPU via custom CUDA kernels for maximum throughput.

Example:
    import torch
    from minecu import MineEnv

    env = MineEnv(batch_size=4096, world_size=32, device="cuda")
    obs = env.reset()

    for _ in range(1000):
        actions = torch.randint(0, env.action_dim, (4096,), device="cuda")
        obs, rewards, dones = env.step(actions)
"""

import torch
from typing import Tuple, Optional

try:
    from minecu import _C
except ImportError:
    raise ImportError(
        "CUDA extension not found. Build with: pip install -e . "
        "or: python setup.py build_ext --inplace"
    )

__version__ = "0.1.0"
__all__ = ["MineEnv", "BlockType", "EnvConfig", "TrainConfig"]

from minecu.config import EnvConfig, TrainConfig


class BlockType:
    """Block type constants matching CUDA kernel definitions."""
    AIR = -1
    GRASS = 0
    DIRT = 1
    STONE = 2
    OAKLOG = 3    # Wood log
    LEAVES = 4
    SAND = 5
    WATER = 6
    GLASS = 7
    BRICK = 8
    COBBLESTONE = 9
    PLANKS = 10
    SNOW = 11
    BEDROCK = 12


class MineEnv:
    """
    Batched voxel environment with CUDA-accelerated rendering and physics.

    All tensors live on GPU. The environment supports batched operations
    for training thousands of agents in parallel.

    Args:
        batch_size: Number of parallel environment instances
        world_size: Size of cubic voxel world (world_size x world_size x world_size)
        render_width: Width of rendered observation in pixels
        render_height: Height of rendered observation in pixels
        device: PyTorch device (must be CUDA)
        max_steps: Maximum raymarching steps for rendering
        view_distance: Maximum view distance in blocks
        fov_degrees: Field of view in degrees
        dt: Physics timestep
        gravity: Gravity acceleration (negative = down)
        walk_speed: Agent walking speed in blocks/second
        jump_vel: Initial jump velocity
        ground_height: Y level of ground surface
        target_block: Block type that gives reward when broken (-1 for any)
        reward_value: Reward for breaking target block
        episode_length: Steps per episode (None for infinite)
    """

    # Action space: 0=noop, 1=forward, 2=back, 3=left, 4=right, 5=jump, 6=break, 7-10=look
    action_dim = 11

    @classmethod
    def from_config(cls, config: "EnvConfig") -> "MineEnv":
        """Create environment from config object."""
        return cls(
            batch_size=config.batch_size,
            world_size=config.world_size,
            render_width=config.render_width,
            render_height=config.render_height,
            device=config.device,
            max_steps=config.max_steps,
            view_distance=config.view_distance,
            fov_degrees=config.fov_degrees,
            dt=config.dt,
            gravity=config.gravity,
            walk_speed=config.walk_speed,
            jump_vel=config.jump_vel,
            ground_height=config.ground_height,
            target_block=config.target_block,
            reward_value=config.reward_value,
            episode_length=config.episode_length,
            seed=config.seed,
        )

    def __init__(
        self,
        batch_size: int = 4096,
        world_size: int = 32,
        render_width: int = 64,
        render_height: int = 64,
        device: str = "cuda",
        max_steps: int = 64,
        view_distance: float = 32.0,
        fov_degrees: float = 70.0,
        dt: float = 0.05,
        gravity: float = -20.0,
        walk_speed: float = 4.0,
        jump_vel: float = 8.0,
        ground_height: int = 8,
        target_block: int = BlockType.OAKLOG,
        reward_value: float = 1.0,
        episode_length: Optional[int] = None,
        seed: int = 42,
    ):
        assert "cuda" in device, "MineEnv requires CUDA device"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.world_size = world_size
        self.render_width = render_width
        self.render_height = render_height
        self.max_steps = max_steps
        self.view_distance = view_distance
        self.fov_degrees = fov_degrees
        self.dt = dt
        self.gravity = gravity
        self.walk_speed = walk_speed
        self.jump_vel = jump_vel
        self.ground_height = ground_height
        self.target_block = target_block
        self.reward_value = reward_value
        self.episode_length = episode_length
        self.seed = seed

        # Spawn position (center of world, above ground)
        self.spawn_x = world_size / 2.0
        self.spawn_y = float(ground_height + 2)
        self.spawn_z = world_size / 2.0

        # Allocate state tensors
        self._allocate_state()

        # Generate world and place initial tree
        self._generate_world()

    def _allocate_state(self):
        """Allocate all GPU tensors for environment state."""
        B, W = self.batch_size, self.world_size
        H, WW = self.render_height, self.render_width

        # Voxel world: [B, Y, X, Z] - Y is vertical (height)
        self.voxels = torch.zeros((B, W, W, W), dtype=torch.int8, device=self.device)

        # Agent state
        self.positions = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        self.velocities = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        self.yaws = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.pitches = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.on_ground = torch.ones((B,), dtype=torch.bool, device=self.device)

        # Render output: [B, H, W, 3] RGB float
        self.obs_buffer = torch.zeros((B, H, WW, 3), dtype=torch.float32, device=self.device)

        # Camera buffer for rendering: [B, 5] = (x, y, z, yaw, pitch)
        self.cameras = torch.zeros((B, 5), dtype=torch.float32, device=self.device)

        # Action buffers
        self.forward_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.strafe_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.delta_yaw_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.delta_pitch_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.jump_in = torch.zeros((B,), dtype=torch.bool, device=self.device)
        self.do_break = torch.zeros((B,), dtype=torch.bool, device=self.device)

        # Rewards and episode tracking
        self.rewards = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.step_count = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.do_reset = torch.zeros((B,), dtype=torch.bool, device=self.device)

    def _generate_world(self):
        """Generate flat world with ground and a tree."""
        _C.generate_world(self.voxels, self.ground_height, self.seed)

        # Place a single tree 2 blocks in front of spawn
        tree_x = int(self.spawn_x)
        tree_z = int(self.spawn_z) + 2
        _C.place_tree(self.voxels, tree_x, tree_z, self.ground_height)

    def reset(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reset environments.

        Args:
            mask: Boolean tensor [B] indicating which envs to reset.
                  If None, resets all environments.

        Returns:
            obs: Rendered observations [B, H, W, 3]
        """
        if mask is None:
            self.do_reset.fill_(True)
        else:
            self.do_reset.copy_(mask)

        _C.reset(
            self.positions, self.velocities, self.yaws, self.pitches, self.on_ground,
            self.do_reset, self.spawn_x, self.spawn_y, self.spawn_z
        )

        # Reset step counter for reset envs
        self.step_count.masked_fill_(self.do_reset, 0)

        # Regenerate world for reset envs (restore broken blocks)
        if mask is None or mask.any():
            _C.generate_world(self.voxels, self.ground_height, self.seed)
            tree_x = int(self.spawn_x)
            tree_z = int(self.spawn_z) + 2
            _C.place_tree(self.voxels, tree_x, tree_z, self.ground_height)

        return self._render()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a step in all environments.

        Args:
            actions: Integer actions [B] in range [0, action_dim)
                0: noop
                1: forward
                2: backward
                3: strafe left
                4: strafe right
                5: jump
                6: break block
                7: look left
                8: look right
                9: look up
                10: look down

        Returns:
            obs: Rendered observations [B, H, W, 3]
            rewards: Rewards received [B]
            dones: Episode termination flags [B]
        """
        # Decode actions into movement/look/break signals
        self._decode_actions(actions)

        # Physics step
        _C.physics(
            self.positions, self.velocities, self.yaws, self.pitches, self.on_ground,
            self.forward_in, self.strafe_in, self.delta_yaw_in, self.delta_pitch_in, self.jump_in,
            self.world_size, self.dt, self.gravity, self.walk_speed, self.jump_vel
        )

        # Block breaking with reward
        self.rewards.zero_()
        _C.raycast_break(
            self.voxels, self.positions, self.yaws, self.pitches,
            self.do_break, self.rewards, self.target_block, self.reward_value
        )

        # Increment step counter
        self.step_count += 1

        # Check for episode termination
        if self.episode_length is not None:
            dones = self.step_count >= self.episode_length
        else:
            dones = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)

        # Auto-reset terminated episodes
        if dones.any():
            self.reset(dones)

        return self._render(), self.rewards.clone(), dones

    def _decode_actions(self, actions: torch.Tensor):
        """Decode integer actions into continuous control signals."""
        # Reset all inputs
        self.forward_in.zero_()
        self.strafe_in.zero_()
        self.delta_yaw_in.zero_()
        self.delta_pitch_in.zero_()
        self.jump_in.zero_()
        self.do_break.zero_()

        # Movement
        self.forward_in.masked_fill_(actions == 1, 1.0)   # forward
        self.forward_in.masked_fill_(actions == 2, -1.0)  # backward
        self.strafe_in.masked_fill_(actions == 3, -1.0)   # left
        self.strafe_in.masked_fill_(actions == 4, 1.0)    # right

        # Jump
        self.jump_in.masked_fill_(actions == 5, True)

        # Break
        self.do_break.masked_fill_(actions == 6, True)

        # Look (0.1 radians per step)
        look_speed = 0.1
        self.delta_yaw_in.masked_fill_(actions == 7, look_speed)   # left
        self.delta_yaw_in.masked_fill_(actions == 8, -look_speed)  # right
        self.delta_pitch_in.masked_fill_(actions == 9, look_speed)  # up
        self.delta_pitch_in.masked_fill_(actions == 10, -look_speed)  # down

    def _render(self) -> torch.Tensor:
        """Render observations for all environments."""
        # Update camera buffer from agent state
        self.cameras[:, 0] = self.positions[:, 0]
        self.cameras[:, 1] = self.positions[:, 1]
        self.cameras[:, 2] = self.positions[:, 2]
        self.cameras[:, 3] = self.yaws
        self.cameras[:, 4] = self.pitches

        _C.render(
            self.voxels, self.cameras, self.obs_buffer,
            self.max_steps, self.view_distance, self.fov_degrees
        )

        return self.obs_buffer

    def get_obs(self) -> torch.Tensor:
        """Get current observations without stepping."""
        return self._render()

    def close(self):
        """Clean up resources."""
        pass  # PyTorch handles GPU memory

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Shape of observation tensor: (H, W, 3)."""
        return (self.render_height, self.render_width, 3)
