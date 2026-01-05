"""
mine.cu - High-performance batched voxel RL environment with custom CUDA kernels

A minimal Minecraft-like environment designed for reinforcement learning research.
All environment logic runs on GPU via custom CUDA kernels for maximum throughput.

Example:
    import torch
    from minecu import MineEnv

    # Cubic world (32x32x32)
    env = MineEnv(batch_size=4096, world_size=32, device="cuda")

    # Non-cubic world (64x32x64 - wider X and Z)
    env = MineEnv(batch_size=4096, world_x=64, world_y=32, world_z=64, device="cuda")

    obs = env.reset()

    for _ in range(1000):
        # Multi-hot buttons: [forward, back, left, right, jump, break, place, sprint]
        buttons = torch.zeros(4096, 8, dtype=torch.int8, device="cuda")
        buttons[:, 0] = 1  # forward

        # Continuous look: [delta_yaw, delta_pitch] in radians
        look = torch.zeros(4096, 2, dtype=torch.float32, device="cuda")

        obs, rewards, dones = env.step(buttons, look)
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
        world_size: Size of cubic voxel world (used if world_x/y/z not specified)
        world_x: X dimension of voxel world (overrides world_size)
        world_y: Y dimension of voxel world - height (overrides world_size)
        world_z: Z dimension of voxel world (overrides world_size)
        render_width: Width of rendered observation in pixels
        render_height: Height of rendered observation in pixels
        device: PyTorch device (must be CUDA)
        max_steps: Maximum raymarching steps for rendering
        view_distance: Maximum view distance in blocks
        fov_degrees: Field of view in degrees
        dt: Physics timestep
        gravity: Gravity acceleration (negative = down)
        walk_speed: Agent walking speed in blocks/second
        sprint_mult: Sprint speed multiplier (applied to forward movement only)
        jump_vel: Initial jump velocity
        ground_height: Y level of ground surface
        target_block: Block type that gives reward when broken (-1 for any)
        reward_value: Reward for breaking target block
        episode_length: Steps per episode (None for infinite)

    Action Format:
        buttons: int8[B, 8] multi-hot encoding
            0: forward (W)
            1: backward (S)
            2: strafe_left (A)
            3: strafe_right (D)
            4: jump (Space)
            5: break block (LMB)
            6: place block (RMB)
            7: sprint (Shift)

        look: float32[B, 2] continuous
            0: delta_yaw (radians, + = look left)
            1: delta_pitch (radians, + = look up)
    """

    # Action dimensions
    button_dim = 8   # Multi-hot buttons
    look_dim = 2     # Continuous look

    @classmethod
    def from_config(cls, config: "EnvConfig") -> "MineEnv":
        """Create environment from config object."""
        return cls(
            batch_size=config.batch_size,
            world_size=getattr(config, 'world_size', 32),
            world_x=getattr(config, 'world_x', None),
            world_y=getattr(config, 'world_y', None),
            world_z=getattr(config, 'world_z', None),
            render_width=config.render_width,
            render_height=config.render_height,
            device=config.device,
            max_steps=config.max_steps,
            view_distance=config.view_distance,
            fov_degrees=config.fov_degrees,
            dt=config.dt,
            gravity=config.gravity,
            walk_speed=config.walk_speed,
            sprint_mult=getattr(config, 'sprint_mult', 1.5),
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
        world_x: Optional[int] = None,
        world_y: Optional[int] = None,
        world_z: Optional[int] = None,
        render_width: int = 64,
        render_height: int = 64,
        device: str = "cuda",
        max_steps: int = 64,
        view_distance: float = 32.0,
        fov_degrees: float = 70.0,
        dt: float = 0.05,
        gravity: float = -20.0,
        walk_speed: float = 4.0,
        sprint_mult: float = 1.5,
        jump_vel: float = 8.0,
        ground_height: int = 8,
        target_block: int = BlockType.OAKLOG,
        reward_value: float = 1.0,
        episode_length: Optional[int] = None,
        seed: int = 42,
        obs_dtype: str = "float32",
    ):
        assert "cuda" in device, "MineEnv requires CUDA device"
        assert obs_dtype in ("float32", "uint8", "uint8_minimal", "uint8_fp16", "uint8_prebasis", "uint8_prebasis_full", "uint8_smem"), \
            "obs_dtype must be 'float32', 'uint8', 'uint8_minimal', 'uint8_fp16', 'uint8_prebasis', 'uint8_prebasis_full', or 'uint8_smem'"
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Support non-cubic worlds: world_x/y/z override world_size
        self.world_x = world_x if world_x is not None else world_size
        self.world_y = world_y if world_y is not None else world_size
        self.world_z = world_z if world_z is not None else world_size
        self.world_size = world_size  # Keep for backward compat

        self.render_width = render_width
        self.render_height = render_height
        self.max_steps = max_steps
        self.view_distance = view_distance
        self.fov_degrees = fov_degrees
        self.dt = dt
        self.gravity = gravity
        self.walk_speed = walk_speed
        self.sprint_mult = sprint_mult
        self.jump_vel = jump_vel
        self.ground_height = ground_height
        self.target_block = target_block
        self.reward_value = reward_value
        self.episode_length = episode_length
        self.seed = seed
        self.obs_dtype = obs_dtype

        # Spawn position (center of world, above ground)
        self.spawn_x = self.world_x / 2.0
        self.spawn_y = float(ground_height + 2)
        self.spawn_z = self.world_z / 2.0

        # Tree position (2 blocks in front of spawn)
        self.tree_x = int(self.spawn_x)
        self.tree_z = int(self.spawn_z) + 2

        # House position (to the right of spawn)
        self.house_x = int(self.spawn_x) + 6
        self.house_z = int(self.spawn_z) - 2

        # Allocate state tensors
        self._allocate_state()

        # Generate world and place initial tree
        self._generate_world()

    def _allocate_state(self):
        """Allocate all GPU tensors for environment state."""
        B = self.batch_size
        WX, WY, WZ = self.world_x, self.world_y, self.world_z
        H, WW = self.render_height, self.render_width

        # Voxel world: [B, Y, X, Z] - Y is vertical (height)
        self.voxels = torch.zeros((B, WY, WX, WZ), dtype=torch.int8, device=self.device)

        # Agent state
        self.positions = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        self.velocities = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        self.yaws = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.pitches = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.on_ground = torch.ones((B,), dtype=torch.bool, device=self.device)

        # Render output: [B, H, W, 3] RGB (float32 or uint8)
        obs_torch_dtype = torch.uint8 if self.obs_dtype in ("uint8", "uint8_minimal", "uint8_fp16", "uint8_prebasis", "uint8_prebasis_full", "uint8_smem") else torch.float32
        self.obs_buffer = torch.zeros((B, H, WW, 3), dtype=obs_torch_dtype, device=self.device)

        # Camera buffer for rendering: [B, 5] = (x, y, z, yaw, pitch)
        self.cameras = torch.zeros((B, 5), dtype=torch.float32, device=self.device)

        # Precomputed camera basis for prebasis/smem mode: [B, 14]
        if self.obs_dtype in ("uint8_prebasis", "uint8_prebasis_full", "uint8_smem"):
            self.basis = torch.zeros((B, 14), dtype=torch.float32, device=self.device)
        else:
            self.basis = None

        # Action decode buffers (internal, written by decode_actions_kernel)
        self.forward_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.strafe_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.delta_yaw_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.delta_pitch_in = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.jump_in = torch.zeros((B,), dtype=torch.bool, device=self.device)
        self.do_break = torch.zeros((B,), dtype=torch.bool, device=self.device)
        self.do_place = torch.zeros((B,), dtype=torch.bool, device=self.device)
        self.speed_mult = torch.ones((B,), dtype=torch.float32, device=self.device)

        # Rewards and episode tracking
        self.rewards = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.step_count = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.do_reset = torch.zeros((B,), dtype=torch.bool, device=self.device)

    def _generate_world(self):
        """Generate flat world with ground, a tree, and a house."""
        _C.generate_world(self.voxels, self.ground_height, self.seed)

        # Place a single tree 2 blocks in front of spawn
        tree_x = int(self.spawn_x)
        tree_z = int(self.spawn_z) + 2
        _C.place_tree(self.voxels, tree_x, tree_z, self.ground_height)

        # Place a house to the right of spawn
        _C.place_house(self.voxels, self.house_x, self.house_z, self.ground_height)

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
            _C.place_house(self.voxels, self.house_x, self.house_z, self.ground_height)

        return self._render()

    def step(
        self,
        buttons: torch.Tensor,
        look: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a step in all environments using the new multi-hot action format.

        Args:
            buttons: int8[B, 8] multi-hot button states
                0: forward (W)
                1: backward (S)
                2: strafe_left (A)
                3: strafe_right (D)
                4: jump (Space)
                5: break block (LMB)
                6: place block (RMB)
                7: sprint (Shift)

            look: float32[B, 2] continuous look deltas in radians
                0: delta_yaw (+ = look left)
                1: delta_pitch (+ = look up)

        Returns:
            obs: Rendered observations [B, H, W, 3]
            rewards: Rewards received [B]
            dones: Episode termination flags [B]
        """
        # uint8_prebasis_full: episode check/update in C++, world regen in Python
        # This eliminates PyTorch kernels for step counter while avoiding
        # the 0.5ms cost of launching world regen every step
        if self.obs_dtype == "uint8_prebasis_full":
            # C++ handles: episode check, step kernels, counter update
            _C.step_uint8_prebasis_full(
                self.voxels,
                self.positions, self.velocities, self.yaws, self.pitches, self.on_ground,
                buttons, look,
                self.do_reset,
                self.step_count,
                self.forward_in, self.strafe_in, self.delta_yaw_in, self.delta_pitch_in,
                self.jump_in, self.do_break, self.do_place, self.speed_mult,
                self.cameras,
                self.basis,
                self.obs_buffer, self.rewards,
                self.render_width, self.render_height,
                self.max_steps, self.view_distance, self.fov_degrees,
                self.dt, self.gravity, self.walk_speed, self.sprint_mult, self.jump_vel,
                self.target_block, self.reward_value,
                self.spawn_x, self.spawn_y, self.spawn_z,
                self.episode_length if self.episode_length is not None else 0,
            )
            # World regen only when needed (saves 0.5ms when no resets)
            if self.do_reset.any():
                _C.generate_world(self.voxels, self.ground_height, self.seed)
                _C.place_tree(self.voxels, self.tree_x, self.tree_z, self.ground_height)
                _C.place_house(self.voxels, self.house_x, self.house_z, self.ground_height)
            return self.obs_buffer, self.rewards, self.do_reset.clone()

        # Check for episode termination (before step, for auto-reset)
        if self.episode_length is not None:
            self.do_reset = self.step_count >= self.episode_length
        else:
            self.do_reset.zero_()

        # Single unified C++ step function
        # - step: float32 output with fog/shading
        # - step_uint8: uint8 output with fog/shading
        # - step_uint8_minimal: uint8 output with flat colors (no fog/shading)
        # - step_uint8_fp16: uint8 output with fp16 DDA (flat colors)
        # - step_uint8_prebasis: uint8 output with precomputed camera basis (no per-pixel trig)
        # - step_uint8_smem: uint8 output with entire voxel grid in shared memory
        if self.obs_dtype == "uint8_smem":
            _C.step_uint8_smem(
                self.voxels,
                self.positions, self.velocities, self.yaws, self.pitches, self.on_ground,
                buttons, look,
                self.do_reset,
                self.forward_in, self.strafe_in, self.delta_yaw_in, self.delta_pitch_in,
                self.jump_in, self.do_break, self.do_place, self.speed_mult,
                self.cameras,
                self.basis,
                self.obs_buffer, self.rewards,
                self.render_width, self.render_height,
                self.max_steps, self.view_distance, self.fov_degrees,
                self.dt, self.gravity, self.walk_speed, self.sprint_mult, self.jump_vel,
                self.target_block, self.reward_value,
                self.spawn_x, self.spawn_y, self.spawn_z,
            )
        elif self.obs_dtype == "uint8_prebasis":
            _C.step_uint8_prebasis(
                self.voxels,
                self.positions, self.velocities, self.yaws, self.pitches, self.on_ground,
                buttons, look,
                self.do_reset,
                self.forward_in, self.strafe_in, self.delta_yaw_in, self.delta_pitch_in,
                self.jump_in, self.do_break, self.do_place, self.speed_mult,
                self.cameras,
                self.basis,
                self.obs_buffer, self.rewards,
                self.render_width, self.render_height,
                self.max_steps, self.view_distance, self.fov_degrees,
                self.dt, self.gravity, self.walk_speed, self.sprint_mult, self.jump_vel,
                self.target_block, self.reward_value,
                self.spawn_x, self.spawn_y, self.spawn_z,
            )
        else:
            if self.obs_dtype == "uint8_fp16":
                step_fn = _C.step_uint8_fp16
            elif self.obs_dtype == "uint8_minimal":
                step_fn = _C.step_uint8_minimal
            elif self.obs_dtype == "uint8":
                step_fn = _C.step_uint8
            else:
                step_fn = _C.step
            step_fn(
                self.voxels,
                self.positions, self.velocities, self.yaws, self.pitches, self.on_ground,
                buttons, look,
                self.do_reset,
                self.forward_in, self.strafe_in, self.delta_yaw_in, self.delta_pitch_in,
                self.jump_in, self.do_break, self.do_place, self.speed_mult,
                self.cameras,
                self.obs_buffer, self.rewards,
                self.render_width, self.render_height,
                self.max_steps, self.view_distance, self.fov_degrees,
                self.dt, self.gravity, self.walk_speed, self.sprint_mult, self.jump_vel,
                self.target_block, self.reward_value,
                self.spawn_x, self.spawn_y, self.spawn_z,
            )

        # Increment step counter (reset handled in kernel for do_reset agents)
        self.step_count += 1
        self.step_count.masked_fill_(self.do_reset, 1)  # Reset counter starts at 1 after reset step

        # Regenerate world for reset envs
        if self.do_reset.any():
            _C.generate_world(self.voxels, self.ground_height, self.seed)
            _C.place_tree(self.voxels, self.tree_x, self.tree_z, self.ground_height)
            _C.place_house(self.voxels, self.house_x, self.house_z, self.ground_height)

        return self.obs_buffer, self.rewards, self.do_reset.clone()

    def _render(self) -> torch.Tensor:
        """Render observations for all environments."""
        # Update camera buffer from agent state
        self.cameras[:, 0] = self.positions[:, 0]
        self.cameras[:, 1] = self.positions[:, 1]
        self.cameras[:, 2] = self.positions[:, 2]
        self.cameras[:, 3] = self.yaws
        self.cameras[:, 4] = self.pitches

        if self.obs_dtype in ("uint8", "uint8_minimal", "uint8_fp16", "uint8_prebasis", "uint8_prebasis_full", "uint8_smem"):
            _C.render_uint8(
                self.voxels, self.cameras, self.obs_buffer,
                self.max_steps, self.view_distance, self.fov_degrees
            )
        else:
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
