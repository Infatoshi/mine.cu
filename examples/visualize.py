#!/usr/bin/env python3
"""
Visualize trained or random agents in the environment.

Generates an MP4 video showing multiple environment instances side by side.

Usage:
    python examples/visualize.py --output demo.mp4 --steps 200 --fps 20
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Use non-interactive backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from minecu import MineEnv, BlockType


def create_demo_video(
    output_path: str = "demo.mp4",
    num_envs: int = 8,
    steps: int = 200,
    fps: int = 20,
    render_size: int = 64,
    random_policy: bool = True,
):
    """Generate visualization video."""
    device = "cuda"

    env = MineEnv(
        batch_size=num_envs,
        world_size=32,
        render_width=render_size,
        render_height=render_size,
        episode_length=60,
        target_block=BlockType.OAKLOG,
        device=device,
    )

    # Collect frames
    print(f"Collecting {steps} frames...")
    frames = []
    obs = env.reset()

    for i in range(steps):
        # Random or learned policy
        if random_policy:
            actions = torch.randint(0, env.action_dim, (num_envs,), device=device)
        else:
            # Placeholder for trained policy
            actions = torch.randint(0, env.action_dim, (num_envs,), device=device)

        obs, rewards, dones = env.step(actions)
        frame = obs.cpu().numpy()  # [B, H, W, 3]
        frames.append(frame)

        if (i + 1) % 50 == 0:
            print(f"  Frame {i + 1}/{steps}")

    # Arrange in grid
    print("Rendering video...")
    n_cols = int(np.ceil(np.sqrt(num_envs)))
    n_rows = int(np.ceil(num_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if num_envs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Initialize images
    ims = []
    for i in range(num_envs):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        ax.axis('off')
        im = ax.imshow(frames[0][i])
        ims.append(im)

    # Hide unused subplots
    for i in range(num_envs, n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        axes[r, c].axis('off')

    plt.tight_layout()

    def update(frame_idx):
        for i, im in enumerate(ims):
            im.set_array(frames[frame_idx][i])
        return ims

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)

    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize mine.cu environment")
    parser.add_argument("--output", "-o", default="demo.mp4", help="Output video path")
    parser.add_argument("--envs", "-n", type=int, default=8, help="Number of environments")
    parser.add_argument("--steps", "-s", type=int, default=200, help="Number of steps")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--size", type=int, default=64, help="Render resolution")
    args = parser.parse_args()

    create_demo_video(
        output_path=args.output,
        num_envs=args.envs,
        steps=args.steps,
        fps=args.fps,
        render_size=args.size,
    )


if __name__ == "__main__":
    main()
