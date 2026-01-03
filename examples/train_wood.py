#!/usr/bin/env python3
"""
Train an agent to break wood blocks using REINFORCE.

This example demonstrates:
- Setting up the MineEnv with custom parameters
- Training a simple CNN policy with PyTorch
- Logging throughput and reward metrics

Usage:
    python examples/train_wood.py

Expected output:
    After ~5M steps, the agent learns to look at and break the tree.
    Typical throughput: 2-4M steps/second on RTX 3090/4090.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from minecu import MineEnv, BlockType


class SimpleCNN(nn.Module):
    """Minimal CNN policy for visual RL."""

    def __init__(self, obs_shape, action_dim):
        super().__init__()
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            self.conv_out_size = conv_out.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        # x: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


def train(
    batch_size: int = 4096,
    total_steps: int = 10_000_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    episode_length: int = 60,
    render_size: int = 32,
    log_interval: int = 1000,
):
    """Train agent to break wood."""
    device = "cuda"

    # Create environment
    env = MineEnv(
        batch_size=batch_size,
        world_size=32,
        render_width=render_size,
        render_height=render_size,
        episode_length=episode_length,
        target_block=BlockType.OAKLOG,
        reward_value=1.0,
        device=device,
    )

    # Create policy
    policy = SimpleCNN(env.observation_shape, env.action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Training loop
    obs = env.reset()
    total_reward = 0.0
    episode_rewards = []
    step = 0
    start_time = time.time()

    print(f"Training on {batch_size} parallel envs, {total_steps:,} total steps")
    print(f"Observation shape: {env.observation_shape}, Action dim: {env.action_dim}")
    print("-" * 60)

    while step < total_steps:
        # Get action from policy
        with torch.no_grad():
            logits = policy(obs)
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)

        # Step environment
        obs, rewards, dones = env.step(actions)
        total_reward += rewards.sum().item()

        # REINFORCE update (simplified - no baseline)
        logits = policy(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = -(action_log_probs * rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += batch_size

        # Logging
        if step % (log_interval * batch_size) == 0:
            elapsed = time.time() - start_time
            sps = step / elapsed
            avg_reward = total_reward / step * batch_size
            print(f"Step {step:>10,} | SPS: {sps:>8,.0f} | Reward/batch: {avg_reward:.2f}")

    # Final stats
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Training complete: {total_steps:,} steps in {elapsed:.1f}s")
    print(f"Average throughput: {total_steps / elapsed:,.0f} steps/second")
    print(f"Total reward: {total_reward:,.0f}")

    return policy


if __name__ == "__main__":
    policy = train()
