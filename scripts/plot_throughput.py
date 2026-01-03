#!/usr/bin/env python3
"""
Generate throughput visualization for README.

Creates a professional bar chart showing throughput across different
hyperparameter configurations.
"""

import matplotlib.pyplot as plt
import numpy as np

# Throughput data from benchmarks (M steps/sec)
# Optimized kernel: AABB early termination + loop unrolling + __ldg + block_size=128
# Format: (world_size, resolution, batch_size) -> throughput
data = {
    # World 16x16x16
    (16, 32, 4096): 24.47,
    (16, 32, 8192): 34.11,
    (16, 32, 16384): 42.39,
    (16, 32, 32768): 49.63,
    (16, 64, 4096): 11.10,
    (16, 64, 8192): 12.57,
    (16, 64, 16384): 13.40,
    (16, 64, 32768): 13.74,
    (16, 128, 4096): 3.29,
    (16, 128, 8192): 3.42,
    (16, 128, 16384): 3.47,
    (16, 128, 32768): 3.49,
    # World 32x32x32
    (32, 32, 4096): 8.10,
    (32, 32, 8192): 9.20,
    (32, 32, 16384): 9.91,
    (32, 32, 32768): 10.15,
    (32, 64, 4096): 2.78,
    (32, 64, 8192): 2.83,
    (32, 64, 16384): 2.90,
    (32, 64, 32768): 2.91,
    (32, 128, 4096): 0.82,
    (32, 128, 8192): 0.84,
    (32, 128, 16384): 0.84,
    (32, 128, 32768): 0.83,
    # World 48x48x48
    (48, 32, 4096): 5.51,
    (48, 32, 8192): 5.94,
    (48, 32, 16384): 6.30,
    (48, 64, 4096): 1.83,
    (48, 64, 8192): 1.87,
    (48, 64, 16384): 1.88,
    (48, 128, 4096): 0.53,
    (48, 128, 8192): 0.53,
    (48, 128, 16384): 0.53,
}


def plot_throughput_bars():
    """Create grouped bar chart of throughput by configuration."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by (world_size, resolution), show batch_size progression
    configs = [
        ("16³, 32px", 16, 32),
        ("16³, 64px", 16, 64),
        ("16³, 128px", 16, 128),
        ("32³, 32px", 32, 32),
        ("32³, 64px", 32, 64),
        ("32³, 128px", 32, 128),
        ("48³, 32px", 48, 32),
        ("48³, 64px", 48, 64),
        ("48³, 128px", 48, 128),
    ]

    batch_sizes = [4096, 8192, 16384, 32768]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    x = np.arange(len(configs))
    width = 0.2

    for i, bs in enumerate(batch_sizes):
        values = []
        for label, ws, res in configs:
            val = data.get((ws, res, bs), 0)
            values.append(val)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=f'batch={bs:,}', color=colors[i], alpha=0.85)

        # Highlight peak
        for j, v in enumerate(values):
            if v == 49.63:
                bars[j].set_edgecolor('black')
                bars[j].set_linewidth(2)
                ax.annotate('49.6M', (x[j] + offset, v + 1.5), ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Throughput (M steps/sec)', fontsize=11)
    ax.set_xlabel('Configuration (world size, resolution)', fontsize=11)
    ax.set_title('Environment Throughput by Hyperparameters (RTX 3090)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 58)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Add horizontal line at peak
    ax.axhline(y=49.63, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig('assets/throughput.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved assets/throughput.png')


def plot_throughput_surface():
    """Create 3D surface plot showing throughput."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 5))

    resolutions = [32, 64, 128]
    batch_sizes = np.array([4096, 8192, 16384, 32768])
    world_sizes = np.array([16, 32, 48])

    # Create meshgrid for surface
    BS, WS = np.meshgrid(np.arange(len(batch_sizes)), np.arange(len(world_sizes)))

    for idx, res in enumerate(resolutions):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        # Build Z matrix (throughput values)
        Z = np.zeros((len(world_sizes), len(batch_sizes)))
        for i, ws in enumerate(world_sizes):
            for j, bs in enumerate(batch_sizes):
                Z[i, j] = data.get((ws, res, bs), 0)

        # Create surface
        surf = ax.plot_surface(BS, WS, Z, cmap='plasma', edgecolor='black',
                               linewidth=0.3, alpha=0.9, antialiased=True)

        # Mark peak for 32px resolution
        if res == 32:
            peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
            ax.scatter([peak_idx[1]], [peak_idx[0]], [Z[peak_idx]],
                      color='white', s=100, edgecolors='black', linewidths=2, zorder=5)
            ax.text(peak_idx[1], peak_idx[0], Z[peak_idx] + 3,
                   f'{Z[peak_idx]:.1f}M', fontsize=9, fontweight='bold', ha='center')

        # Labels
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels([f'{bs//1000}K' for bs in batch_sizes], fontsize=8)
        ax.set_yticks(range(len(world_sizes)))
        ax.set_yticklabels([f'{ws}³' for ws in world_sizes], fontsize=8)
        ax.set_xlabel('Batch Size', fontsize=9, labelpad=5)
        ax.set_ylabel('World Size', fontsize=9, labelpad=5)
        ax.set_zlabel('M steps/sec', fontsize=9, labelpad=5)
        ax.set_title(f'Resolution: {res}x{res}', fontsize=11, fontweight='bold', pad=10)

        # Set consistent z-axis limits
        ax.set_zlim(0, 55)

        # Better viewing angle
        ax.view_init(elev=25, azim=45)

    fig.suptitle('Throughput Surface by Hyperparameters (RTX 3090)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('assets/throughput_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved assets/throughput_heatmap.png')


def plot_scaling():
    """Show how throughput scales with batch size for key configs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    batch_sizes = np.array([4096, 8192, 16384, 32768])

    configs = [
        ("16³ world, 32px", 16, 32, '#e74c3c'),
        ("32³ world, 32px", 32, 32, '#3498db'),
        ("32³ world, 64px", 32, 64, '#2ecc71'),
    ]

    for label, ws, res, color in configs:
        throughputs = [data.get((ws, res, bs), 0) for bs in batch_sizes]
        ax.plot(batch_sizes, throughputs, 'o-', label=label, color=color, linewidth=2, markersize=8)

    # Mark peak
    ax.scatter([32768], [49.63], s=150, c='#e74c3c', zorder=5, edgecolors='black', linewidths=2)
    ax.annotate('Peak: 49.6M', (32768, 49.63), xytext=(25000, 44), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Throughput (M steps/sec)', fontsize=11)
    ax.set_title('Throughput Scaling with Batch Size', fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([f'{bs//1000}K' for bs in batch_sizes])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 58)

    plt.tight_layout()
    plt.savefig('assets/scaling.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved assets/scaling.png')


if __name__ == '__main__':
    import os
    os.makedirs('assets', exist_ok=True)

    plot_throughput_bars()
    plot_throughput_surface()
    plot_scaling()
    print('Done!')
