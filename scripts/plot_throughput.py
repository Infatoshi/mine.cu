#!/usr/bin/env python3
"""
Generate throughput visualization for README.

Creates a professional bar chart showing throughput across different
hyperparameter configurations.
"""

import matplotlib.pyplot as plt
import numpy as np

# Throughput data from benchmarks (M steps/sec)
# Format: (world_size, resolution, batch_size) -> throughput
data = {
    # World 16x16x16
    (16, 32, 4096): 13.60,
    (16, 32, 8192): 17.64,
    (16, 32, 16384): 19.62,
    (16, 32, 32768): 20.54,
    (16, 64, 4096): 5.08,
    (16, 64, 8192): 5.40,
    (16, 64, 16384): 5.53,
    (16, 64, 32768): 5.58,
    (16, 128, 4096): 1.51,
    (16, 128, 8192): 1.53,
    (16, 128, 16384): 1.53,
    (16, 128, 32768): 1.54,
    # World 32x32x32
    (32, 32, 4096): 5.81,
    (32, 32, 8192): 6.26,
    (32, 32, 16384): 6.54,
    (32, 32, 32768): 6.66,
    (32, 64, 4096): 1.76,
    (32, 64, 8192): 1.79,
    (32, 64, 16384): 1.81,
    (32, 64, 32768): 1.81,
    (32, 128, 4096): 0.49,
    (32, 128, 8192): 0.50,
    (32, 128, 16384): 0.50,
    (32, 128, 32768): 0.49,
    # World 48x48x48
    (48, 32, 4096): 4.40,
    (48, 32, 8192): 4.68,
    (48, 32, 16384): 4.81,
    (48, 64, 4096): 1.35,
    (48, 64, 8192): 1.37,
    (48, 64, 16384): 1.38,
    (48, 128, 4096): 0.37,
    (48, 128, 8192): 0.37,
    (48, 128, 16384): 0.37,
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
            if v == 20.54:
                bars[j].set_edgecolor('black')
                bars[j].set_linewidth(2)
                ax.annotate('20.5M', (x[j] + offset, v + 0.5), ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Throughput (M steps/sec)', fontsize=11)
    ax.set_xlabel('Configuration (world size, resolution)', fontsize=11)
    ax.set_title('Environment Throughput by Hyperparameters (RTX 3090)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 24)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Add horizontal line at peak
    ax.axhline(y=20.54, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig('assets/throughput.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved assets/throughput.png')


def plot_throughput_heatmap():
    """Create heatmap showing throughput surface."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    resolutions = [32, 64, 128]
    batch_sizes = [4096, 8192, 16384, 32768]
    world_sizes = [16, 32, 48]

    for idx, res in enumerate(resolutions):
        ax = axes[idx]

        # Build matrix
        matrix = np.zeros((len(batch_sizes), len(world_sizes)))
        for i, bs in enumerate(batch_sizes):
            for j, ws in enumerate(world_sizes):
                matrix[i, j] = data.get((ws, res, bs), 0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=21)

        # Labels
        ax.set_xticks(range(len(world_sizes)))
        ax.set_xticklabels([f'{ws}³' for ws in world_sizes])
        ax.set_yticks(range(len(batch_sizes)))
        ax.set_yticklabels([f'{bs//1000}K' for bs in batch_sizes])
        ax.set_xlabel('World Size')
        ax.set_ylabel('Batch Size')
        ax.set_title(f'Resolution: {res}x{res}', fontweight='bold')

        # Annotate values
        for i in range(len(batch_sizes)):
            for j in range(len(world_sizes)):
                val = matrix[i, j]
                color = 'white' if val > 10 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=9)

    fig.suptitle('Throughput (M steps/sec) by Configuration', fontsize=13, fontweight='bold', y=1.02)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('M steps/sec')

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
    ax.scatter([32768], [20.54], s=150, c='#e74c3c', zorder=5, edgecolors='black', linewidths=2)
    ax.annotate('Peak: 20.5M', (32768, 20.54), xytext=(25000, 18), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Throughput (M steps/sec)', fontsize=11)
    ax.set_title('Throughput Scaling with Batch Size', fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([f'{bs//1000}K' for bs in batch_sizes])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 24)

    plt.tight_layout()
    plt.savefig('assets/scaling.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved assets/scaling.png')


if __name__ == '__main__':
    import os
    os.makedirs('assets', exist_ok=True)

    plot_throughput_bars()
    plot_throughput_heatmap()
    plot_scaling()
    print('Done!')
