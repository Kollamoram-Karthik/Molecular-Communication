"""
Test heatmap generation and visualization in Python
"""

import numpy as np
import matplotlib.pyplot as plt
from ml_model.data_loader import load_dataset
import h5py


def visualize_heatmaps(mat_path, num_samples=4):
    """Visualize heatmaps from the dataset"""
    
    print("Loading dataset with heatmaps...")
    samples = load_dataset(mat_path, load_heatmaps=True)
    
    # Load metadata
    with h5py.File(mat_path, 'r') as f:
        time_bins = int(np.array(f['time_bins']).flatten()[0])
        angle_bins = int(np.array(f['angle_bins']).flatten()[0])
        time_min = float(np.array(f['time_min']).flatten()[0])
        time_max = float(np.array(f['time_max']).flatten()[0])
        angle_min = float(np.array(f['angle_min']).flatten()[0])
        angle_max = float(np.array(f['angle_max']).flatten()[0])
    
    print(f"Dataset loaded: {len(samples)} samples")
    print(f"Heatmap resolution: {time_bins} × {angle_bins}")
    print(f"Time range: [{time_min}, {time_max}] s")
    print(f"Angle range: [{angle_min:.2f}, {angle_max:.2f}] rad")
    
    # Select random samples
    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        sample = samples[idx]
        
        if 'heatmap' not in sample:
            print(f"Warning: Sample {idx} does not have heatmap data")
            continue
        
        heatmap = sample['heatmap']
        
        # Display heatmap
        im = axes[i].imshow(heatmap, 
                           aspect='auto',
                           origin='lower',
                           extent=[angle_min, angle_max, time_min, time_max],
                           cmap='hot',
                           interpolation='nearest')
        
        axes[i].set_xlabel('Impact Angle (rad)', fontsize=10)
        axes[i].set_ylabel('Absorption Time (s)', fontsize=10)
        axes[i].set_title(f'Sample {idx}: ({sample["x0"]:.1f}, {sample["y0"]:.1f}) μm, N0={sample["N0"]}',
                         fontsize=11)
        axes[i].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Molecule Count')
    
    plt.suptitle(f'Time-Angle Heatmaps ({time_bins}×{angle_bins} bins)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/heatmap_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: outputs/heatmap_visualization.png")
    plt.show()
    
    # Print statistics
    print("\n=== Heatmap Statistics ===")
    heatmaps = [s['heatmap'] for s in samples if 'heatmap' in s]
    if heatmaps:
        heatmaps_array = np.array(heatmaps)
        print(f"Shape: {heatmaps_array.shape}")
        print(f"Min value: {heatmaps_array.min()}")
        print(f"Max value: {heatmaps_array.max()}")
        print(f"Mean value: {heatmaps_array.mean():.2f}")
        print(f"Std value: {heatmaps_array.std():.2f}")
        
        # Memory usage
        memory_mb = heatmaps_array.nbytes / (1024**2)
        print(f"Total memory: {memory_mb:.2f} MB")
        print(f"Memory per sample: {memory_mb / len(samples) * 1024:.2f} KB")
        
        # Sparsity analysis
        sparsity = (heatmaps_array == 0).sum() / heatmaps_array.size * 100
        print(f"Sparsity: {sparsity:.1f}% (zero pixels)")


if __name__ == '__main__':
    visualize_heatmaps('molecular_comm_dataset.mat', num_samples=4)
