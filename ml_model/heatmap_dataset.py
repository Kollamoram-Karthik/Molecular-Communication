"""
PyTorch Dataset class for CNN training with heatmaps
Template for future CNN implementation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ml_model.data_loader import load_dataset, split_data


class HeatmapDataset(Dataset):
    """
    PyTorch Dataset for molecular communication heatmaps.
    
    Each sample is a 2D heatmap (time x angle) representing molecule absorption pattern.
    Target is the transmitter position (x0, y0).
    """
    
    def __init__(self, samples, normalize=True, log_transform=False):
        """
        Args:
            samples: List of sample dicts from load_dataset()
            normalize: Whether to normalize heatmaps
            log_transform: Whether to apply log(x+1) transform
        """
        self.samples = samples
        self.normalize = normalize
        self.log_transform = log_transform
        
        # Precompute normalization statistics if needed
        if self.normalize:
            heatmaps = np.array([s['heatmap'] for s in samples])
            self.mean = heatmaps.mean()
            self.std = heatmaps.std()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get heatmap and convert to tensor
        heatmap = sample['heatmap'].astype(np.float32)
        
        # Apply log transform if requested
        if self.log_transform:
            heatmap = np.log1p(heatmap)  # log(1 + x)
        
        # Normalize if requested
        if self.normalize:
            heatmap = (heatmap - self.mean) / (self.std + 1e-8)
        
        # Convert to tensor with channel dimension: (1, H, W)
        heatmap_tensor = torch.FloatTensor(heatmap).unsqueeze(0)
        
        # Target: (x0, y0) position
        target = torch.FloatTensor([sample['x0'], sample['y0']])
        
        return heatmap_tensor, target


def create_dataloaders(mat_path, batch_size=32, num_workers=4, 
                       normalize=True, log_transform=False):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        mat_path: Path to molecular_comm_dataset.mat
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        normalize: Whether to normalize heatmaps
        log_transform: Whether to apply log transform
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Load dataset with heatmaps
    print("Loading dataset with heatmaps...")
    samples = load_dataset(mat_path, load_heatmaps=True)
    
    # Split data
    train_samples, val_samples, test_samples = split_data(samples)
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")
    
    # Create datasets
    train_dataset = HeatmapDataset(train_samples, normalize, log_transform)
    val_dataset = HeatmapDataset(val_samples, normalize, log_transform)
    test_dataset = HeatmapDataset(test_samples, normalize, log_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def test_dataloader():
    """Test the dataloader implementation"""
    
    print("=== Testing HeatmapDataset ===\n")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        'molecular_comm_dataset.mat',
        batch_size=8,
        num_workers=0,  # Use 0 for testing
        normalize=True,
        log_transform=True
    )
    
    # Test batch
    print("\nTesting first batch...")
    heatmaps, targets = next(iter(train_loader))
    
    print(f"Heatmap batch shape: {heatmaps.shape}")  # Should be (8, 1, 100, 100)
    print(f"Target batch shape: {targets.shape}")    # Should be (8, 2)
    print(f"Heatmap dtype: {heatmaps.dtype}")
    print(f"Target dtype: {targets.dtype}")
    print(f"\nHeatmap statistics:")
    print(f"  Min: {heatmaps.min():.4f}")
    print(f"  Max: {heatmaps.max():.4f}")
    print(f"  Mean: {heatmaps.mean():.4f}")
    print(f"  Std: {heatmaps.std():.4f}")
    print(f"\nTarget statistics:")
    print(f"  x0 range: [{targets[:, 0].min():.2f}, {targets[:, 0].max():.2f}]")
    print(f"  y0 range: [{targets[:, 1].min():.2f}, {targets[:, 1].max():.2f}]")
    
    # Visualize one sample
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i in range(4):
        heatmap = heatmaps[i, 0].cpu().numpy()
        x0, y0 = targets[i].cpu().numpy()
        
        im = axes[i].imshow(heatmap, cmap='hot', aspect='auto', origin='lower')
        axes[i].set_title(f'Sample {i+1}\n(x0={x0:.1f}, y0={y0:.1f})')
        axes[i].set_xlabel('Angle Bin')
        axes[i].set_ylabel('Time Bin')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle('Preprocessed Heatmaps (Log-transformed & Normalized)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/dataloader_test.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: outputs/dataloader_test.png")
    plt.show()
    
    print("\n✓ Dataloader test successful!")


if __name__ == '__main__':
    test_dataloader()
