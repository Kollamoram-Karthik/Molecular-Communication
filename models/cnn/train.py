"""
Training script for CNN Model.

Usage:
    python -m models.cnn.train
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loader import load_dataset, split_data
from utils.metrics import calculate_position_metrics, print_position_metrics
from models.cnn.model import HeatmapCNN


# ============================================================================
# Dataset
# ============================================================================

class HeatmapDataset(Dataset):
    """Dataset for heatmap images with normalized targets."""
    
    def __init__(self, samples, log_transform=True, mean=None, std=None):
        self.samples = samples
        self.log_transform = log_transform
        
        # Compute normalization stats if not provided
        if mean is None or std is None:
            heatmaps = []
            for s in samples:
                h = s['heatmap'].astype(np.float32)
                if log_transform:
                    h = np.log1p(h)
                heatmaps.append(h)
            heatmaps = np.array(heatmaps)
            self.mean = heatmaps.mean()
            self.std = heatmaps.std()
        else:
            self.mean = mean
            self.std = std
        
        # Target range: [15, 90] μm
        self.x_min, self.x_max = 15.0, 90.0
        self.y_min, self.y_max = 15.0, 90.0
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process heatmap
        heatmap = sample['heatmap'].astype(np.float32)
        if self.log_transform:
            heatmap = np.log1p(heatmap)
        heatmap = (heatmap - self.mean) / (self.std + 1e-8)
        
        # Add channel dimension
        heatmap_tensor = torch.FloatTensor(heatmap).unsqueeze(0)
        
        # Normalize targets to [0, 1]
        x0_norm = (sample['x0'] - self.x_min) / (self.x_max - self.x_min)
        y0_norm = (sample['y0'] - self.y_min) / (self.y_max - self.y_min)
        target = torch.FloatTensor([x0_norm, y0_norm])
        
        return heatmap_tensor, target
    
    def denormalize_target(self, pred):
        """Convert normalized prediction back to μm."""
        if isinstance(pred, torch.Tensor):
            x0 = pred[:, 0] * (self.x_max - self.x_min) + self.x_min
            y0 = pred[:, 1] * (self.y_max - self.y_min) + self.y_min
            return torch.stack([x0, y0], dim=1)
        else:
            x0 = pred[0] * (self.x_max - self.x_min) + self.x_min
            y0 = pred[1] * (self.y_max - self.y_min) + self.y_min
            return np.array([x0, y0])


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for heatmaps, targets in loader:
        heatmaps, targets = heatmaps.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(heatmaps)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for heatmaps, targets in loader:
            heatmaps, targets = heatmaps.to(device), targets.to(device)
            outputs = model(heatmaps)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


def get_predictions(model, loader, dataset, device):
    """Get predictions and targets in μm."""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for heatmaps, targets in loader:
            heatmaps = heatmaps.to(device)
            outputs = model(heatmaps)
            preds_um = dataset.denormalize_target(outputs.cpu())
            targets_um = dataset.denormalize_target(targets)
            all_preds.append(preds_um)
            all_targets.append(targets_um)
    
    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


def save_model(model, dataset, path, metrics=None):
    """Save model with normalization parameters."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalization': {
            'x_min': dataset.x_min,
            'x_max': dataset.x_max,
            'y_min': dataset.y_min,
            'y_max': dataset.y_max,
            'heatmap_mean': dataset.mean,
            'heatmap_std': dataset.std,
            'log_transform': dataset.log_transform,
        },
        'metrics': metrics,
    }, path)


def load_model(path, device='cpu'):
    """Load a saved model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model = HeatmapCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint['normalization']


# ============================================================================
# Main Training
# ============================================================================

def train(
    data_path='data/molecular_comm_dataset.mat',
    output_dir='outputs/cnn',
    epochs=100,
    batch_size=64,
    lr=1e-3,
    patience=20,
    dropout=0.3,
    device=None
):
    """Train the CNN model."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CNN MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    
    # Load data with heatmaps
    print(f"\nLoading data from: {data_path}")
    samples = load_dataset(data_path, load_heatmaps=True)
    print(f"Total samples: {len(samples)}")
    print(f"Heatmap shape: {samples[0]['heatmap'].shape}")
    
    # Split data
    train_samples, val_samples, test_samples = split_data(samples, seed=42)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Create datasets
    train_dataset = HeatmapDataset(train_samples, log_transform=True)
    val_dataset = HeatmapDataset(val_samples, log_transform=True, 
                                  mean=train_dataset.mean, std=train_dataset.std)
    test_dataset = HeatmapDataset(test_samples, log_transform=True,
                                   mean=train_dataset.mean, std=train_dataset.std)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = HeatmapCNN(dropout=dropout).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    
    # Training loop
    print("\n" + "="*60)
    print("Training started")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            preds, targets = get_predictions(model, val_loader, val_dataset, device)
            metrics = calculate_position_metrics(targets, preds)
            tqdm.write(f"Epoch {epoch+1:3d} | Val Loss: {val_loss:.6f} | MAE: {metrics['dist_mean']:.2f}μm")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    preds, targets = get_predictions(model, test_loader, test_dataset, device)
    metrics = calculate_position_metrics(targets, preds)
    print_position_metrics(metrics, "TEST SET METRICS")
    
    # Save model
    save_model(model, train_dataset, output_dir / 'model.pt', metrics)
    print(f"\nModel saved to: {output_dir / 'model.pt'}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].scatter(targets[:, 0], preds[:, 0], alpha=0.5, s=10, label='x0')
    axes[1].scatter(targets[:, 1], preds[:, 1], alpha=0.5, s=10, label='y0')
    axes[1].plot([15, 90], [15, 90], 'k--', label='Perfect')
    axes[1].set_xlabel('True (μm)')
    axes[1].set_ylabel('Predicted (μm)')
    axes[1].set_title('Predictions vs Targets')
    axes[1].legend()
    axes[1].grid(True)
    
    errors = np.sqrt((preds[:, 0] - targets[:, 0])**2 + (preds[:, 1] - targets[:, 1])**2)
    axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}μm')
    axes[2].set_xlabel('Euclidean Error (μm)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to: {output_dir / 'training_results.png'}")
    print("\nTraining complete!")
    
    return model, metrics


if __name__ == '__main__':
    train()
