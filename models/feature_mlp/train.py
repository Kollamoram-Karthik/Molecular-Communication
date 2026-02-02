"""
Training script for Feature-based MLP Model.

Usage:
    python -m models.feature_mlp.train
    
    or from repo root:
    python models/feature_mlp/train.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.data_loader import load_dataset, split_data
from utils.metrics import calculate_position_metrics, print_position_metrics
from models.feature_mlp.model import FeatureMLPModel


# ============================================================================
# Dataset
# ============================================================================

def augment_sample(sample, noise_scale=0.02):
    """
    Data augmentation: add noise to times, rotate angles.
    """
    aug_sample = sample.copy()
    times = sample['absorption_times'].copy()
    angles = sample['impact_angles'].copy()
    
    if len(times) > 0:
        # Time noise (±2%)
        time_noise = np.random.normal(0, noise_scale, len(times)) * times
        times = np.clip(times + time_noise, 0.01, 100)
        
        # Small random rotation (±5°)
        rotation = np.random.uniform(-0.087, 0.087)
        angles = angles + rotation
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        
        aug_sample['absorption_times'] = times
        aug_sample['impact_angles'] = angles
    
    return aug_sample


class FeatureDataset(Dataset):
    """PyTorch dataset with optional augmentation."""
    
    def __init__(self, samples, model, augment=False, num_augmentations=0):
        self.features = []
        self.targets = []
        
        for sample in samples:
            # Original sample
            feat = model.extract_features(sample)
            target = model.normalize_target(sample['x0'], sample['y0'])
            self.features.append(feat)
            self.targets.append(target)
            
            # Augmented samples
            if augment and num_augmentations > 0:
                for _ in range(num_augmentations):
                    aug_sample = augment_sample(sample)
                    feat = model.extract_features(aug_sample)
                    self.features.append(feat)
                    self.targets.append(target)
        
        self.features = np.stack(self.features)
        self.targets = np.stack(self.targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            preds = model(features)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.vstack(all_preds), np.vstack(all_targets)


# ============================================================================
# Save/Load
# ============================================================================

def save_model(model, path):
    """Save the complete model"""
    save_dict = {
        'network_state': model.network.state_dict(),
        'time_mean': model.feature_extractor.time_mean,
        'time_std': model.feature_extractor.time_std,
        'distance_calibration': model.feature_extractor.distance_calibration,
        'n_molecules': model.n_molecules,
        'x0_min': model.x0_min,
        'x0_max': model.x0_max,
        'y0_min': model.y0_min,
        'y0_max': model.y0_max,
    }
    torch.save(save_dict, path)


def load_model(path, hidden_dims=[128, 128, 64], dropout=0.3):
    """Load a saved model"""
    save_dict = torch.load(path, map_location='cpu', weights_only=False)
    
    n_molecules = save_dict.get('n_molecules', 2000)
    model = FeatureMLPModel(hidden_dims=hidden_dims, dropout=dropout, n_molecules=n_molecules)
    
    model.network.load_state_dict(save_dict['network_state'])
    model.feature_extractor.time_mean = save_dict['time_mean']
    model.feature_extractor.time_std = save_dict['time_std']
    model.feature_extractor.distance_calibration = save_dict['distance_calibration']
    model.feature_extractor.n_molecules = n_molecules
    model.x0_min = save_dict['x0_min']
    model.x0_max = save_dict['x0_max']
    model.y0_min = save_dict['y0_min']
    model.y0_max = save_dict['y0_max']
    
    return model


# ============================================================================
# Main Training
# ============================================================================

def train(
    data_path='data/molecular_comm_dataset.mat',
    output_dir='outputs/feature_mlp',
    epochs=500,
    batch_size=64,
    lr=0.001,
    patience=80,
    num_augmentations=4,
    device=None
):
    """Train the Feature MLP model."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FEATURE MLP MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    samples = load_dataset(data_path)
    print(f"Total samples: {len(samples)}")
    
    # Split
    train_data, val_data, test_data = split_data(samples, seed=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create model
    model = FeatureMLPModel(hidden_dims=[128, 128, 64], dropout=0.3)
    
    # Fit physics calibration
    print("\nFitting physics calibration...")
    model.fit_physics(train_data)
    
    # Create datasets
    train_dataset = FeatureDataset(train_data, model, augment=True, num_augmentations=num_augmentations)
    val_dataset = FeatureDataset(val_data, model, augment=False)
    test_dataset = FeatureDataset(test_data, model, augment=False)
    
    print(f"Training samples (with augmentation): {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Training setup
    network = model.network.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    
    # Training loop
    print("\n" + "="*60)
    print("Training started")
    print("="*60)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_epoch(network, train_loader, optimizer, criterion, device)
        val_loss = evaluate(network, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        pbar.set_postfix({'train': f'{train_loss:.4f}', 'val': f'{val_loss:.4f}'})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in network.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    network.load_state_dict(best_state)
    model.network = network
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    y_pred, y_true = get_predictions(network, test_loader, device)
    
    # Denormalize
    true_denorm = np.array([model.denormalize_target(y) for y in y_true])
    pred_denorm = np.array([model.denormalize_target(y) for y in y_pred])
    
    metrics = calculate_position_metrics(true_denorm, pred_denorm)
    print_position_metrics(metrics, "TEST SET METRICS")
    
    # Save model
    save_model(model, output_dir / 'model.pt')
    print(f"\nModel saved to: {output_dir / 'model.pt'}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].scatter(true_denorm[:, 0], pred_denorm[:, 0], alpha=0.5, s=10, label='x0')
    axes[1].scatter(true_denorm[:, 1], pred_denorm[:, 1], alpha=0.5, s=10, label='y0')
    axes[1].plot([15, 90], [15, 90], 'k--', label='Perfect')
    axes[1].set_xlabel('True (μm)')
    axes[1].set_ylabel('Predicted (μm)')
    axes[1].set_title('Predictions vs Targets')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to: {output_dir / 'training_results.png'}")
    print("\nTraining complete!")
    
    return model, metrics


if __name__ == '__main__':
    train()
