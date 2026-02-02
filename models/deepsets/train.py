"""
Training script for DeepSets Model.

Usage:
    python -m models.deepsets.train
"""

import sys
from pathlib import Path

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
from models.deepsets.model import DeepSetsModel


# ============================================================================
# Dataset
# ============================================================================

class DeepSetsDataset(Dataset):
    """
    Dataset for DeepSets that provides raw (time, angle) pairs.
    """
    
    def __init__(self, samples, model, augment=False, num_augmentations=0):
        self.data_list = []
        self.mask_list = []
        self.target_list = []
        
        for sample in samples:
            # Original sample
            data, mask = model.preprocess_sample(sample)
            target = model.normalize_target(sample['x0'], sample['y0'])
            
            self.data_list.append(data)
            self.mask_list.append(mask)
            self.target_list.append(target)
            
            # Augmented samples
            if augment and num_augmentations > 0 and len(sample['absorption_times']) > 0:
                for _ in range(num_augmentations):
                    aug_sample = self._augment_sample(sample)
                    data, mask = model.preprocess_sample(aug_sample)
                    self.data_list.append(data)
                    self.mask_list.append(mask)
                    self.target_list.append(target)
        
        self.data = np.stack(self.data_list)
        self.masks = np.stack(self.mask_list)
        self.targets = np.stack(self.target_list)
    
    def _augment_sample(self, sample):
        """Data augmentation: time noise + angle rotation"""
        aug_sample = sample.copy()
        times = sample['absorption_times'].copy()
        angles = sample['impact_angles'].copy()
        
        # Time noise (±2%)
        noise = 1.0 + np.random.uniform(-0.02, 0.02, len(times))
        times = times * noise
        
        # Small angle rotation (±5°)
        rotation = np.random.uniform(-0.087, 0.087)
        angles = angles + rotation
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        
        aug_sample['absorption_times'] = times
        aug_sample['impact_angles'] = angles
        
        return aug_sample
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for data, masks, targets in loader:
        data = data.to(device)
        masks = masks.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        preds = model(data, masks)
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
        for data, masks, targets in loader:
            data = data.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            preds = model(data, masks)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, masks, targets in loader:
            data = data.to(device)
            masks = masks.to(device)
            
            preds = model(data, masks)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    
    return np.vstack(all_preds), np.vstack(all_targets)


# ============================================================================
# Save/Load
# ============================================================================

def save_model(model, path):
    """Save the complete model"""
    torch.save({
        'network_state': model.network.state_dict(),
        'max_molecules': model.max_molecules,
        'time_mean': model.time_mean,
        'time_std': model.time_std,
        'x0_min': model.x0_min,
        'x0_max': model.x0_max,
        'y0_min': model.y0_min,
        'y0_max': model.y0_max,
    }, path)


def load_model(path, device='cpu'):
    """Load a saved model"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model = DeepSetsModel(max_molecules=checkpoint['max_molecules'])
    model.network.load_state_dict(checkpoint['network_state'])
    model.time_mean = checkpoint['time_mean']
    model.time_std = checkpoint['time_std']
    model.x0_min = checkpoint['x0_min']
    model.x0_max = checkpoint['x0_max']
    model.y0_min = checkpoint['y0_min']
    model.y0_max = checkpoint['y0_max']
    
    model.network.to(device)
    
    return model


# ============================================================================
# Main Training
# ============================================================================

def train(
    data_path='data/molecular_comm_dataset.mat',
    output_dir='outputs/deepsets',
    epochs=200,
    batch_size=64,
    lr=0.001,
    patience=50,
    num_augmentations=4,
    aggregation='mean',
    device=None
):
    """Train the DeepSets model."""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DEEPSETS MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Aggregation: {aggregation}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    samples = load_dataset(data_path)
    print(f"Total samples: {len(samples)}")
    
    # Split data
    train_data, val_data, test_data = split_data(samples, seed=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create model
    model = DeepSetsModel(
        phi_hidden_dims=[64, 128],
        phi_output_dim=128,
        rho_hidden_dims=[64, 32],
        dropout=0.3,
        aggregation=aggregation
    )
    
    # Fit normalization
    print("\nFitting normalization parameters...")
    model.fit(train_data)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create datasets
    train_dataset = DeepSetsDataset(train_data, model, augment=True, num_augmentations=num_augmentations)
    val_dataset = DeepSetsDataset(val_data, model, augment=False)
    test_dataset = DeepSetsDataset(test_data, model, augment=False)
    
    print(f"Training samples (with augmentation): {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Training setup
    network = model.network.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15)
    
    # Training loop
    print("\n" + "="*60)
    print("Training started")
    print("="*60)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_epoch(network, train_loader, optimizer, criterion, device)
        val_loss = evaluate(network, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        pbar.set_postfix({'train': f'{train_loss:.4f}', 'val': f'{val_loss:.4f}'})
        
        # Early stopping
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
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
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
    
    errors = np.sqrt((pred_denorm[:, 0] - true_denorm[:, 0])**2 + 
                     (pred_denorm[:, 1] - true_denorm[:, 1])**2)
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
