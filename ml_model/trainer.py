"""
Training utilities for the physics-informed model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle


def augment_sample(sample, noise_scale=0.02):
    """
    Data augmentation for a single sample.
    
    Augmentations:
    1. Add small noise to absorption times
    2. Small rotation of angles (symmetry)
    """
    aug_sample = sample.copy()
    
    times = sample['absorption_times'].copy()
    angles = sample['impact_angles'].copy()
    
    if len(times) > 0:
        # Add time noise (±2% of each time)
        time_noise = np.random.normal(0, noise_scale, len(times)) * times
        times = np.clip(times + time_noise, 0.01, 100)
        
        # Small random rotation (±5 degrees)
        rotation = np.random.uniform(-0.087, 0.087)  # ±5° in radians
        angles = angles + rotation
        # Wrap to [-π, π]
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        
        aug_sample['absorption_times'] = times
        aug_sample['impact_angles'] = angles
    
    return aug_sample


class MolecularDataset(Dataset):
    """PyTorch dataset for molecular communication data"""
    
    def __init__(self, samples, model, augment=False, num_augmentations=0):
        self.samples = samples
        self.model = model
        self.augment = augment
        self.num_augmentations = num_augmentations
        
        # Pre-compute features
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
                    self.targets.append(target)  # Same target
        
        self.features = np.stack(self.features)
        self.targets = np.stack(self.targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            predictions = model(features)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def get_predictions(model, data_loader, device):
    """Get all predictions from a model"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            preds = model(features)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    
    return np.vstack(all_preds), np.vstack(all_targets)


def calculate_metrics(y_true, y_pred, physics_model):
    """
    Calculate comprehensive metrics.
    
    Args:
        y_true: (N, 2) normalized true values
        y_pred: (N, 2) normalized predictions
        physics_model: for denormalization
    
    Returns:
        dict of metrics
    """
    # Denormalize
    true_denorm = np.array([physics_model.denormalize_target(y) for y in y_true])
    pred_denorm = np.array([physics_model.denormalize_target(y) for y in y_pred])
    
    # Per-coordinate errors
    x_errors = true_denorm[:, 0] - pred_denorm[:, 0]
    y_errors = true_denorm[:, 1] - pred_denorm[:, 1]
    
    x_rmse = np.sqrt(np.mean(x_errors**2))
    y_rmse = np.sqrt(np.mean(y_errors**2))
    x_mae = np.mean(np.abs(x_errors))
    y_mae = np.mean(np.abs(y_errors))
    
    # Distance (Euclidean) error
    dist_errors = np.sqrt(x_errors**2 + y_errors**2)
    
    metrics = {
        'x_rmse': x_rmse,
        'y_rmse': y_rmse,
        'x_mae': x_mae,
        'y_mae': y_mae,
        'dist_mean': np.mean(dist_errors),
        'dist_median': np.median(dist_errors),
        'dist_std': np.std(dist_errors),
        'dist_max': np.max(dist_errors),
        'dist_p90': np.percentile(dist_errors, 90),
        'dist_p95': np.percentile(dist_errors, 95),
        'within_1um': np.mean(dist_errors < 1) * 100,
        'within_2um': np.mean(dist_errors < 2) * 100,
        'within_5um': np.mean(dist_errors < 5) * 100,
    }
    
    return metrics


def train_model(physics_model, train_data, val_data, 
                num_epochs=200, batch_size=32, lr=0.001,
                patience=30, device='cpu', augment=True, num_augmentations=3):
    """
    Full training loop with early stopping and data augmentation.
    
    Returns:
        Trained model, training history
    """
    # Create datasets (with augmentation for training)
    train_dataset = MolecularDataset(
        train_data, physics_model, 
        augment=augment, 
        num_augmentations=num_augmentations
    )
    val_dataset = MolecularDataset(val_data, physics_model, augment=False)
    
    print(f"Training samples (with augmentation): {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Move network to device
    network = physics_model.network.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    pbar = tqdm(range(num_epochs), desc="Training")
    
    for epoch in pbar:
        train_loss = train_epoch(network, train_loader, optimizer, criterion, device)
        val_loss = evaluate(network, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
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
    physics_model.network = network
    
    return physics_model, history


def save_model(physics_model, path):
    """Save the complete model"""
    save_dict = {
        'network_state': physics_model.network.state_dict(),
        'time_mean': physics_model.feature_extractor.time_mean,
        'time_std': physics_model.feature_extractor.time_std,
        'distance_calibration': physics_model.feature_extractor.distance_calibration,
        'n_molecules': physics_model.n_molecules,
        'x0_min': physics_model.x0_min,
        'x0_max': physics_model.x0_max,
        'y0_min': physics_model.y0_min,
        'y0_max': physics_model.y0_max,
    }
    torch.save(save_dict, path)


def load_model(path, hidden_dims=[128, 128, 64], dropout=0.3):
    """Load a saved model"""
    from ml_model.physics_model import PhysicsInformedModel
    
    save_dict = torch.load(path, map_location='cpu', weights_only=False)
    
    n_molecules = save_dict.get('n_molecules', 500)
    model = PhysicsInformedModel(hidden_dims=hidden_dims, dropout=dropout, n_molecules=n_molecules)
    
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
