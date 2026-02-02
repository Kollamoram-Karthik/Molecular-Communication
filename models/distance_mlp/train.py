"""
Training script for Distance MLP Model.

Usage:
    python -m models.distance_mlp.train
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.data_loader import load_dataset
from utils.metrics import calculate_distance_metrics, print_distance_metrics
from models.distance_mlp.model import DistanceMLPModel


def prepare_data(samples, model, augment=True, num_augmentations=2):
    """Prepare features and targets with optional augmentation"""
    features_list = []
    targets_list = []
    
    for s in samples:
        features = model.extract_features(s)
        target = model.normalize_target(s['distance'])
        features_list.append(features)
        targets_list.append(target)
        
        # Time-only augmentation
        if augment and len(s['absorption_times']) > 0:
            for _ in range(num_augmentations):
                aug_sample = s.copy()
                noise = 1.0 + np.random.uniform(-0.02, 0.02, len(s['absorption_times']))
                aug_sample['absorption_times'] = s['absorption_times'] * noise
                features = model.extract_features(aug_sample)
                features_list.append(features)
                targets_list.append(target)
    
    return np.array(features_list), np.array(targets_list)


def save_model(model, state, path):
    """Save the trained model"""
    torch.save({
        'network_state': state['network_state'],
        'feature_extractor': {
            'time_mean': model.feature_extractor.time_mean,
            'time_std': model.feature_extractor.time_std,
            'distance_calibration': model.feature_extractor.distance_calibration,
            'n_molecules': model.n_molecules
        },
        'dist_range': (model.dist_min, model.dist_max)
    }, path)


def load_model(path):
    """Load a trained model"""
    state = torch.load(path, weights_only=False)
    
    fe_state = state['feature_extractor']
    model = DistanceMLPModel(
        hidden_dims=[64, 32],
        dropout=0.3,
        n_molecules=fe_state['n_molecules']
    )
    
    model.feature_extractor.time_mean = fe_state['time_mean']
    model.feature_extractor.time_std = fe_state['time_std']
    model.feature_extractor.distance_calibration = fe_state['distance_calibration']
    model.dist_min, model.dist_max = state['dist_range']
    model.network.load_state_dict(state['network_state'])
    
    return model


def train(
    data_path='data/molecular_comm_dataset.mat',
    output_dir='outputs/distance_mlp',
    epochs=300,
    batch_size=64,
    lr=0.001,
    patience=50,
    num_augmentations=2
):
    """Train the Distance MLP model."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DISTANCE MLP MODEL TRAINING")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading data from: {data_path}")
    samples = load_dataset(data_path)
    print(f"Total samples: {len(samples)}")
    
    n_molecules = samples[0]['N0']
    
    # Distance statistics
    distances = np.array([s['distance'] for s in samples])
    print(f"\nDistance range: {distances.min():.1f} - {distances.max():.1f} μm")
    
    # Initialize model
    model = DistanceMLPModel(hidden_dims=[64, 32], dropout=0.3, n_molecules=n_molecules)
    
    # Split data
    train_val_samples, test_samples = train_test_split(samples, test_size=0.15, random_state=42)
    train_samples, val_samples = train_test_split(train_val_samples, test_size=0.15/0.85, random_state=42)
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Fit physics calibration
    print("\nFitting physics calibration...")
    model.fit_physics(train_samples)
    
    # Prepare datasets
    X_train, y_train = prepare_data(train_samples, model, augment=True, num_augmentations=num_augmentations)
    X_val, y_val = prepare_data(val_samples, model, augment=False)
    X_test, y_test = prepare_data(test_samples, model, augment=False)
    
    print(f"Training samples (with augmentation): {len(X_train)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size
    )
    
    # Training setup
    network = model.network
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    
    # Training loop
    print("\n" + "="*60)
    print("Training started")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        network.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = network(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        network.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = network(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {'network_state': network.state_dict()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    network.load_state_dict(best_state['network_state'])
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    network.eval()
    predictions = []
    physics_estimates = []
    actuals = []
    
    with torch.no_grad():
        for s in test_samples:
            features = model.extract_features(s)
            X = torch.FloatTensor(features).unsqueeze(0)
            pred_norm = network(X).numpy()[0]
            pred_dist = model.denormalize_target(pred_norm)
            
            predictions.append(pred_dist)
            physics_estimates.append(model.get_physics_estimate(s))
            actuals.append(s['distance'])
    
    predictions = np.array(predictions)
    physics_estimates = np.array(physics_estimates)
    actuals = np.array(actuals)
    
    metrics = calculate_distance_metrics(actuals, predictions)
    print_distance_metrics(metrics, "TEST SET METRICS")
    
    physics_metrics = calculate_distance_metrics(actuals, physics_estimates)
    print_distance_metrics(physics_metrics, "PHYSICS BASELINE")
    
    improvement = (physics_metrics['mae'] - metrics['mae']) / physics_metrics['mae'] * 100
    print(f"\nNN improvement over physics: {improvement:.1f}%")
    
    # Save model
    save_model(model, best_state, output_dir / 'model.pt')
    print(f"\nModel saved to: {output_dir / 'model.pt'}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].scatter(actuals, predictions, alpha=0.5, s=10)
    axes[1].plot([20, 130], [20, 130], 'k--', label='Perfect')
    axes[1].set_xlabel('True Distance (μm)')
    axes[1].set_ylabel('Predicted Distance (μm)')
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
