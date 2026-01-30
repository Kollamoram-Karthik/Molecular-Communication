"""
Training script for the Distance Prediction Model.

This simpler model predicts only the Euclidean distance of the transmitter
from the origin, using only absorption time information.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

from ml_model.data_loader import load_dataset
from ml_model.distance_model import DistanceModel


def prepare_data(samples, model, augment=True, num_augmentations=2):
    """Prepare features and targets with optional augmentation"""
    
    features_list = []
    targets_list = []
    
    for s in samples:
        # Original sample
        features = model.extract_features(s)
        target = model.normalize_target(s['distance'])
        
        features_list.append(features)
        targets_list.append(target)
        
        # Time-only augmentation (no angle rotation needed)
        if augment and len(s['absorption_times']) > 0:
            for _ in range(num_augmentations):
                aug_sample = s.copy()
                # Small time noise (±2%)
                noise = 1.0 + np.random.uniform(-0.02, 0.02, len(s['absorption_times']))
                aug_sample['absorption_times'] = s['absorption_times'] * noise
                
                features = model.extract_features(aug_sample)
                features_list.append(features)
                targets_list.append(target)
    
    return np.array(features_list), np.array(targets_list)


def train_distance_model(
    data_path='molecular_comm_dataset.mat',
    output_dir='outputs',
    epochs=300,
    batch_size=64,
    lr=0.001,
    patience=50,
    num_augmentations=2,
    val_split=0.15,
    test_split=0.15
):
    """Train the distance prediction model"""
    
    print("=" * 60)
    print("Distance Prediction Model Training")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from {data_path}...")
    samples = load_dataset(data_path)
    print(f"Total samples: {len(samples)}")
    
    # Get n_molecules from first sample
    n_molecules = samples[0]['N0']
    print(f"Molecules per sample: {n_molecules}")
    
    # Show distance distribution
    distances = np.array([s['distance'] for s in samples])
    print(f"\nDistance statistics:")
    print(f"  Min: {distances.min():.2f} μm")
    print(f"  Max: {distances.max():.2f} μm")
    print(f"  Mean: {distances.mean():.2f} μm")
    print(f"  Std: {distances.std():.2f} μm")
    
    # Initialize model
    model = DistanceModel(
        hidden_dims=[64, 32],
        dropout=0.3,
        n_molecules=n_molecules
    )
    
    # Split data
    train_val_samples, test_samples = train_test_split(
        samples, test_size=test_split, random_state=42
    )
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=val_split/(1-test_split), random_state=42
    )
    
    print(f"\nData splits:")
    print(f"  Training: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")
    
    # Fit physics calibration on training data
    print("\nCalibrating physics model on training data...")
    model.fit_physics(train_samples)
    
    # Prepare datasets
    print("Extracting features...")
    X_train, y_train = prepare_data(train_samples, model, augment=True, num_augmentations=num_augmentations)
    X_val, y_val = prepare_data(val_samples, model, augment=False)
    X_test, y_test = prepare_data(test_samples, model, augment=False)
    
    print(f"\nFeature shapes:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training setup
    network = model.network
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training started")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                'network_state': network.state_dict(),
                'feature_extractor': {
                    'time_mean': model.feature_extractor.time_mean,
                    'time_std': model.feature_extractor.time_std,
                    'distance_calibration': model.feature_extractor.distance_calibration,
                    'n_molecules': model.n_molecules
                },
                'dist_range': (model.dist_min, model.dist_max)
            }
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or patience_counter == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, patience={patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    network.load_state_dict(best_state['network_state'])
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    network.eval()
    predictions = []
    physics_estimates = []
    actuals = []
    
    with torch.no_grad():
        for i, s in enumerate(test_samples):
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
    
    # Calculate errors
    nn_errors = np.abs(predictions - actuals)
    physics_errors = np.abs(physics_estimates - actuals)
    
    print(f"\nNeural Network Performance:")
    print(f"  Mean error: {nn_errors.mean():.2f} μm")
    print(f"  Median error: {np.median(nn_errors):.2f} μm")
    print(f"  Std error: {nn_errors.std():.2f} μm")
    print(f"  Max error: {nn_errors.max():.2f} μm")
    print(f"  Within 2 μm: {100 * np.mean(nn_errors < 2):.1f}%")
    print(f"  Within 5 μm: {100 * np.mean(nn_errors < 5):.1f}%")
    print(f"  Within 10 μm: {100 * np.mean(nn_errors < 10):.1f}%")
    
    print(f"\nPhysics Baseline:")
    print(f"  Mean error: {physics_errors.mean():.2f} μm")
    print(f"  Median error: {np.median(physics_errors):.2f} μm")
    
    improvement = (physics_errors.mean() - nn_errors.mean()) / physics_errors.mean() * 100
    print(f"\nNN improvement over physics: {improvement:.1f}%")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'distance_model.pt')
    torch.save(best_state, save_path)
    print(f"\nModel saved to {save_path}")
    
    # Show sample predictions
    print("\n" + "-" * 60)
    print("Sample Predictions:")
    print("-" * 60)
    print(f"{'Actual':>10} {'Predicted':>10} {'Physics':>10} {'NN Error':>10}")
    print("-" * 60)
    
    for i in range(min(10, len(test_samples))):
        print(f"{actuals[i]:>10.2f} {predictions[i]:>10.2f} {physics_estimates[i]:>10.2f} {nn_errors[i]:>10.2f}")
    
    print("-" * 60)
    
    return model, best_state


def load_distance_model(model_path='outputs/distance_model.pt'):
    """Load a trained distance model"""
    state = torch.load(model_path, weights_only=False)
    
    fe_state = state['feature_extractor']
    model = DistanceModel(
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


if __name__ == '__main__':
    train_distance_model(
        epochs=300,
        batch_size=64,
        lr=0.001,
        patience=50,
        num_augmentations=2
    )
