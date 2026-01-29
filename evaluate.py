#!/usr/bin/env python3
"""
Evaluate a trained model on test data with detailed sample predictions.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml_model.data_loader import load_dataset, split_data
from ml_model.physics_model import PhysicsInformedModel, CorrectionNetwork
from ml_model.trainer import MolecularDataset, get_predictions, calculate_metrics
from torch.utils.data import DataLoader


def load_model(path, hidden_dims=[128, 64, 32], dropout=0.3):
    """Load a saved model"""
    model = PhysicsInformedModel(hidden_dims=hidden_dims, dropout=dropout)
    
    save_dict = torch.load(path, map_location='cpu', weights_only=False)
    
    model.network.load_state_dict(save_dict['network_state'])
    model.feature_extractor.time_mean = save_dict['time_mean']
    model.feature_extractor.time_std = save_dict['time_std']
    model.feature_extractor.distance_calibration = save_dict['distance_calibration']
    model.x0_min = save_dict['x0_min']
    model.x0_max = save_dict['x0_max']
    model.y0_min = save_dict['y0_min']
    model.y0_max = save_dict['y0_max']
    
    return model


def main():
    MODEL_PATH = 'outputs/physics_model.pt'
    DATASET_PATH = 'molecular_comm_dataset.mat'
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    model.network.eval()
    
    # Load data
    print(f"Loading dataset from: {DATASET_PATH}")
    samples = load_dataset(DATASET_PATH)
    _, _, test_data = split_data(samples, seed=42)
    print(f"Test samples: {len(test_data)}")
    
    # Create test loader
    test_dataset = MolecularDataset(test_data, model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    y_pred, y_true = get_predictions(model.network, test_loader, 'cpu')
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, model)
    
    print(f"\n{'='*60}")
    print("TEST SET METRICS")
    print('='*60)
    print(f"X Coordinate:")
    print(f"  RMSE: {metrics['x_rmse']:.3f} μm")
    print(f"  MAE:  {metrics['x_mae']:.3f} μm")
    print(f"Y Coordinate:")
    print(f"  RMSE: {metrics['y_rmse']:.3f} μm")
    print(f"  MAE:  {metrics['y_mae']:.3f} μm")
    print(f"Distance Error:")
    print(f"  Mean:   {metrics['dist_mean']:.3f} μm")
    print(f"  Median: {metrics['dist_median']:.3f} μm")
    print(f"  Std:    {metrics['dist_std']:.3f} μm")
    print(f"  Max:    {metrics['dist_max']:.3f} μm")
    print(f"  P90:    {metrics['dist_p90']:.3f} μm")
    print(f"  P95:    {metrics['dist_p95']:.3f} μm")
    print(f"Accuracy:")
    print(f"  Within 1 μm: {metrics['within_1um']:.1f}%")
    print(f"  Within 2 μm: {metrics['within_2um']:.1f}%")
    print(f"  Within 5 μm: {metrics['within_5um']:.1f}%")
    print('='*60)
    
    # Show 10 sample predictions
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS (10 random test samples)")
    print('='*60)
    print(f"{'#':>3} {'True X0':>10} {'True Y0':>10} {'Pred X0':>10} {'Pred Y0':>10} {'Error':>10}")
    print("-"*60)
    
    # Random sample indices
    np.random.seed(123)
    sample_indices = np.random.choice(len(y_true), 10, replace=False)
    
    true_denorm = np.array([model.denormalize_target(y) for y in y_true])
    pred_denorm = np.array([model.denormalize_target(y) for y in y_pred])
    
    for idx in sample_indices:
        error = np.sqrt((true_denorm[idx, 0] - pred_denorm[idx, 0])**2 + 
                        (true_denorm[idx, 1] - pred_denorm[idx, 1])**2)
        print(f"{idx:3d} {true_denorm[idx, 0]:10.2f} {true_denorm[idx, 1]:10.2f} "
              f"{pred_denorm[idx, 0]:10.2f} {pred_denorm[idx, 1]:10.2f} "
              f"{error:10.2f} μm")
    
    print('='*60)


if __name__ == '__main__':
    main()
