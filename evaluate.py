#!/usr/bin/env python3
"""
Evaluate a trained model on test data with detailed analysis.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml_model.data_loader import load_dataset, split_data
from ml_model.physics_model import PhysicsInformedModel
from ml_model.trainer import MolecularDataset, get_predictions, calculate_metrics, load_model
from torch.utils.data import DataLoader


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
    test_dataset = MolecularDataset(test_data, model, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
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
    
    # Denormalize for display
    true_denorm = np.array([model.denormalize_target(y) for y in y_true])
    pred_denorm = np.array([model.denormalize_target(y) for y in y_pred])
    
    # Show 10 random sample predictions
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS (10 random test samples)")
    print('='*60)
    print(f"{'#':>3} {'True X0':>10} {'True Y0':>10} {'Pred X0':>10} {'Pred Y0':>10} {'Error':>10}")
    print("-"*60)
    
    np.random.seed(456)
    sample_indices = np.random.choice(len(y_true), 10, replace=False)
    
    for idx in sample_indices:
        error = np.sqrt((true_denorm[idx, 0] - pred_denorm[idx, 0])**2 + 
                        (true_denorm[idx, 1] - pred_denorm[idx, 1])**2)
        print(f"{idx:3d} {true_denorm[idx, 0]:10.2f} {true_denorm[idx, 1]:10.2f} "
              f"{pred_denorm[idx, 0]:10.2f} {pred_denorm[idx, 1]:10.2f} "
              f"{error:10.2f} μm")
    
    print('='*60)
    
    # Error distribution by distance
    print(f"\n{'='*60}")
    print("ERROR BY TRANSMITTER DISTANCE")
    print('='*60)
    
    distances = np.sqrt(true_denorm[:, 0]**2 + true_denorm[:, 1]**2)
    errors = np.sqrt(np.sum((true_denorm - pred_denorm)**2, axis=1))
    
    # Bin by distance
    bins = [(20, 40), (40, 60), (60, 80), (80, 120)]
    for d_min, d_max in bins:
        mask = (distances >= d_min) & (distances < d_max)
        if mask.sum() > 0:
            mean_err = np.mean(errors[mask])
            median_err = np.median(errors[mask])
            print(f"Distance {d_min:3d}-{d_max:3d} μm: Mean={mean_err:.2f} μm, Median={median_err:.2f} μm (n={mask.sum()})")
    
    print('='*60)


if __name__ == '__main__':
    main()
