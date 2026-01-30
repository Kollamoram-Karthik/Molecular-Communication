#!/usr/bin/env python3
"""
Main training script for Physics-Informed Transmitter Localization.

This model combines:
1. Physics-based feature extraction (diffusion theory)
2. Neural network for learning corrections
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add ml_model to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_model.data_loader import load_dataset, split_data
from ml_model.physics_model import PhysicsInformedModel
from ml_model.trainer import (
    train_model, save_model, calculate_metrics,
    MolecularDataset, get_predictions
)
from torch.utils.data import DataLoader


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{title}")
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


def evaluate_physics_baseline(samples, model):
    """Evaluate pure physics-based estimation (no neural network)"""
    errors = []
    for sample in samples:
        x0_est, y0_est = model.get_physics_estimate(sample)
        x0_true, y0_true = sample['x0'], sample['y0']
        dist_error = np.sqrt((x0_est - x0_true)**2 + (y0_est - y0_true)**2)
        errors.append(dist_error)
    
    errors = np.array(errors)
    print(f"\n{'='*60}")
    print("Physics-Only Baseline (No Neural Network)")
    print('='*60)
    print(f"  Mean Distance Error:   {np.mean(errors):.3f} μm")
    print(f"  Median Distance Error: {np.median(errors):.3f} μm")
    print(f"  Max Distance Error:    {np.max(errors):.3f} μm")
    print(f"  P90 Distance Error:    {np.percentile(errors, 90):.3f} μm")
    print('='*60)
    return errors


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(y_true, y_pred, model, save_path):
    """Plot predictions vs true values"""
    # Denormalize
    true_denorm = np.array([model.denormalize_target(y) for y in y_true])
    pred_denorm = np.array([model.denormalize_target(y) for y in y_pred])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X coordinate
    axes[0].scatter(true_denorm[:, 0], pred_denorm[:, 0], alpha=0.6, s=30)
    axes[0].plot([15, 90], [15, 90], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('True X0 (μm)')
    axes[0].set_ylabel('Predicted X0 (μm)')
    axes[0].set_title('X Coordinate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Y coordinate
    axes[1].scatter(true_denorm[:, 1], pred_denorm[:, 1], alpha=0.6, s=30)
    axes[1].plot([15, 90], [15, 90], 'r--', linewidth=2, label='Perfect')
    axes[1].set_xlabel('True Y0 (μm)')
    axes[1].set_ylabel('Predicted Y0 (μm)')
    axes[1].set_title('Y Coordinate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    # 2D scatter
    axes[2].scatter(true_denorm[:, 0], true_denorm[:, 1], c='blue', alpha=0.6, s=30, label='True')
    axes[2].scatter(pred_denorm[:, 0], pred_denorm[:, 1], c='red', alpha=0.6, s=30, label='Predicted')
    for i in range(len(true_denorm)):
        axes[2].plot([true_denorm[i, 0], pred_denorm[i, 0]], 
                     [true_denorm[i, 1], pred_denorm[i, 1]], 
                     'gray', alpha=0.3, linewidth=0.5)
    axes[2].set_xlabel('X0 (μm)')
    axes[2].set_ylabel('Y0 (μm)')
    axes[2].set_title('2D Position (True vs Predicted)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_error_distribution(y_true, y_pred, model, save_path):
    """Plot error distribution"""
    true_denorm = np.array([model.denormalize_target(y) for y in y_true])
    pred_denorm = np.array([model.denormalize_target(y) for y in y_pred])
    
    dist_errors = np.sqrt(np.sum((true_denorm - pred_denorm)**2, axis=1))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(dist_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(dist_errors), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(dist_errors):.2f} μm')
    axes[0].axvline(np.median(dist_errors), color='g', linestyle='--', 
                    label=f'Median: {np.median(dist_errors):.2f} μm')
    axes[0].set_xlabel('Distance Error (μm)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CDF
    sorted_errors = np.sort(dist_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cdf * 100, linewidth=2)
    axes[1].axhline(90, color='r', linestyle='--', alpha=0.5, label='90%')
    axes[1].axvline(np.percentile(dist_errors, 90), color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Distance Error (μm)')
    axes[1].set_ylabel('Cumulative %')
    axes[1].set_title('Cumulative Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # Configuration
    DATASET_PATH = 'molecular_comm_dataset.mat'
    OUTPUT_DIR = Path('outputs')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    samples = load_dataset(DATASET_PATH)
    print(f"Loaded {len(samples)} samples")
    
    # Split data
    train_data, val_data, test_data = split_data(samples, seed=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    model = PhysicsInformedModel(
        hidden_dims=[128, 128, 64],
        dropout=0.3
    )
    
    # Fit physics calibration
    print("Fitting physics calibration on training data...")
    model.fit_physics(train_data)
    
    # Evaluate physics-only baseline
    physics_errors = evaluate_physics_baseline(test_data, model)
    
    # Train neural network
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK")
    print("="*60)
    
    model, history = train_model(
        model, train_data, val_data,
        num_epochs=500,
        batch_size=64,  # Larger batch for stability
        lr=0.001,
        patience=80,  # More patience
        device=DEVICE,
        augment=True,
        num_augmentations=4
    )
    
    # Save training history plot
    plot_training_history(history, OUTPUT_DIR / 'training_history.png')
    print(f"\nTraining plot saved to: {OUTPUT_DIR / 'training_history.png'}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_dataset = MolecularDataset(test_data, model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    y_pred, y_true = get_predictions(model.network.to(DEVICE), test_loader, DEVICE)
    metrics = calculate_metrics(y_true, y_pred, model)
    
    print_metrics(metrics, "TEST SET METRICS")
    
    # Save prediction plots
    plot_predictions(y_true, y_pred, model, OUTPUT_DIR / 'predictions.png')
    plot_error_distribution(y_true, y_pred, model, OUTPUT_DIR / 'error_distribution.png')
    print(f"Plots saved to: {OUTPUT_DIR}")
    
    # Save model
    save_model(model, OUTPUT_DIR / 'physics_model.pt')
    print(f"\nModel saved to: {OUTPUT_DIR / 'physics_model.pt'}")
    
    # Print sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (First 10 test samples)")
    print("="*60)
    print(f"{'True X0':>10} {'True Y0':>10} {'Pred X0':>10} {'Pred Y0':>10} {'Error':>10}")
    print("-"*60)
    
    true_denorm = np.array([model.denormalize_target(y) for y in y_true[:10]])
    pred_denorm = np.array([model.denormalize_target(y) for y in y_pred[:10]])
    
    for i in range(10):
        error = np.sqrt((true_denorm[i, 0] - pred_denorm[i, 0])**2 + 
                        (true_denorm[i, 1] - pred_denorm[i, 1])**2)
        print(f"{true_denorm[i, 0]:10.2f} {true_denorm[i, 1]:10.2f} "
              f"{pred_denorm[i, 0]:10.2f} {pred_denorm[i, 1]:10.2f} "
              f"{error:10.2f} μm")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
