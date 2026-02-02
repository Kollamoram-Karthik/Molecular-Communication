"""
Shared evaluation metrics for all models.

This module provides standardized metric calculation
for comparing different model architectures.
"""

import numpy as np


def calculate_position_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for (x0, y0) prediction.
    
    Args:
        y_true: np.array of shape (N, 2) - true [x0, y0] in μm
        y_pred: np.array of shape (N, 2) - predicted [x0, y0] in μm
    
    Returns:
        dict of metrics
    """
    # Per-coordinate errors
    x_errors = y_true[:, 0] - y_pred[:, 0]
    y_errors = y_true[:, 1] - y_pred[:, 1]
    
    x_rmse = np.sqrt(np.mean(x_errors**2))
    y_rmse = np.sqrt(np.mean(y_errors**2))
    x_mae = np.mean(np.abs(x_errors))
    y_mae = np.mean(np.abs(y_errors))
    
    # Euclidean distance error
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
        'within_10um': np.mean(dist_errors < 10) * 100,
    }
    
    return metrics


def calculate_distance_metrics(d_true, d_pred):
    """
    Calculate metrics for distance-only prediction.
    
    Args:
        d_true: np.array of shape (N,) - true distances in μm
        d_pred: np.array of shape (N,) - predicted distances in μm
    
    Returns:
        dict of metrics
    """
    errors = np.abs(d_true - d_pred)
    
    metrics = {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'median': np.median(errors),
        'std': np.std(errors),
        'max': np.max(errors),
        'p90': np.percentile(errors, 90),
        'p95': np.percentile(errors, 95),
        'within_2um': np.mean(errors < 2) * 100,
        'within_5um': np.mean(errors < 5) * 100,
        'within_10um': np.mean(errors < 10) * 100,
    }
    
    return metrics


def print_position_metrics(metrics, title="Position Prediction Metrics"):
    """Pretty print position metrics"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(f"X Coordinate:")
    print(f"  RMSE: {metrics['x_rmse']:.3f} μm")
    print(f"  MAE:  {metrics['x_mae']:.3f} μm")
    print(f"Y Coordinate:")
    print(f"  RMSE: {metrics['y_rmse']:.3f} μm")
    print(f"  MAE:  {metrics['y_mae']:.3f} μm")
    print(f"Euclidean Distance Error:")
    print(f"  Mean:   {metrics['dist_mean']:.3f} μm")
    print(f"  Median: {metrics['dist_median']:.3f} μm")
    print(f"  Std:    {metrics['dist_std']:.3f} μm")
    print(f"  P90:    {metrics['dist_p90']:.3f} μm")
    print(f"  P95:    {metrics['dist_p95']:.3f} μm")
    print(f"Accuracy:")
    print(f"  Within 1 μm:  {metrics['within_1um']:.1f}%")
    print(f"  Within 2 μm:  {metrics['within_2um']:.1f}%")
    print(f"  Within 5 μm:  {metrics['within_5um']:.1f}%")
    print(f"  Within 10 μm: {metrics['within_10um']:.1f}%")
    print('='*60)


def print_distance_metrics(metrics, title="Distance Prediction Metrics"):
    """Pretty print distance metrics"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(f"Error Statistics:")
    print(f"  MAE:    {metrics['mae']:.3f} μm")
    print(f"  RMSE:   {metrics['rmse']:.3f} μm")
    print(f"  Median: {metrics['median']:.3f} μm")
    print(f"  Std:    {metrics['std']:.3f} μm")
    print(f"  P90:    {metrics['p90']:.3f} μm")
    print(f"Accuracy:")
    print(f"  Within 2 μm:  {metrics['within_2um']:.1f}%")
    print(f"  Within 5 μm:  {metrics['within_5um']:.1f}%")
    print(f"  Within 10 μm: {metrics['within_10um']:.1f}%")
    print('='*60)
