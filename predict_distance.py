"""
Evaluation and Prediction script for the Distance Model.
"""

import torch
import numpy as np
import argparse

from ml_model.data_loader import load_dataset
from train_distance import load_distance_model


def evaluate_model(model_path='outputs/distance_model.pt', data_path='molecular_comm_dataset.mat'):
    """Evaluate distance model on full dataset"""
    
    print("=" * 60)
    print("Distance Model Evaluation")
    print("=" * 60)
    
    # Load model and data
    model = load_distance_model(model_path)
    model.network.eval()
    
    samples = load_dataset(data_path)
    print(f"\nTotal samples: {len(samples)}")
    
    # Predict on all samples
    predictions = []
    physics_estimates = []
    actuals = []
    
    with torch.no_grad():
        for s in samples:
            features = model.extract_features(s)
            X = torch.FloatTensor(features).unsqueeze(0)
            pred_norm = model.network(X).numpy()[0]
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
    
    # Results
    print("\n" + "=" * 60)
    print("Neural Network Performance")
    print("=" * 60)
    print(f"Mean error:   {nn_errors.mean():.2f} μm")
    print(f"Median error: {np.median(nn_errors):.2f} μm")
    print(f"Std error:    {nn_errors.std():.2f} μm")
    print(f"Max error:    {nn_errors.max():.2f} μm")
    print(f"\nWithin 2 μm:  {100 * np.mean(nn_errors < 2):.1f}%")
    print(f"Within 5 μm:  {100 * np.mean(nn_errors < 5):.1f}%")
    print(f"Within 10 μm: {100 * np.mean(nn_errors < 10):.1f}%")
    
    print("\n" + "=" * 60)
    print("Physics Baseline")
    print("=" * 60)
    print(f"Mean error:   {physics_errors.mean():.2f} μm")
    print(f"Median error: {np.median(physics_errors):.2f} μm")
    
    improvement = (physics_errors.mean() - nn_errors.mean()) / physics_errors.mean() * 100
    print(f"\nNN improvement over physics: {improvement:.1f}%")
    
    # Error by distance bins
    print("\n" + "=" * 60)
    print("Error by True Distance Range")
    print("=" * 60)
    
    bins = [(20, 40), (40, 60), (60, 80), (80, 100), (100, 130)]
    for lo, hi in bins:
        mask = (actuals >= lo) & (actuals < hi)
        if mask.sum() > 0:
            bin_mean = nn_errors[mask].mean()
            bin_count = mask.sum()
            print(f"  {lo:3d}-{hi:3d} μm: {bin_mean:.2f} μm error ({bin_count} samples)")


def predict_distance(model_path='outputs/distance_model.pt', 
                     data_path='molecular_comm_dataset.mat',
                     sample_idx=None, num_samples=5):
    """Make predictions on specific samples"""
    
    model = load_distance_model(model_path)
    model.network.eval()
    
    samples = load_dataset(data_path)
    
    if sample_idx is not None:
        indices = [sample_idx]
    else:
        indices = np.random.choice(len(samples), size=min(num_samples, len(samples)), replace=False)
    
    print("\n" + "=" * 60)
    print("Distance Predictions")
    print("=" * 60)
    print(f"{'Sample':>8} {'Actual':>10} {'Predicted':>10} {'Physics':>10} {'Error':>10}")
    print("-" * 60)
    
    errors = []
    with torch.no_grad():
        for idx in indices:
            s = samples[idx]
            features = model.extract_features(s)
            X = torch.FloatTensor(features).unsqueeze(0)
            pred_norm = model.network(X).numpy()[0]
            pred_dist = model.denormalize_target(pred_norm)
            physics_est = model.get_physics_estimate(s)
            
            actual = s['distance']
            error = abs(pred_dist - actual)
            errors.append(error)
            
            print(f"{idx:>8} {actual:>10.2f} {pred_dist:>10.2f} {physics_est:>10.2f} {error:>10.2f}")
    
    print("-" * 60)
    print(f"{'Mean':>8} {'':>10} {'':>10} {'':>10} {np.mean(errors):>10.2f}")


def predict_from_times(model_path='outputs/distance_model.pt', times=None):
    """Predict distance from custom absorption times"""
    
    if times is None or len(times) == 0:
        print("Error: Must provide absorption times")
        return
    
    model = load_distance_model(model_path)
    model.network.eval()
    
    # Create sample dict
    sample = {'absorption_times': np.array(times)}
    
    features = model.extract_features(sample)
    X = torch.FloatTensor(features).unsqueeze(0)
    
    with torch.no_grad():
        pred_norm = model.network(X).numpy()[0]
        pred_dist = model.denormalize_target(pred_norm)
    
    physics_est = model.get_physics_estimate(sample)
    
    print("\n" + "=" * 60)
    print("Custom Prediction")
    print("=" * 60)
    print(f"Number of absorbed molecules: {len(times)}")
    print(f"Mean absorption time: {np.mean(times):.2f}s")
    print(f"Median absorption time: {np.median(times):.2f}s")
    print(f"\nPhysics estimate: {physics_est:.2f} μm")
    print(f"NN prediction:    {pred_dist:.2f} μm")
    
    return pred_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distance Model Prediction')
    parser.add_argument('--model', default='outputs/distance_model.pt', help='Model path')
    parser.add_argument('--data', default='molecular_comm_dataset.mat', help='Dataset path')
    parser.add_argument('--evaluate', action='store_true', help='Run full evaluation')
    parser.add_argument('--sample_idx', type=int, default=None, help='Specific sample index')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples')
    parser.add_argument('--times', type=float, nargs='+', help='Custom absorption times')
    
    args = parser.parse_args()
    
    if args.times:
        predict_from_times(args.model, args.times)
    elif args.evaluate:
        evaluate_model(args.model, args.data)
    else:
        predict_distance(args.model, args.data, args.sample_idx, args.num_samples)
