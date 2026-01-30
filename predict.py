#!/usr/bin/env python3
"""
Predict transmitter location from molecular absorption data.

Usage:
    # Predict for a specific sample from the dataset
    python predict.py --sample_idx 0
    
    # Predict for multiple random samples
    python predict.py --num_samples 10
    
    # Predict from custom data (provide times and angles as comma-separated values)
    python predict.py --times "1.5,2.3,3.1,4.0" --angles "0.5,-0.3,1.2,-1.0"
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml_model.data_loader import load_dataset, split_data
from ml_model.trainer import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Predict transmitter location')
    parser.add_argument('--model', type=str, default='outputs/physics_model.pt',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, default='molecular_comm_dataset.mat',
                        help='Path to dataset')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Specific sample index to predict')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of random samples to predict')
    parser.add_argument('--times', type=str, default=None,
                        help='Comma-separated absorption times for custom prediction')
    parser.add_argument('--angles', type=str, default=None,
                        help='Comma-separated impact angles for custom prediction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    return parser.parse_args()


def predict_single(model, absorption_times, impact_angles):
    """
    Predict transmitter location from absorption data.
    
    Args:
        model: Trained PhysicsInformedModel
        absorption_times: Array of absorption times
        impact_angles: Array of impact angles
        
    Returns:
        Tuple of (x0, y0) predicted coordinates
    """
    model.network.eval()
    
    # Extract features using the physics-based extractor
    features = model.feature_extractor.extract_features(
        np.array(absorption_times),
        np.array(impact_angles)
    )
    
    # Convert to tensor and predict
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        pred_normalized = model.network(features_tensor).numpy()[0]
    
    # Denormalize
    x0, y0 = model.denormalize_target(pred_normalized)
    
    return x0, y0


def predict_from_dataset(model, samples, indices):
    """Predict for multiple samples from dataset"""
    results = []
    
    for idx in indices:
        sample = samples[idx]
        
        # Get prediction
        x0_pred, y0_pred = predict_single(
            model,
            sample['absorption_times'],
            sample['impact_angles']
        )
        
        # Get physics-only estimate for comparison
        x0_physics, y0_physics = model.get_physics_estimate(sample)
        
        # Calculate errors
        x0_true, y0_true = sample['x0'], sample['y0']
        error_model = np.sqrt((x0_pred - x0_true)**2 + (y0_pred - y0_true)**2)
        error_physics = np.sqrt((x0_physics - x0_true)**2 + (y0_physics - y0_true)**2)
        
        results.append({
            'idx': idx,
            'x0_true': x0_true,
            'y0_true': y0_true,
            'x0_pred': x0_pred,
            'y0_pred': y0_pred,
            'x0_physics': x0_physics,
            'y0_physics': y0_physics,
            'error_model': error_model,
            'error_physics': error_physics,
            'n_absorbed': len(sample['absorption_times'])
        })
    
    return results


def main():
    args = parse_args()
    
    print("="*70)
    print("TRANSMITTER LOCALIZATION - PREDICTION")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model)
    model.network.eval()
    print("Model loaded successfully!")
    
    # Custom data prediction
    if args.times is not None and args.angles is not None:
        print("\n" + "="*70)
        print("CUSTOM DATA PREDICTION")
        print("="*70)
        
        times = np.array([float(t.strip()) for t in args.times.split(',')])
        angles = np.array([float(a.strip()) for a in args.angles.split(',')])
        
        print(f"\nInput:")
        print(f"  Absorption times: {times}")
        print(f"  Impact angles: {angles}")
        print(f"  N molecules absorbed: {len(times)}")
        
        x0_pred, y0_pred = predict_single(model, times, angles)
        
        print(f"\nPredicted Transmitter Location:")
        print(f"  X0 = {x0_pred:.2f} μm")
        print(f"  Y0 = {y0_pred:.2f} μm")
        print(f"  Distance from origin = {np.sqrt(x0_pred**2 + y0_pred**2):.2f} μm")
        print("="*70)
        return
    
    # Load dataset for sample prediction
    print(f"Loading dataset from: {args.data}")
    samples = load_dataset(args.data)
    _, _, test_data = split_data(samples, seed=42)
    print(f"Test samples available: {len(test_data)}")
    
    # Determine which samples to predict
    if args.sample_idx is not None:
        indices = [args.sample_idx]
    else:
        np.random.seed(args.seed)
        indices = np.random.choice(len(test_data), min(args.num_samples, len(test_data)), replace=False)
    
    # Get predictions
    results = predict_from_dataset(model, test_data, indices)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    print(f"\n{'Idx':>4} {'True X0':>9} {'True Y0':>9} │ {'Pred X0':>9} {'Pred Y0':>9} │ {'Error':>8} │ {'Physics':>8} │ {'N0':>4}")
    print("-"*70)
    
    total_error_model = 0
    total_error_physics = 0
    
    for r in results:
        print(f"{r['idx']:4d} {r['x0_true']:9.2f} {r['y0_true']:9.2f} │ "
              f"{r['x0_pred']:9.2f} {r['y0_pred']:9.2f} │ "
              f"{r['error_model']:7.2f}μm │ "
              f"{r['error_physics']:7.2f}μm │ "
              f"{r['n_absorbed']:4d}")
        total_error_model += r['error_model']
        total_error_physics += r['error_physics']
    
    print("-"*70)
    n = len(results)
    print(f"{'MEAN':>4} {' ':>9} {' ':>9} │ {' ':>9} {' ':>9} │ "
          f"{total_error_model/n:7.2f}μm │ {total_error_physics/n:7.2f}μm │")
    print("="*70)
    
    print(f"\nModel improvement over physics baseline: "
          f"{(total_error_physics - total_error_model) / total_error_physics * 100:.1f}%")


if __name__ == '__main__':
    main()
