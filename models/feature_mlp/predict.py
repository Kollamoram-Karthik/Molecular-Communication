"""
Prediction script for Feature-based MLP Model.

Usage:
    python -m models.feature_mlp.predict --num_samples 10
    python -m models.feature_mlp.predict --times "1.5,2.3,3.1" --angles "0.5,-0.3,1.2"
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np

from utils.data_loader import load_dataset, split_data
from models.feature_mlp.train import load_model


def predict_from_dataset(model, samples, indices):
    """Predict for samples from dataset."""
    results = []
    
    for idx in indices:
        sample = samples[idx]
        x0_pred, y0_pred = model.predict(sample)
        x0_physics, y0_physics = model.get_physics_estimate(sample)
        
        x0_true, y0_true = sample['x0'], sample['y0']
        error_model = np.sqrt((x0_pred - x0_true)**2 + (y0_pred - y0_true)**2)
        error_physics = np.sqrt((x0_physics - x0_true)**2 + (y0_physics - y0_true)**2)
        
        results.append({
            'idx': idx,
            'x0_true': x0_true, 'y0_true': y0_true,
            'x0_pred': x0_pred, 'y0_pred': y0_pred,
            'x0_physics': x0_physics, 'y0_physics': y0_physics,
            'error_model': error_model, 'error_physics': error_physics,
            'n_absorbed': len(sample['absorption_times'])
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Feature MLP Prediction')
    parser.add_argument('--model', default='outputs/feature_mlp/model.pt')
    parser.add_argument('--data', default='data/molecular_comm_dataset.mat')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--times', type=str, default=None)
    parser.add_argument('--angles', type=str, default=None)
    args = parser.parse_args()
    
    print("="*70)
    print("FEATURE MLP - PREDICTION")
    print("="*70)
    
    model = load_model(args.model)
    model.network.eval()
    
    # Custom prediction
    if args.times and args.angles:
        times = np.array([float(t.strip()) for t in args.times.split(',')])
        angles = np.array([float(a.strip()) for a in args.angles.split(',')])
        
        sample = {'absorption_times': times, 'impact_angles': angles}
        x0_pred, y0_pred = model.predict(sample)
        
        print(f"\nCustom Prediction:")
        print(f"  N molecules: {len(times)}")
        print(f"  Predicted: ({x0_pred:.2f}, {y0_pred:.2f}) μm")
        return
    
    # Dataset prediction
    samples = load_dataset(args.data)
    _, _, test_data = split_data(samples, seed=42)
    
    np.random.seed(42)
    indices = np.random.choice(len(test_data), min(args.num_samples, len(test_data)), replace=False)
    
    results = predict_from_dataset(model, test_data, indices)
    
    print(f"\n{'Idx':>4} {'True X0':>9} {'True Y0':>9} │ {'Pred X0':>9} {'Pred Y0':>9} │ {'Error':>8}")
    print("-"*70)
    
    total_error = 0
    for r in results:
        print(f"{r['idx']:4d} {r['x0_true']:9.2f} {r['y0_true']:9.2f} │ "
              f"{r['x0_pred']:9.2f} {r['y0_pred']:9.2f} │ {r['error_model']:7.2f}μm")
        total_error += r['error_model']
    
    print("-"*70)
    print(f"Mean Error: {total_error/len(results):.2f} μm")


if __name__ == '__main__':
    main()
