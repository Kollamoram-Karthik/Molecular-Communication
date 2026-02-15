"""
Prediction script for Distance MLP Model.

Usage:
    python -m models.distance_mlp.predict --evaluate
    python -m models.distance_mlp.predict --num_samples 10
    python -m models.distance_mlp.predict --times 1.5 2.3 3.1 4.0
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import torch

from utils.data_loader import load_dataset, split_data
from utils.metrics import calculate_distance_metrics, print_distance_metrics
from models.distance_mlp.train import load_model


def evaluate(model_path, data_path):
    """Full evaluation on test set."""
    model = load_model(model_path)
    model.network.eval()
    
    samples = load_dataset(data_path)
    _, _, test_data = split_data(samples, seed=42)
    
    predictions = []
    physics_estimates = []
    actuals = []
    
    with torch.no_grad():
        for s in test_data:
            pred = model.predict(s)
            predictions.append(pred)
            physics_estimates.append(model.get_physics_estimate(s))
            actuals.append(s['distance'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    metrics = calculate_distance_metrics(actuals, predictions)
    print_distance_metrics(metrics, "TEST SET EVALUATION")


def predict_samples(model_path, data_path, num_samples):
    """Predict on random samples."""
    model = load_model(model_path)
    model.network.eval()
    
    samples = load_dataset(data_path)
    _, _, test_data = split_data(samples, seed=42)
    
    np.random.seed(42)
    indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    
    print(f"\n{'Sample':>8} {'Actual':>10} {'Predicted':>10} {'Physics':>10} {'NN Error':>10} {'Phys Error':>12}")
    print("-"*72)
    
    errors = []
    physics_errors = []
    for idx in indices:
        s = test_data[idx]
        pred = model.predict(s)
        physics = model.get_physics_estimate(s)
        actual = s['distance']
        error = abs(pred - actual)
        physics_error = abs(physics - actual)
        errors.append(error)
        physics_errors.append(physics_error)
        print(f"{idx:>8} {actual:>10.2f} {pred:>10.2f} {physics:>10.2f} {error:>10.2f} {physics_error:>12.2f}")
    
    print("-"*72)
    print(f"{'Mean':>8} {'':>10} {'':>10} {'':>10} {np.mean(errors):>10.2f} {np.mean(physics_errors):>12.2f}")


def predict_custom(model_path, times):
    """Predict from custom times."""
    model = load_model(model_path)
    
    sample = {'absorption_times': np.array(times)}
    pred = model.predict(sample)
    physics = model.get_physics_estimate(sample)
    
    print(f"\nCustom Prediction:")
    print(f"  N molecules absorbed: {len(times)}")
    print(f"  Mean absorption time: {np.mean(times):.2f}s")
    print(f"  Physics estimate: {physics:.2f} μm")
    print(f"  NN prediction:    {pred:.2f} μm")


def main():
    parser = argparse.ArgumentParser(description='Distance MLP Prediction')
    parser.add_argument('--model', default='outputs/distance_mlp/model.pt')
    parser.add_argument('--data', default='data/molecular_comm_dataset.mat')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--times', type=float, nargs='+')
    args = parser.parse_args()
    
    print("="*60)
    print("DISTANCE MLP - PREDICTION")
    print("="*60)
    
    if args.times:
        predict_custom(args.model, args.times)
    elif args.evaluate:
        evaluate(args.model, args.data)
    else:
        predict_samples(args.model, args.data, args.num_samples)


if __name__ == '__main__':
    main()
