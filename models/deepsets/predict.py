"""
Prediction script for DeepSets Model.

Usage:
    python -m models.deepsets.predict --num_samples 10
    python -m models.deepsets.predict --evaluate
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np

from utils.data_loader import load_dataset, split_data
from utils.metrics import calculate_position_metrics, print_position_metrics
from models.deepsets.train import load_model


def main():
    parser = argparse.ArgumentParser(description='DeepSets Prediction')
    parser.add_argument('--model', default='outputs/deepsets/model.pt')
    parser.add_argument('--data', default='data/molecular_comm_dataset.mat')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    
    print("="*60)
    print("DEEPSETS MODEL - PREDICTION")
    print("="*60)
    
    model = load_model(args.model)
    model.network.eval()
    
    # Load data
    samples = load_dataset(args.data)
    _, _, test_data = split_data(samples, seed=42)
    
    if args.evaluate:
        # Full evaluation
        predictions = []
        actuals = []
        
        for s in test_data:
            pred = model.predict(s)
            predictions.append(pred)
            actuals.append([s['x0'], s['y0']])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        metrics = calculate_position_metrics(actuals, predictions)
        print_position_metrics(metrics, "TEST SET EVALUATION")
    else:
        # Sample predictions
        np.random.seed(42)
        indices = np.random.choice(len(test_data), min(args.num_samples, len(test_data)), replace=False)
        
        print(f"\n{'Idx':>4} {'True X0':>9} {'True Y0':>9} │ {'Pred X0':>9} {'Pred Y0':>9} │ {'Error':>8} │ {'N0':>5}")
        print("-"*75)
        
        total_error = 0
        for idx in indices:
            s = test_data[idx]
            x0_pred, y0_pred = model.predict(s)
            
            error = np.sqrt((x0_pred - s['x0'])**2 + (y0_pred - s['y0'])**2)
            total_error += error
            
            print(f"{idx:4d} {s['x0']:9.2f} {s['y0']:9.2f} │ "
                  f"{x0_pred:9.2f} {y0_pred:9.2f} │ {error:7.2f}μm │ {len(s['absorption_times']):5d}")
        
        print("-"*75)
        print(f"Mean Error: {total_error/len(indices):.2f} μm")


if __name__ == '__main__':
    main()
