"""
Prediction script for CNN Model.

Usage:
    python -m models.cnn.predict --num_samples 10
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import torch

from utils.data_loader import load_dataset, split_data
from utils.metrics import calculate_position_metrics, print_position_metrics
from models.cnn.train import load_model, HeatmapDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='CNN Prediction')
    parser.add_argument('--model', default='outputs/cnn/model.pt')
    parser.add_argument('--data', default='data/molecular_comm_dataset.mat')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("CNN MODEL - PREDICTION")
    print("="*60)
    
    # Load model
    model, norm_params = load_model(args.model, device)
    model.eval()
    
    # Load data
    samples = load_dataset(args.data, load_heatmaps=True)
    _, _, test_samples = split_data(samples, seed=42)
    
    # Create dataset with saved normalization
    test_dataset = HeatmapDataset(
        test_samples, 
        log_transform=norm_params['log_transform'],
        mean=norm_params['heatmap_mean'],
        std=norm_params['heatmap_std']
    )
    
    if args.evaluate:
        # Full evaluation
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        all_preds, all_targets = [], []
        with torch.no_grad():
            for heatmaps, targets in test_loader:
                heatmaps = heatmaps.to(device)
                outputs = model(heatmaps)
                preds_um = test_dataset.denormalize_target(outputs.cpu())
                targets_um = test_dataset.denormalize_target(targets)
                all_preds.append(preds_um)
                all_targets.append(targets_um)
        
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        metrics = calculate_position_metrics(targets, preds)
        print_position_metrics(metrics, "TEST SET EVALUATION")
    else:
        # Sample predictions
        np.random.seed(42)
        indices = np.random.choice(len(test_dataset), min(args.num_samples, len(test_dataset)), replace=False)
        
        print(f"\n{'Idx':>4} {'True X0':>9} {'True Y0':>9} │ {'Pred X0':>9} {'Pred Y0':>9} │ {'Error':>8}")
        print("-"*70)
        
        total_error = 0
        with torch.no_grad():
            for idx in indices:
                heatmap, target = test_dataset[idx]
                heatmap = heatmap.unsqueeze(0).to(device)
                
                output = model(heatmap).cpu()
                pred_um = test_dataset.denormalize_target(output[0].numpy())
                target_um = test_dataset.denormalize_target(target.numpy())
                
                error = np.sqrt((pred_um[0] - target_um[0])**2 + (pred_um[1] - target_um[1])**2)
                total_error += error
                
                print(f"{idx:4d} {target_um[0]:9.2f} {target_um[1]:9.2f} │ "
                      f"{pred_um[0]:9.2f} {pred_um[1]:9.2f} │ {error:7.2f}μm")
        
        print("-"*70)
        print(f"Mean Error: {total_error/len(indices):.2f} μm")


if __name__ == '__main__':
    main()
