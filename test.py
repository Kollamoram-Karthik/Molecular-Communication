"""
Test script for evaluating trained models on the test set.

Usage:
    python test.py --data molecular_comm_dataset.mat --model outputs/deepsets_model.pt
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data import load_matlab_dataset, MolecularDataPreprocessor, MolecularCommDataset, collate_variable_length
from src.models import DeepSets
from src.models.baseline import load_baseline, extract_features
from src.training.metrics import calculate_metrics, print_metrics
from src.utils.visualization import plot_predictions, plot_error_distribution


def parse_args():
    parser = argparse.ArgumentParser(description='Test transmitter localization model')
    parser.add_argument('--data', type=str, default='molecular_comm_dataset.mat',
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='outputs/deepsets_model.pt',
                        help='Path to trained DeepSets model')
    parser.add_argument('--preprocessor', type=str, default='outputs/preprocessor.pkl',
                        help='Path to preprocessor')
    parser.add_argument('--baseline', type=str, default='outputs/baseline_model.pkl',
                        help='Path to baseline model (optional)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for plots')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    print('Loading dataset...')
    samples = load_matlab_dataset(args.data)
    
    # Split data (same split as training)
    train_samples, temp_samples = train_test_split(
        samples, test_size=0.3, random_state=args.seed
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, random_state=args.seed
    )
    
    print(f'Test set: {len(test_samples)} samples')
    
    # Load preprocessor
    print('Loading preprocessor...')
    preprocessor = MolecularDataPreprocessor.load(args.preprocessor)
    
    # Create test dataset and loader
    test_dataset = MolecularCommDataset(test_samples, preprocessor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=0
    )
    
    # =========================================================================
    # Evaluate DeepSets Model
    # =========================================================================
    print('\n' + '='*50)
    print('Evaluating DeepSets Model')
    print('='*50)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepSets(
        input_dim=3,
        phi_hidden=[64, 128],
        rho_hidden=[128, 64],
        output_dim=2,
        dropout=0.3
    )
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Loaded model from {args.model}')
    print(f'Best validation loss: {checkpoint["best_val_loss"]:.4f}')
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            
            preds = model(features, mask)
            all_preds.append(preds.cpu())
            all_targets.append(torch.stack([batch['x0_orig'], batch['y0_orig']], dim=1))
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Convert predictions back to original scale
    all_preds_orig = preprocessor.inverse_transform_target(all_preds)
    
    # Calculate and print metrics
    metrics = calculate_metrics(all_targets, all_preds_orig)
    print_metrics(metrics, 'DeepSets Test Metrics')
    
    # Plot predictions
    plot_predictions(
        all_targets,
        all_preds_orig,
        title='DeepSets - Test Set Predictions',
        save_path=os.path.join(args.output_dir, 'test_predictions_deepsets.png')
    )
    
    # Plot error distribution
    plot_error_distribution(
        all_targets,
        all_preds_orig,
        save_path=os.path.join(args.output_dir, 'error_distribution_deepsets.png')
    )
    
    # =========================================================================
    # Evaluate Baseline Model (if available)
    # =========================================================================
    if os.path.exists(args.baseline):
        print('\n' + '='*50)
        print('Evaluating Baseline Model')
        print('='*50)
        
        baseline_model, baseline_scaler = load_baseline(args.baseline)
        
        X_test = extract_features(test_samples)
        X_test_scaled = baseline_scaler.transform(X_test)
        
        baseline_preds = baseline_model.predict(X_test_scaled)
        
        baseline_metrics = calculate_metrics(all_targets, baseline_preds)
        print_metrics(baseline_metrics, 'Baseline Test Metrics')
        
        plot_predictions(
            all_targets,
            baseline_preds,
            title='Baseline (Gradient Boosting) - Test Set Predictions',
            save_path=os.path.join(args.output_dir, 'test_predictions_baseline.png')
        )
    
    # =========================================================================
    # Print Comparison
    # =========================================================================
    print('\n' + '='*50)
    print('Model Comparison Summary')
    print('='*50)
    print(f'DeepSets Mean Distance Error: {metrics["mean_distance_error"]:.2f} μm')
    if os.path.exists(args.baseline):
        print(f'Baseline Mean Distance Error: {baseline_metrics["mean_distance_error"]:.2f} μm')
    
    print('\nTest complete!')


if __name__ == '__main__':
    main()
