"""
Model Comparison Script

Compares all 4 models on the same test set:
1. Feature MLP: Hand-crafted features → (x0, y0)
2. Distance MLP: Hand-crafted features → distance only
3. CNN: Heatmap image → (x0, y0)
4. DeepSets: Raw (time, angle) data → (x0, y0)

Usage:
    python analysis/compare_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.data_loader import load_dataset, split_data
from utils.metrics import calculate_position_metrics, calculate_distance_metrics


def load_all_models():
    """Load all trained models."""
    models = {}
    
    # Feature MLP
    try:
        from models.feature_mlp.train import load_model as load_feature_mlp
        models['feature_mlp'] = load_feature_mlp('outputs/feature_mlp/model.pt')
        models['feature_mlp'].network.eval()
        print("✓ Loaded Feature MLP")
    except Exception as e:
        print(f"✗ Could not load Feature MLP: {e}")
    
    # Distance MLP
    try:
        from models.distance_mlp.train import load_model as load_distance_mlp
        models['distance_mlp'] = load_distance_mlp('outputs/distance_mlp/model.pt')
        models['distance_mlp'].network.eval()
        print("✓ Loaded Distance MLP")
    except Exception as e:
        print(f"✗ Could not load Distance MLP: {e}")
    
    # CNN
    try:
        from models.cnn.train import load_model as load_cnn
        model, norm_params = load_cnn('outputs/cnn/model.pt')
        model.eval()
        models['cnn'] = {'model': model, 'norm': norm_params}
        print("✓ Loaded CNN")
    except Exception as e:
        print(f"✗ Could not load CNN: {e}")
    
    # DeepSets
    try:
        from models.deepsets.train import load_model as load_deepsets
        models['deepsets'] = load_deepsets('outputs/deepsets/model.pt')
        models['deepsets'].network.eval()
        print("✓ Loaded DeepSets")
    except Exception as e:
        print(f"✗ Could not load DeepSets: {e}")
    
    return models


def evaluate_models(models, test_samples, test_samples_with_heatmaps=None):
    """Evaluate all models on test set."""
    results = {}
    
    # Feature MLP
    if 'feature_mlp' in models:
        model = models['feature_mlp']
        preds, actuals = [], []
        for s in test_samples:
            pred = model.predict(s)
            preds.append(pred)
            actuals.append([s['x0'], s['y0']])
        preds = np.array(preds)
        actuals = np.array(actuals)
        results['feature_mlp'] = {
            'metrics': calculate_position_metrics(actuals, preds),
            'preds': preds,
            'actuals': actuals
        }
    
    # Distance MLP
    if 'distance_mlp' in models:
        model = models['distance_mlp']
        preds, actuals = [], []
        for s in test_samples:
            pred = model.predict(s)
            preds.append(pred)
            actuals.append(s['distance'])
        preds = np.array(preds)
        actuals = np.array(actuals)
        results['distance_mlp'] = {
            'metrics': calculate_distance_metrics(actuals, preds),
            'preds': preds,
            'actuals': actuals
        }
    
    # CNN
    if 'cnn' in models and test_samples_with_heatmaps is not None:
        from models.cnn.train import HeatmapDataset
        from torch.utils.data import DataLoader
        
        model_info = models['cnn']
        model = model_info['model']
        norm = model_info['norm']
        
        dataset = HeatmapDataset(
            test_samples_with_heatmaps,
            log_transform=norm['log_transform'],
            mean=norm['heatmap_mean'],
            std=norm['heatmap_std']
        )
        loader = DataLoader(dataset, batch_size=64)
        
        all_preds, all_actuals = [], []
        with torch.no_grad():
            for heatmaps, targets in loader:
                outputs = model(heatmaps)
                preds_um = dataset.denormalize_target(outputs)
                targets_um = dataset.denormalize_target(targets)
                all_preds.append(preds_um.numpy())
                all_actuals.append(targets_um.numpy())
        
        preds = np.vstack(all_preds)
        actuals = np.vstack(all_actuals)
        results['cnn'] = {
            'metrics': calculate_position_metrics(actuals, preds),
            'preds': preds,
            'actuals': actuals
        }
    
    # DeepSets
    if 'deepsets' in models:
        model = models['deepsets']
        preds, actuals = [], []
        for s in test_samples:
            pred = model.predict(s)
            preds.append(pred)
            actuals.append([s['x0'], s['y0']])
        preds = np.array(preds)
        actuals = np.array(actuals)
        results['deepsets'] = {
            'metrics': calculate_position_metrics(actuals, preds),
            'preds': preds,
            'actuals': actuals
        }
    
    return results


def print_comparison_table(results):
    """Print comparison table of all models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON - Position Prediction (x0, y0)")
    print("="*80)
    
    position_models = ['feature_mlp', 'cnn', 'deepsets']
    available = [m for m in position_models if m in results]
    
    if available:
        header = f"{'Metric':<25}" + "".join([f"{m:>15}" for m in available])
        print(header)
        print("-"*80)
        
        metrics_to_show = [
            ('MAE X (μm)', 'x_mae'),
            ('MAE Y (μm)', 'y_mae'),
            ('Mean Dist Error (μm)', 'dist_mean'),
            ('Median Dist Error (μm)', 'dist_median'),
            ('P90 Dist Error (μm)', 'dist_p90'),
            ('Within 5 μm (%)', 'within_5um'),
            ('Within 10 μm (%)', 'within_10um'),
        ]
        
        for label, key in metrics_to_show:
            row = f"{label:<25}"
            for m in available:
                val = results[m]['metrics'].get(key, 0)
                row += f"{val:>15.2f}"
            print(row)
    
    # Distance model
    if 'distance_mlp' in results:
        print("\n" + "-"*80)
        print("Distance MLP (distance-only prediction):")
        dm = results['distance_mlp']['metrics']
        print(f"  MAE: {dm['mae']:.2f} μm, RMSE: {dm['rmse']:.2f} μm, Within 5μm: {dm['within_5um']:.1f}%")
    
    print("="*80)


def plot_comparison(results, output_path='analysis/model_comparison.png'):
    """Create comparison plots."""
    position_models = ['feature_mlp', 'cnn', 'deepsets']
    available = [m for m in position_models if m in results]
    
    if not available:
        print("No position models available for plotting")
        return
    
    fig, axes = plt.subplots(2, len(available), figsize=(5*len(available), 8))
    if len(available) == 1:
        axes = axes.reshape(-1, 1)
    
    model_names = {
        'feature_mlp': 'Feature MLP',
        'cnn': 'CNN',
        'deepsets': 'DeepSets'
    }
    
    for i, model_key in enumerate(available):
        res = results[model_key]
        preds = res['preds']
        actuals = res['actuals']
        metrics = res['metrics']
        
        # Scatter plot
        ax = axes[0, i]
        ax.scatter(actuals[:, 0], preds[:, 0], alpha=0.3, s=10, label='x0')
        ax.scatter(actuals[:, 1], preds[:, 1], alpha=0.3, s=10, label='y0')
        ax.plot([15, 90], [15, 90], 'k--', lw=2, label='Perfect')
        ax.set_xlabel('True (μm)')
        ax.set_ylabel('Predicted (μm)')
        ax.set_title(f"{model_names[model_key]}\nMAE: {metrics['dist_mean']:.2f}μm")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Error histogram
        ax = axes[1, i]
        errors = np.sqrt((preds[:, 0] - actuals[:, 0])**2 + (preds[:, 1] - actuals[:, 1])**2)
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(errors), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(errors):.2f}μm')
        ax.axvline(np.median(errors), color='g', linestyle='--',
                   label=f'Median: {np.median(errors):.2f}μm')
        ax.set_xlabel('Euclidean Error (μm)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved to: {output_path}")


def main():
    print("="*80)
    print("MODEL COMPARISON ANALYSIS")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    models = load_all_models()
    
    if not models:
        print("\nNo models found. Please train models first:")
        print("  python -m models.feature_mlp.train")
        print("  python -m models.distance_mlp.train")
        print("  python -m models.cnn.train")
        print("  python -m models.deepsets.train")
        return
    
    # Load test data
    print("\nLoading test data...")
    
    # Check if CNN needs heatmaps
    needs_heatmaps = 'cnn' in models
    samples = load_dataset('data/molecular_comm_dataset.mat', load_heatmaps=needs_heatmaps)
    _, _, test_samples = split_data(samples, seed=42)
    print(f"Test samples: {len(test_samples)}")
    
    # Evaluate all models
    print("\nEvaluating models...")
    results = evaluate_models(
        models, 
        test_samples,
        test_samples_with_heatmaps=test_samples if needs_heatmaps else None
    )
    
    # Print comparison
    print_comparison_table(results)
    
    # Create plots
    plot_comparison(results)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
