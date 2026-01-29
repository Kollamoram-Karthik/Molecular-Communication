"""
Prediction script for single sample inference.

Usage:
    python predict.py --model outputs/deepsets_model.pt --sample_idx 0
"""

import argparse
import torch
import numpy as np

from src.data import load_matlab_dataset, MolecularDataPreprocessor
from src.models import DeepSets


def parse_args():
    parser = argparse.ArgumentParser(description='Predict transmitter location')
    parser.add_argument('--data', type=str, default='molecular_comm_dataset.mat',
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='outputs/deepsets_model.pt',
                        help='Path to trained model')
    parser.add_argument('--preprocessor', type=str, default='outputs/preprocessor.pkl',
                        help='Path to preprocessor')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to predict')
    return parser.parse_args()


def predict_single(model, preprocessor, absorption_times, impact_angles, device='cpu'):
    """
    Predict transmitter location from absorption data.
    
    Args:
        model: Trained DeepSets model
        preprocessor: Fitted preprocessor
        absorption_times: Array of absorption times
        impact_angles: Array of impact angles
        device: Device to run inference on
        
    Returns:
        Tuple of (x0, y0) predicted coordinates
    """
    model.eval()
    
    # Preprocess input
    features = preprocessor.transform_input(
        np.array(absorption_times),
        np.array(impact_angles)
    )
    
    # Convert to tensors
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, N0, 3)
    mask = torch.ones(1, features.shape[0], dtype=torch.bool).to(device)  # (1, N0)
    
    # Predict
    with torch.no_grad():
        pred = model(features_tensor, mask)
    
    # Convert to original scale
    pred_np = pred.cpu().numpy()[0]
    x0, y0 = preprocessor.inverse_transform_target(pred_np)
    
    return x0, y0


def main():
    args = parse_args()
    
    # Load data
    samples = load_matlab_dataset(args.data)
    sample = samples[args.sample_idx]
    
    # Load preprocessor
    preprocessor = MolecularDataPreprocessor.load(args.preprocessor)
    
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
    
    # Predict
    x0_pred, y0_pred = predict_single(
        model, 
        preprocessor,
        sample['absorption_times'],
        sample['impact_angles'],
        device
    )
    
    # Print results
    print('\n' + '='*50)
    print(f'Sample {args.sample_idx}')
    print('='*50)
    print(f'Number of absorbed molecules: {sample["N0"]}')
    print(f'\nTrue Position:')
    print(f'  x0 = {sample["x0"]:.2f} μm')
    print(f'  y0 = {sample["y0"]:.2f} μm')
    print(f'  Distance from origin = {sample["distance"]:.2f} μm')
    print(f'\nPredicted Position:')
    print(f'  x0 = {x0_pred:.2f} μm')
    print(f'  y0 = {y0_pred:.2f} μm')
    pred_dist = np.sqrt(x0_pred**2 + y0_pred**2)
    print(f'  Distance from origin = {pred_dist:.2f} μm')
    print(f'\nError:')
    error = np.sqrt((sample["x0"] - x0_pred)**2 + (sample["y0"] - y0_pred)**2)
    print(f'  Position error = {error:.2f} μm')
    print('='*50)


if __name__ == '__main__':
    main()
