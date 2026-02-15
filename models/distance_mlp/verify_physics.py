"""
Verify physics polynomial by manually applying it to test samples.

Usage:
    python -m models.distance_mlp.verify_physics --num_samples 10
"""

import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.data_loader import load_dataset, split_data


def apply_physics_polynomial(median_time, coeffs):
    """
    Apply polynomial: d = a·√t² + b·√t + c
    
    Args:
        median_time: Median absorption time
        coeffs: [a, b, c] polynomial coefficients
    
    Returns:
        Estimated distance in μm
    """
    sqrt_t = np.sqrt(median_time)
    distance = coeffs[0] * (sqrt_t ** 2) + coeffs[1] * sqrt_t + coeffs[2]
    return np.clip(distance, 15, 130)


def main():
    parser = argparse.ArgumentParser(description='Verify physics polynomial')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of test samples to verify')
    parser.add_argument('--data_path', type=str, default='data/molecular_comm_dataset.mat',
                        help='Path to dataset')
    args = parser.parse_args()
    
    # These are the coefficients from the trained model
    # d = 2.2309·√t² + -0.2570·√t + 20.6909
    coeffs = [2.2309, -0.2570, 20.6909]
    
    print("PHYSICS POLYNOMIAL VERIFICATION")
    print("=" * 70)
    print(f"Polynomial: d = {coeffs[0]:.4f}·√t² + {coeffs[1]:.4f}·√t + {coeffs[2]:.4f}")
    print("=" * 70)
    
    # Load dataset and get test samples
    print(f"\nLoading data from: {args.data_path}")
    samples = load_dataset(args.data_path)
    
    # Use same split as predict.py (70/15/15 with seed=42)
    _, _, test_samples = split_data(samples, seed=42)
    
    print(f"Total test samples: {len(test_samples)}")
    
    # Select random samples to verify
    np.random.seed(42)
    selected_indices = np.random.choice(len(test_samples), 
                                       size=min(args.num_samples, len(test_samples)), 
                                       replace=False)
    
    print(f"\n{'Sample':>8} {'Actual':>10} {'Median_t':>10} {'√Median_t':>10} {'Physics_Est':>12} {'Error':>10}")
    print("-" * 70)
    
    errors = []
    for idx in sorted(selected_indices):
        sample = test_samples[idx]
        actual_distance = sample['distance']
        absorption_times = sample['absorption_times']
        
        if len(absorption_times) == 0:
            print(f"{idx:8d} {actual_distance:10.2f} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
            continue
        
        median_t = np.median(absorption_times)
        sqrt_median_t = np.sqrt(median_t)
        physics_estimate = apply_physics_polynomial(median_t, coeffs)
        error = abs(actual_distance - physics_estimate)
        
        errors.append(error)
        
        print(f"{idx:8d} {actual_distance:10.2f} {median_t:10.2f} {sqrt_median_t:10.4f} "
              f"{physics_estimate:12.2f} {error:10.2f}")
    
    print("-" * 70)
    if errors:
        print(f"{'Mean':>8} {' '*10} {' '*10} {' '*10} {' '*12} {np.mean(errors):10.2f}")
        print(f"{'Std':>8} {' '*10} {' '*10} {' '*10} {' '*12} {np.std(errors):10.2f}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION DETAILS")
    print("=" * 70)
    print("\nThe polynomial maps √(median absorption time) to distance:")
    print(f"  d = {coeffs[0]:.4f} * (√t)² + {coeffs[1]:.4f} * √t + {coeffs[2]:.4f}")
    print(f"  d = {coeffs[0]:.4f} * t + {coeffs[1]:.4f} * √t + {coeffs[2]:.4f}")
    print("\nNote: Since (√t)² = t, the first term is linear in t.")
    print("This represents a combination of linear and square-root relationships.")


if __name__ == '__main__':
    main()
