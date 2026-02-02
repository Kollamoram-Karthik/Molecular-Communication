"""
Data loading utilities for molecular communication dataset.

This module provides functions to load the MATLAB-generated dataset
and split it into train/val/test sets.
"""

import numpy as np
import h5py


def load_dataset(mat_path, load_heatmaps=False):
    """
    Load molecular communication dataset from MATLAB .mat file (v7.3 / HDF5 format).
    
    Args:
        mat_path: Path to the .mat file
        load_heatmaps: If True, load heatmap data for CNN training
    
    Returns:
        List of dicts, each containing:
        - x0, y0: transmitter coordinates (μm)
        - distance: Euclidean distance from origin (μm)
        - N0: number of molecules absorbed
        - absorption_times: array of absorption times (s)
        - impact_angles: array of impact angles (radians)
        - heatmap: 2D array (time_bins x angle_bins) if load_heatmaps=True
    """
    samples = []
    
    with h5py.File(mat_path, 'r') as f:
        dataset_ref = f['dataset']
        n_samples = dataset_ref.shape[1]
        
        for i in range(n_samples):
            ref = dataset_ref[0, i]
            sample_group = f[ref]
            
            x0 = float(np.array(sample_group['x0']).flatten()[0])
            y0 = float(np.array(sample_group['y0']).flatten()[0])
            distance = float(np.array(sample_group['distance']).flatten()[0])
            n0 = int(np.array(sample_group['N0']).flatten()[0])
            
            absorption_times = np.array(sample_group['absorption_times']).flatten()
            impact_angles = np.array(sample_group['impact_angles']).flatten()
            
            # Handle empty arrays
            if absorption_times.size == 0:
                absorption_times = np.array([])
                impact_angles = np.array([])
            
            sample_dict = {
                'x0': x0,
                'y0': y0,
                'distance': distance,
                'N0': n0,
                'absorption_times': absorption_times,
                'impact_angles': impact_angles
            }
            
            # Load heatmap if requested
            if load_heatmaps and 'heatmap' in sample_group:
                heatmap = np.array(sample_group['heatmap'])
                sample_dict['heatmap'] = heatmap
            
            samples.append(sample_dict)
    
    return samples


def split_data(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split data into train/val/test sets.
    
    Args:
        samples: List of sample dicts
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data: Lists of sample dicts
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))
    
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_data = [samples[i] for i in train_idx]
    val_data = [samples[i] for i in val_idx]
    test_data = [samples[i] for i in test_idx]
    
    return train_data, val_data, test_data


def get_raw_data_for_deepsets(samples, max_molecules=None):
    """
    Prepare raw absorption data for DeepSets model.
    
    Args:
        samples: List of sample dicts
        max_molecules: Maximum number of molecules per sample (for padding).
                      If None, uses the maximum found in the dataset.
    
    Returns:
        data: np.array of shape (n_samples, max_molecules, 2)
              Each row is [absorption_time, impact_angle]
        masks: np.array of shape (n_samples, max_molecules)
               Boolean mask indicating valid molecules
        targets: np.array of shape (n_samples, 2) containing [x0, y0]
    """
    if max_molecules is None:
        max_molecules = max(len(s['absorption_times']) for s in samples)
    
    n_samples = len(samples)
    data = np.zeros((n_samples, max_molecules, 2), dtype=np.float32)
    masks = np.zeros((n_samples, max_molecules), dtype=bool)
    targets = np.zeros((n_samples, 2), dtype=np.float32)
    
    for i, sample in enumerate(samples):
        n_absorbed = len(sample['absorption_times'])
        n_to_use = min(n_absorbed, max_molecules)
        
        if n_to_use > 0:
            data[i, :n_to_use, 0] = sample['absorption_times'][:n_to_use]
            data[i, :n_to_use, 1] = sample['impact_angles'][:n_to_use]
            masks[i, :n_to_use] = True
        
        targets[i] = [sample['x0'], sample['y0']]
    
    return data, masks, targets
