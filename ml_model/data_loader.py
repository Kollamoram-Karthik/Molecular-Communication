"""
Data loading utilities for molecular communication dataset.
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
        - x0, y0: transmitter coordinates
        - distance: distance from origin
        - absorption_times: array of absorption times
        - impact_angles: array of impact angles
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
    """Split data into train/val/test sets"""
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
