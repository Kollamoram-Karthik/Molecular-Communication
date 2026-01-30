"""
Distance Prediction Model (Predecessor Model)

A simpler model that predicts only the Euclidean distance of the transmitter
from the origin, using only absorption time information (no impact angles).

This serves as a predecessor to the full (x0, y0) localization model.

Physics basis:
- Diffusion theory: Mean first passage time scales with distance²
- E[T] ∝ d² / D (for 2D diffusion to absorbing boundary)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


class TimeFeatureExtractor:
    """
    Extract features from absorption times only.
    No angle information used.
    """
    
    def __init__(self, n_molecules=2000):
        self.n_molecules = n_molecules
        self.time_mean = None
        self.time_std = None
        self.distance_calibration = None
        
    def fit(self, samples):
        """Fit calibration using training data"""
        all_times = []
        for s in samples:
            if len(s['absorption_times']) > 0:
                all_times.extend(s['absorption_times'])
        
        self.time_mean = np.mean(all_times)
        self.time_std = np.std(all_times)
        
        # Calibrate time → distance relationship
        # From diffusion theory: distance ∝ sqrt(time)
        times_median = []
        distances = []
        for s in samples:
            if len(s['absorption_times']) > 0:
                times_median.append(np.median(s['absorption_times']))
                distances.append(s['distance'])
        
        sqrt_times = np.sqrt(times_median)
        self.distance_calibration = np.polyfit(sqrt_times, distances, deg=2)
        
    def extract_features(self, absorption_times):
        """
        Extract time-based features only.
        
        Returns: numpy array of shape (num_features,)
        """
        n_absorbed = len(absorption_times)
        
        if n_absorbed == 0:
            return np.zeros(20, dtype=np.float32)
        
        # Basic statistics
        t_mean = np.mean(absorption_times)
        t_median = np.median(absorption_times)
        t_std = np.std(absorption_times) if n_absorbed > 1 else 0
        t_min = np.min(absorption_times)
        t_max = np.max(absorption_times)
        t_range = t_max - t_min
        
        # Percentiles
        if n_absorbed >= 10:
            t_p5 = np.percentile(absorption_times, 5)
            t_p10 = np.percentile(absorption_times, 10)
            t_p25 = np.percentile(absorption_times, 25)
            t_p75 = np.percentile(absorption_times, 75)
            t_p90 = np.percentile(absorption_times, 90)
            t_p95 = np.percentile(absorption_times, 95)
            t_iqr = t_p75 - t_p25
        else:
            t_p5 = t_p10 = t_p25 = t_p75 = t_p90 = t_p95 = t_median
            t_iqr = 0
        
        # Distribution shape
        if n_absorbed >= 3:
            t_skew = stats.skew(absorption_times)
            t_kurt = stats.kurtosis(absorption_times)
        else:
            t_skew = t_kurt = 0
        
        # Physics-based distance estimate
        sqrt_t_median = np.sqrt(t_median)
        dist_estimate = np.polyval(self.distance_calibration, sqrt_t_median)
        dist_estimate = np.clip(dist_estimate, 15, 130)
        
        # Count-based features
        absorption_prob = n_absorbed / self.n_molecules
        
        # Assemble feature vector (20 features)
        features = np.array([
            # Normalized time statistics (8)
            (t_mean - self.time_mean) / self.time_std,
            (t_median - self.time_mean) / self.time_std,
            t_std / self.time_std if self.time_std > 0 else 0,
            (t_min - self.time_mean) / self.time_std,
            t_iqr / self.time_std if self.time_std > 0 else 0,
            t_range / self.time_std if self.time_std > 0 else 0,
            t_skew,
            t_kurt / 10,
            
            # Percentiles normalized (6)
            t_p5 / 100,
            t_p10 / 100,
            t_p25 / 100,
            t_p75 / 100,
            t_p90 / 100,
            t_p95 / 100,
            
            # Count features (3)
            absorption_prob,
            np.log(n_absorbed + 1) / np.log(self.n_molecules + 1),
            (n_absorbed - self.n_molecules * 0.3) / (self.n_molecules * 0.2),
            
            # Physics-derived features (3)
            dist_estimate / 100,
            np.sqrt(t_median) / 10,
            np.sqrt(t_mean) / 10,
        ], dtype=np.float32)
        
        return features
    
    def get_physics_estimate(self, absorption_times):
        """Get physics-based distance estimate"""
        if len(absorption_times) == 0:
            return 60.0  # Default middle distance
        
        t_median = np.median(absorption_times)
        sqrt_t = np.sqrt(t_median)
        dist = np.polyval(self.distance_calibration, sqrt_t)
        dist = np.clip(dist, 15, 130)
        
        return dist


class DistanceNetwork(nn.Module):
    """Neural network for distance prediction"""
    
    def __init__(self, input_dim=20, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output: single distance value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize output layer
        nn.init.zeros_(self.network[-1].bias)
        nn.init.normal_(self.network[-1].weight, std=0.01)
    
    def forward(self, x):
        x = self.input_bn(x)
        return self.network(x)


class DistanceModel:
    """
    Complete model for distance prediction.
    
    Input: Absorption times only
    Output: Euclidean distance from origin
    """
    
    def __init__(self, hidden_dims=[64, 32], dropout=0.3, n_molecules=2000):
        self.n_molecules = n_molecules
        self.feature_extractor = TimeFeatureExtractor(n_molecules=n_molecules)
        self.network = DistanceNetwork(
            input_dim=20,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        # Distance range based on x0, y0 ∈ [15, 90]
        # min distance ≈ sqrt(15² + 15²) ≈ 21
        # max distance ≈ sqrt(90² + 90²) ≈ 127
        self.dist_min = 20
        self.dist_max = 130
        
    def fit_physics(self, samples):
        """Fit the physics-based calibration"""
        self.feature_extractor.fit(samples)
    
    def extract_features(self, sample):
        """Extract features for a single sample"""
        return self.feature_extractor.extract_features(sample['absorption_times'])
    
    def normalize_target(self, distance):
        """Normalize distance to [0, 1]"""
        return np.array([(distance - self.dist_min) / (self.dist_max - self.dist_min)], dtype=np.float32)
    
    def denormalize_target(self, pred):
        """Convert normalized prediction back to original scale"""
        return pred[0] * (self.dist_max - self.dist_min) + self.dist_min
    
    def get_physics_estimate(self, sample):
        """Get raw physics-based distance estimate"""
        return self.feature_extractor.get_physics_estimate(sample['absorption_times'])
