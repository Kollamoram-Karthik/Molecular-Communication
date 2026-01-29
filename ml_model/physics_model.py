"""
Physics-Informed Transmitter Localization Model - Enhanced Version

Key insight: The problem has two components:
1. DISTANCE estimation: Mean absorption time scales with distance² (diffusion theory)
2. DIRECTION estimation: Impact angle distribution is biased toward transmitter direction

This model uses physics-derived features with calibrated relationships,
then applies a neural network to learn corrections.

IMPORTANT: Sub-micrometer accuracy requires either:
- Much more training data (10,000+ samples)
- More molecules per simulation (5000+)
- Or we accept that diffusion noise limits accuracy to ~5-10 μm
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


class PhysicsFeatureExtractor:
    """
    Extract physics-informed features from molecular absorption data.
    
    Physics background:
    - Diffusion: Mean first passage time ~ d² / D
    - Impact angles: Biased toward transmitter direction
    """
    
    def __init__(self, n_molecules=500):
        self.n_molecules = n_molecules  # Total molecules emitted
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
        times_median = []
        distances = []
        for s in samples:
            if len(s['absorption_times']) > 0:
                times_median.append(np.median(s['absorption_times']))
                distances.append(s['distance'])
        
        sqrt_times = np.sqrt(times_median)
        self.distance_calibration = np.polyfit(sqrt_times, distances, deg=2)
        
    def extract_features(self, absorption_times, impact_angles):
        """Extract comprehensive physics-based features."""
        n_absorbed = len(absorption_times)
        
        if n_absorbed == 0:
            return np.zeros(35, dtype=np.float32)
        
        # ========================
        # TIME-BASED FEATURES
        # ========================
        t_mean = np.mean(absorption_times)
        t_median = np.median(absorption_times)
        t_std = np.std(absorption_times) if n_absorbed > 1 else 0
        t_min = np.min(absorption_times)
        t_max = np.max(absorption_times)
        
        if n_absorbed >= 10:
            t_p10 = np.percentile(absorption_times, 10)
            t_p25 = np.percentile(absorption_times, 25)
            t_p75 = np.percentile(absorption_times, 75)
            t_p90 = np.percentile(absorption_times, 90)
            t_iqr = t_p75 - t_p25
        else:
            t_p10 = t_p25 = t_p75 = t_p90 = t_median
            t_iqr = 0
        
        if n_absorbed >= 3:
            t_skew = stats.skew(absorption_times)
            t_kurt = stats.kurtosis(absorption_times)
        else:
            t_skew = t_kurt = 0
        
        # Distance estimate from calibration
        sqrt_t_median = np.sqrt(t_median)
        dist_estimate = np.polyval(self.distance_calibration, sqrt_t_median)
        dist_estimate = np.clip(dist_estimate, 15, 130)
        
        # ========================
        # ANGLE-BASED FEATURES
        # ========================
        sin_angles = np.sin(impact_angles)
        cos_angles = np.cos(impact_angles)
        
        mean_sin = np.mean(sin_angles)
        mean_cos = np.mean(cos_angles)
        circular_mean = np.arctan2(mean_sin, mean_cos)
        resultant_length = np.sqrt(mean_sin**2 + mean_cos**2)
        circular_var = 1 - resultant_length
        circular_std = np.sqrt(-2 * np.log(resultant_length + 1e-10)) if resultant_length > 0.01 else np.pi
        
        sin_std = np.std(sin_angles)
        cos_std = np.std(cos_angles)
        
        # Higher order angle statistics
        sin2_mean = np.mean(np.sin(2 * impact_angles))
        cos2_mean = np.mean(np.cos(2 * impact_angles))
        
        # ========================
        # CROSS FEATURES
        # ========================
        if n_absorbed >= 5:
            angle_deviation = np.abs(np.angle(np.exp(1j * (impact_angles - circular_mean))))
            time_angle_corr = np.corrcoef(absorption_times, angle_deviation)[0, 1]
            if np.isnan(time_angle_corr):
                time_angle_corr = 0
            
            # Early vs late molecule angle difference
            sorted_idx = np.argsort(absorption_times)
            n_early = max(1, n_absorbed // 4)
            early_angles = impact_angles[sorted_idx[:n_early]]
            late_angles = impact_angles[sorted_idx[-n_early:]]
            
            early_mean_angle = np.arctan2(np.mean(np.sin(early_angles)), np.mean(np.cos(early_angles)))
            late_mean_angle = np.arctan2(np.mean(np.sin(late_angles)), np.mean(np.cos(late_angles)))
            angle_drift = np.angle(np.exp(1j * (late_mean_angle - early_mean_angle)))
        else:
            time_angle_corr = 0
            angle_drift = 0
        
        # ========================
        # PHYSICS ESTIMATES
        # ========================
        x0_physics = dist_estimate * np.cos(circular_mean)
        y0_physics = dist_estimate * np.sin(circular_mean)
        
        # Count features - use self.n_molecules for normalization
        absorption_prob = n_absorbed / self.n_molecules
        
        # Assemble feature vector
        features = np.array([
            # Time features (11)
            (t_mean - self.time_mean) / self.time_std,
            (t_median - self.time_mean) / self.time_std,
            t_std / self.time_std if self.time_std > 0 else 0,
            (t_min - self.time_mean) / self.time_std,
            t_iqr / self.time_std if self.time_std > 0 else 0,
            t_skew,
            t_kurt / 10,  # Scale down
            t_p10 / 100,
            t_p25 / 100,
            t_p75 / 100,
            t_p90 / 100,
            
            # Angle features (10)
            np.sin(circular_mean),
            np.cos(circular_mean),
            resultant_length,
            circular_std / np.pi,
            sin_std,
            cos_std,
            sin2_mean,
            cos2_mean,
            mean_sin,
            mean_cos,
            
            # Cross features (2)
            time_angle_corr,
            angle_drift / np.pi,
            
            # Count features (3)
            absorption_prob,
            np.log(n_absorbed + 1) / np.log(self.n_molecules + 1),
            (n_absorbed - self.n_molecules * 0.3) / (self.n_molecules * 0.2),  # Centered count
            
            # Physics estimates (6)
            dist_estimate / 100,
            x0_physics / 100,
            y0_physics / 100,
            np.sqrt(t_median) / 10,
            absorption_prob * dist_estimate / 100,
            resultant_length * dist_estimate / 100,
            
            # Additional derived (3)
            np.sqrt(t_mean) / 10,
            (t_max - t_min) / 100,
            t_median / t_mean if t_mean > 0 else 1,
        ], dtype=np.float32)
        
        return features
    
    def get_physics_estimate(self, absorption_times, impact_angles):
        """Get raw physics-based (x0, y0) estimate"""
        if len(absorption_times) == 0:
            return 50.0, 50.0
        
        t_median = np.median(absorption_times)
        sqrt_t = np.sqrt(t_median)
        dist = np.polyval(self.distance_calibration, sqrt_t)
        dist = np.clip(dist, 15, 130)
        
        sin_angles = np.sin(impact_angles)
        cos_angles = np.cos(impact_angles)
        direction = np.arctan2(np.mean(sin_angles), np.mean(cos_angles))
        
        x0 = dist * np.cos(direction)
        y0 = dist * np.sin(direction)
        
        return x0, y0


class CorrectionNetwork(nn.Module):
    """Enhanced neural network with residual connections."""
    
    def __init__(self, input_dim=35, hidden_dims=[128, 128, 64], dropout=0.3):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.drop1 = nn.Dropout(dropout)
        
        # Second hidden layer with residual
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop2 = nn.Dropout(dropout)
        
        # Third hidden layer
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.drop3 = nn.Dropout(dropout)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dims[2], 2)
        
        # Activation
        self.act = nn.LeakyReLU(0.1)
        
        # Initialize output layer small
        nn.init.zeros_(self.fc_out.bias)
        nn.init.normal_(self.fc_out.weight, std=0.01)
    
    def forward(self, x):
        x = self.input_bn(x)
        
        # Layer 1
        h1 = self.act(self.bn1(self.fc1(x)))
        h1 = self.drop1(h1)
        
        # Layer 2 with skip from layer 1
        h2 = self.act(self.bn2(self.fc2(h1)))
        h2 = self.drop2(h2) + h1  # Residual connection
        
        # Layer 3
        h3 = self.act(self.bn3(self.fc3(h2)))
        h3 = self.drop3(h3)
        
        # Output
        out = self.fc_out(h3)
        return out


class PhysicsInformedModel:
    """Complete model combining physics-based feature extraction with neural network."""
    
    def __init__(self, hidden_dims=[128, 128, 64], dropout=0.3, n_molecules=500):
        self.n_molecules = n_molecules
        self.feature_extractor = PhysicsFeatureExtractor(n_molecules=n_molecules)
        self.network = CorrectionNetwork(
            input_dim=35,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        self.x0_min, self.x0_max = 15, 90
        self.y0_min, self.y0_max = 15, 90
        
    def fit_physics(self, samples):
        """Fit the physics-based calibration"""
        self.feature_extractor.fit(samples)
    
    def extract_features(self, sample):
        """Extract features for a single sample"""
        return self.feature_extractor.extract_features(
            sample['absorption_times'],
            sample['impact_angles']
        )
    
    def normalize_target(self, x0, y0):
        """Normalize targets to [0, 1]"""
        x_norm = (x0 - self.x0_min) / (self.x0_max - self.x0_min)
        y_norm = (y0 - self.y0_min) / (self.y0_max - self.y0_min)
        return np.array([x_norm, y_norm], dtype=np.float32)
    
    def denormalize_target(self, pred):
        """Convert normalized predictions back to original scale"""
        x0 = pred[0] * (self.x0_max - self.x0_min) + self.x0_min
        y0 = pred[1] * (self.y0_max - self.y0_min) + self.y0_min
        return x0, y0
    
    def get_physics_estimate(self, sample):
        """Get raw physics-based estimate"""
        return self.feature_extractor.get_physics_estimate(
            sample['absorption_times'],
            sample['impact_angles']
        )
