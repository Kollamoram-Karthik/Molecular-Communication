"""
DeepSets Model for Transmitter Localization

This model learns directly from raw molecular absorption data without
hand-crafted features. It uses the DeepSets architecture which is
permutation-invariant over the set of molecules.

Architecture (DeepSets):
    Input: (batch, N_max, 2) - [absorption_time, impact_angle] per molecule
           with mask for variable-length sequences
    
    φ Network (per-molecule encoder):
        Linear(2 → 64) → ReLU
        Linear(64 → 128) → ReLU  
        Linear(128 → 128)
    
    Aggregation (permutation invariant):
        Masked mean pooling over molecules
    
    ρ Network (set-level predictor):
        Linear(128 → 64) → ReLU → Dropout
        Linear(64 → 32) → ReLU → Dropout
        Linear(32 → 2) → (x0, y0)

Key insight:
    DeepSets theorem: Any permutation-invariant function can be decomposed as
    f(X) = ρ(∑ᵢ φ(xᵢ))
    
    This allows the model to learn what features matter, rather than us
    hand-crafting them (like mean, std, percentiles, etc.)

References:
    Zaheer et al., "Deep Sets", NeurIPS 2017
"""

import numpy as np
import torch
import torch.nn as nn


class PhiNetwork(nn.Module):
    """
    Per-element encoder φ(x).
    Processes each molecule's (time, angle) independently.
    """
    
    def __init__(self, input_dim=2, hidden_dims=[64, 128], output_dim=128):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, N_max, 2)
        # output: (batch, N_max, output_dim)
        return self.network(x)


class RhoNetwork(nn.Module):
    """
    Set-level predictor ρ(z).
    Takes aggregated representation and outputs (x0, y0).
    """
    
    def __init__(self, input_dim=128, hidden_dims=[64, 32], output_dim=2, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, input_dim)
        # output: (batch, output_dim)
        return self.network(x)


class DeepSetsNetwork(nn.Module):
    """
    Complete DeepSets model: f(X) = ρ(aggregate(φ(X)))
    
    Permutation invariant over the set of molecules.
    """
    
    def __init__(
        self,
        input_dim=2,
        phi_hidden_dims=[64, 128],
        phi_output_dim=128,
        rho_hidden_dims=[64, 32],
        output_dim=2,
        dropout=0.3,
        aggregation='mean'  # 'mean', 'sum', or 'max'
    ):
        super().__init__()
        
        self.phi = PhiNetwork(input_dim, phi_hidden_dims, phi_output_dim)
        self.rho = RhoNetwork(phi_output_dim, rho_hidden_dims, output_dim, dropout)
        self.aggregation = aggregation
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, N_max, 2) - molecular data [time, angle]
            mask: (batch, N_max) - boolean mask, True for valid molecules
        
        Returns:
            (batch, 2) - predicted (x0, y0) normalized to [0, 1]
        """
        # Apply φ to each molecule
        phi_out = self.phi(x)  # (batch, N_max, phi_output_dim)
        
        # Masked aggregation
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (batch, N_max, 1)
            
            if self.aggregation == 'mean':
                # Masked mean
                phi_out = phi_out * mask
                sum_out = phi_out.sum(dim=1)  # (batch, phi_output_dim)
                count = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
                aggregated = sum_out / count
            elif self.aggregation == 'sum':
                phi_out = phi_out * mask
                aggregated = phi_out.sum(dim=1)
            elif self.aggregation == 'max':
                phi_out = phi_out.masked_fill(~mask.bool(), float('-inf'))
                aggregated = phi_out.max(dim=1)[0]
        else:
            if self.aggregation == 'mean':
                aggregated = phi_out.mean(dim=1)
            elif self.aggregation == 'sum':
                aggregated = phi_out.sum(dim=1)
            elif self.aggregation == 'max':
                aggregated = phi_out.max(dim=1)[0]
        
        # Apply ρ to get final prediction
        out = self.rho(aggregated)
        
        return out


class DeepSetsModel:
    """
    Complete DeepSets model wrapper with preprocessing and normalization.
    
    Input: Raw (absorption_times, impact_angles) arrays
    Output: (x0, y0) in μm
    """
    
    def __init__(
        self,
        max_molecules=None,
        phi_hidden_dims=[64, 128],
        phi_output_dim=128,
        rho_hidden_dims=[64, 32],
        dropout=0.3,
        aggregation='mean'
    ):
        self.max_molecules = max_molecules  # Will be set from data
        self.network = DeepSetsNetwork(
            input_dim=2,
            phi_hidden_dims=phi_hidden_dims,
            phi_output_dim=phi_output_dim,
            rho_hidden_dims=rho_hidden_dims,
            output_dim=2,
            dropout=dropout,
            aggregation=aggregation
        )
        
        # Normalization parameters (set during fit)
        self.time_mean = None
        self.time_std = None
        self.angle_mean = 0.0  # Angles are already in [-π, π]
        self.angle_std = np.pi
        
        # Target range: x0, y0 ∈ [15, 90] μm
        self.x0_min, self.x0_max = 15, 90
        self.y0_min, self.y0_max = 15, 90
    
    def fit(self, samples):
        """
        Compute normalization statistics from training data.
        Also determines max_molecules if not set.
        """
        all_times = []
        max_n = 0
        
        for s in samples:
            times = s['absorption_times']
            if len(times) > 0:
                all_times.extend(times)
            max_n = max(max_n, len(times))
        
        self.time_mean = np.mean(all_times)
        self.time_std = np.std(all_times)
        
        if self.max_molecules is None:
            self.max_molecules = max_n
        
        print(f"DeepSets fit: max_molecules={self.max_molecules}, "
              f"time_mean={self.time_mean:.2f}, time_std={self.time_std:.2f}")
    
    def preprocess_sample(self, sample):
        """
        Convert a sample to normalized (data, mask) format.
        
        Returns:
            data: (max_molecules, 2) array
            mask: (max_molecules,) boolean array
        """
        times = sample['absorption_times']
        angles = sample['impact_angles']
        n_absorbed = len(times)
        
        data = np.zeros((self.max_molecules, 2), dtype=np.float32)
        mask = np.zeros(self.max_molecules, dtype=bool)
        
        n_to_use = min(n_absorbed, self.max_molecules)
        
        if n_to_use > 0:
            # Normalize time: (t - mean) / std
            data[:n_to_use, 0] = (times[:n_to_use] - self.time_mean) / self.time_std
            # Normalize angle: already in [-π, π], divide by π to get [-1, 1]
            data[:n_to_use, 1] = angles[:n_to_use] / np.pi
            mask[:n_to_use] = True
        
        return data, mask
    
    def normalize_target(self, x0, y0):
        """Normalize targets to [0, 1]"""
        x_norm = (x0 - self.x0_min) / (self.x0_max - self.x0_min)
        y_norm = (y0 - self.y0_min) / (self.y0_max - self.y0_min)
        return np.array([x_norm, y_norm], dtype=np.float32)
    
    def denormalize_target(self, pred):
        """Convert normalized predictions back to μm"""
        x0 = pred[0] * (self.x0_max - self.x0_min) + self.x0_min
        y0 = pred[1] * (self.y0_max - self.y0_min) + self.y0_min
        return x0, y0
    
    def predict(self, sample):
        """Make prediction for a single sample"""
        self.network.eval()
        
        data, mask = self.preprocess_sample(sample)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        
        with torch.no_grad():
            pred_normalized = self.network(data_tensor, mask_tensor).numpy()[0]
        
        return self.denormalize_target(pred_normalized)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
