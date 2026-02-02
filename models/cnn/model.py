"""
CNN Model for Transmitter Localization

This model predicts (x0, y0) directly from 2D heatmap images
using a convolutional neural network.

Architecture:
    Input: (batch, 1, 100, 100) heatmap image
    → Conv2d(1→16, k=5) → ReLU → MaxPool(2)     # 100→50
    → Conv2d(16→32, k=5) → ReLU → MaxPool(2)    # 50→25  
    → Conv2d(32→64, k=3) → ReLU → MaxPool(2)    # 25→12
    → AdaptiveAvgPool(1)                         # 12→1
    → Flatten → Linear(64→32) → ReLU → Dropout
    → Linear(32→2) → (x0, y0)

Input preprocessing:
    1. Log transform: log(1 + heatmap)
    2. Normalize: (x - mean) / std
"""

import torch
import torch.nn as nn


class HeatmapCNN(nn.Module):
    """
    CNN for predicting (x0, y0) from heatmap images.
    
    Input: (batch, 1, 100, 100) heatmap
    Output: (batch, 2) -> (x0, y0) normalized to [0, 1]
    
    ~34k parameters, trains fast
    """
    
    def __init__(self, dropout=0.3):
        super().__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # 100x100 -> 50x50
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 50x50 -> 25x25
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 25x25 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
