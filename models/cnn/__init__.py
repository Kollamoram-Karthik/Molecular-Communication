"""
CNN Model for transmitter localization.

Predicts (x0, y0) directly from 2D heatmap images.
"""

from .model import HeatmapCNN
from .train import train, load_model, save_model, HeatmapDataset
