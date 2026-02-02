"""
Distance MLP Model for transmitter localization.

Predicts only distance using time features (no angles).
"""

from .model import DistanceMLPModel, DistanceMLP, TimeFeatureExtractor
from .train import train, load_model, save_model
