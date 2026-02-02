"""
Feature-based MLP Model for transmitter localization.

Uses hand-crafted physics features from absorption times and angles.
"""

from .model import FeatureMLPModel, FeatureMLP, PhysicsFeatureExtractor
from .train import train, load_model, save_model
