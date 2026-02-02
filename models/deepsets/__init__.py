"""
DeepSets Model for transmitter localization.

Learns directly from raw (time, angle) data without hand-crafted features.
Permutation-invariant architecture: f(X) = ρ(∑ᵢ φ(xᵢ))
"""

from .model import DeepSetsModel, DeepSetsNetwork, PhiNetwork, RhoNetwork
from .train import train, load_model, save_model
