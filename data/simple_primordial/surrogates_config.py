from dataclasses import dataclass

from torch import nn


@dataclass
class LatentNeuralODEConfig:
    """Model config for LatentNeuralODE for the simple_primordial dataset."""

    latent_features: int = 9
    layers_factor: int = 46
    learning_rate: float = 0.005
    ode_activation: nn.Module = nn.Softplus()
    ode_tanh_reg: bool = False
    coder_activation: nn.Module = nn.ReLU()
