from dataclasses import dataclass, field

import torch


@dataclass
class NeuralODEConfigOSU:
    "Model config for OSU_2008 dataset"
    # in_features: int = 29
    latent_features: int = 5
    coder_hidden: int = 4
    coder_layers: list[int] = field(default_factory=lambda: [32, 16, 8])
    coder_activation: torch.nn.Module = torch.nn.ReLU()
    ode_activation: torch.nn.Module = torch.nn.Tanh()
    ode_hidden: int = 4
    ode_layer_width: int = 64
    ode_tanh_reg: bool = True
    use_adjoint: bool = False
    rtol: float = 1e-7
    atol: float = 1e-9
    method: str = "dopri8"
    t_steps = 100
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-5
