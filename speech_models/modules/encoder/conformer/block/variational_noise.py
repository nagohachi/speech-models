import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class _WeightNoise(nn.Module):
    """Parametrization that adds Gaussian noise to a weight tensor during training.

    When registered via `torch.nn.utils.parametrize`, noise is sampled each time
    the weight is accessed in a forward pass. Since a Conformer processes all time
    steps simultaneously, the same noised weight is used for every frame within an
    utterance — equivalent to the "per string non-cumulative additive" model in
    Jim et al. (1996).
    """

    def __init__(self, noise_std: float) -> None:
        super().__init__()
        self.noise_std = noise_std

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            return weight + torch.randn_like(weight) * self.noise_std
        return weight


def apply_variational_noise(model: nn.Module, noise_std: float) -> None:
    """Apply variational weight noise to all Linear and Conv layers in the model."""
    if noise_std <= 0.0:
        return
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            parametrize.register_parametrization(
                module, "weight", _WeightNoise(noise_std)
            )
