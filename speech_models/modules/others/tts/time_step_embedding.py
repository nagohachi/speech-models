import math

import torch
import torch.nn as nn


class TimeStepEmbedding(nn.Module):
    """Timestep embedding for diffusion/flow matching models.

    Converts scalar timesteps to embeddings via:
    1. Sinusoidal encoding (fixed, no learnable params)
    2. MLP (Linear -> SiLU -> Linear)
    """

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        assert hidden_size % 2 == 0, "hidden_size must be even for sinusoidal embedding"
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Forward step of timestep embedding.

        Args:
            time_steps (torch.Tensor): Tensor of shape (batch_size,), values in [0, 1].

        Returns:
            torch.Tensor: Timestep embedding of shape (batch_size, output_size).
        """
        emb = self._sinusoidal_embedding(time_steps)
        return self.mlp(emb)

    def _sinusoidal_embedding(
        self, t: torch.Tensor, scale: float = 1000.0
    ) -> torch.Tensor:
        """Convert scalar timesteps to sinusoidal embeddings.

        Args:
            t (torch.Tensor): Timestep tensor of shape (batch_size,).
            scale (float): Scaling factor for timesteps.

        Returns:
            torch.Tensor: Sinusoidal embedding of shape (batch_size, hidden_size).
        """
        half_dim = self.hidden_size // 2
        freq = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * -(math.log(10000) / (half_dim - 1))
        )
        emb = scale * t.unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
