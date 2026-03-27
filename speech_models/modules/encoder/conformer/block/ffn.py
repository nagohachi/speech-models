import torch
import torch.nn as nn


class ConformerFFN(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.mod = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_prob, inplace=True),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(p=dropout_prob, inplace=True),
        )

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mod(x), xlens
