import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self, hidden_size: int, dropout_prob: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        pe = torch.zeros(1, max_len, hidden_size)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]  # type: ignore
        return self.dropout(x)
