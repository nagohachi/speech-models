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


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(2 * max_len - 1, d_model)

        # position: [-4999, -4998, ..., 0, ..., 4998, 4999]
        position = torch.arange(-(max_len - 1), max_len, dtype=torch.float32).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, hidden_size).
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x applied with dropout (absolute PE is NOT added)
                - pos_emb of shape (1, 2*s-1, hidden_size)
        """
        s = x.size(1)

        # center of pe: max_len - 1
        center = self.pe.size(1) // 2  # type: ignore

        pos_emb = self.pe[:, center - s + 1 : center + s]  # type: ignore
        x = self.dropout(x)

        return x, pos_emb
