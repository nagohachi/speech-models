import torch
import torch.nn as nn


def lens_to_mask(lens: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(lens.max().item(), device=lens.device)
    return lens.unsqueeze(1) <= indices


class ConformerSelfAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, attention_dropout_prob: float
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attention_dropout_prob, batch_first=True
        )
        self.dropout = nn.Dropout(p=attention_dropout_prob, inplace=True)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layernorm(x)
        xmask = lens_to_mask(xlens)

        x, _ = self.attention(
            x, x, x, key_padding_mask=xmask
        )  # ignore attn_output_weights

        x = self.dropout(x)

        return x, xlens
