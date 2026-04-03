import torch
import torch.nn as nn
from speech_models.modules.utils.mask import lens_to_mask


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = lens_to_mask(xlens)
        return self.transformer_encoder(x, src_key_padding_mask=mask), xlens
