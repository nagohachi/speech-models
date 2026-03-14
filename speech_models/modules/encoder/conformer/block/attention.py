import torch
import torch.nn as nn

from speech_models.modules.encoder.conformer.block.utils.multihead_attention import (
    MultiheadAttentionWithRelPositionalEncoding,
)


def lens_to_mask(lens: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(lens.max().item(), device=lens.device)
    return lens.unsqueeze(1) <= indices


class ConformerSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout_prob: float,
        use_rel_positional_attn: bool = True,
    ) -> None:
        super().__init__()
        self.use_rel_positional_attn = use_rel_positional_attn
        self.layernorm = nn.LayerNorm(hidden_size)

        if self.use_rel_positional_attn:
            self.attention = MultiheadAttentionWithRelPositionalEncoding(
                hidden_size, num_heads, dropout=attention_dropout_prob
            )

        else:
            self.attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=attention_dropout_prob, batch_first=True
            )

        self.dropout = nn.Dropout(p=attention_dropout_prob, inplace=True)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor, pos_emb: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layernorm(x)
        xmask = lens_to_mask(xlens)

        if self.use_rel_positional_attn:
            x = self.attention(
                x,
                x,
                x,
                key_padding_mask=xmask,
                pos_emb=pos_emb,  # type: ignore
                is_causal=False,
            )
        else:
            x, _ = self.attention(
                x, x, x, key_padding_mask=xmask
            )  # ignore attn_output_weights

        x = self.dropout(x)

        return x, xlens
