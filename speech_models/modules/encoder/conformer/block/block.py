import torch
import torch.nn as nn

from .attention import ConformerSelfAttention
from .conv import ConformerConv
from .ffn import ConformerFFN


class ConformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_dropout_prob: float,
        attention_dropout_prob: float,
        convolution_dropout_prob: float,
        num_heads: int,
        add_trainable_params_to_glu: bool,
        kernel_size_in_depthwise_conv: int,
    ) -> None:
        super().__init__()
        self.ffn1 = ConformerFFN(hidden_size, ffn_dropout_prob)
        self.attention = ConformerSelfAttention(
            hidden_size, num_heads, attention_dropout_prob
        )
        self.conv = ConformerConv(
            hidden_size,
            add_trainable_params_to_glu,
            kernel_size_in_depthwise_conv,
            convolution_dropout_prob,
        )
        self.ffn2 = ConformerFFN(hidden_size, ffn_dropout_prob)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        ffn1_out, xlens = self.ffn1(x, xlens)
        x = x + 0.5 * ffn1_out

        attn_out, xlens = self.attention(x, xlens)
        x = x + attn_out

        conv_out, xlens = self.conv(x, xlens)
        x = x + conv_out

        ffn2_out, xlens = self.ffn2(x, xlens)
        x = x + 0.5 * ffn2_out

        x = self.layernorm(x)

        return x, xlens
