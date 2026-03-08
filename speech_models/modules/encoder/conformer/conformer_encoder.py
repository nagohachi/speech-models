import torch
import torch.nn as nn

from speech_models.modules.encoder.conformer.block.block import ConformerBlock
from speech_models.modules.encoder.conformer.conv_subsampling import ConvSubSampling
from speech_models.modules.encoder.conformer.positional_encoding import (
    SinusoidalPositionalEncoding,
)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        conv_subsample_kernel_size: int,
        hidden_size: int,
        ffn_dropout_prob: float,
        attention_dropout_prob: float,
        convolution_dropout_prob: float,
        num_heads: int,
        add_trainable_params_to_glu: bool,
        kernel_size_in_depthwise_conv: int,
        posenc_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.conv_subsampling = ConvSubSampling(conv_subsample_kernel_size, hidden_size)
        self.positional_encoding = SinusoidalPositionalEncoding(
            hidden_size, posenc_dropout_prob
        )
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    hidden_size,
                    ffn_dropout_prob,
                    attention_dropout_prob,
                    convolution_dropout_prob,
                    num_heads,
                    add_trainable_params_to_glu,
                    kernel_size_in_depthwise_conv,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, xlens = self.conv_subsampling(x, xlens)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x, xlens = block(x, xlens)

        return x, xlens
