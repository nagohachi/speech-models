import torch
import torch.nn as nn

from speech_models.modules.encoder.conformer.block.block import ConformerBlock
from speech_models.modules.encoder.conformer.conv_subsampling import ConvSubSampling
from speech_models.modules.encoder.conformer.positional_encoding import (
    RelPositionalEncoding,
    SinusoidalPositionalEncoding,
)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        num_mels: int,
        num_blocks: int,
        conv_subsample_kernel_size: int,
        hidden_size: int,
        ffn_dropout_prob: float,
        attention_dropout_prob: float,
        convolution_dropout_prob: float,
        num_heads: int,
        add_trainable_params_to_glu: bool,
        use_rel_positional_attn: bool,
        kernel_size_in_depthwise_conv: int,
        posenc_dropout_prob: float,
    ) -> None:
        super().__init__()
        self.conv_subsampling = ConvSubSampling(
            conv_subsample_kernel_size, hidden_size, num_mels
        )

        self.use_rel_positional_attn = use_rel_positional_attn

        if self.use_rel_positional_attn:
            self.positional_encoding = RelPositionalEncoding(
                hidden_size, posenc_dropout_prob
            )
        else:
            self.positional_encoding = SinusoidalPositionalEncoding(
                hidden_size, posenc_dropout_prob
            )

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    hidden_size,
                    use_rel_positional_attn,
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

        if self.use_rel_positional_attn:
            x, pos_emb = self.positional_encoding(x)
        else:
            x = self.positional_encoding(x)
            pos_emb = None

        for block in self.blocks:
            x, xlens = block(x, xlens, pos_emb)

        return x, xlens
