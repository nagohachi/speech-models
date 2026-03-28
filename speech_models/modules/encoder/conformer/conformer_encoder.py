import torch
import torch.nn as nn

from speech_models.modules.encoder.conformer.block.block import ConformerBlock
from speech_models.modules.encoder.conformer.block.variational_noise import (
    apply_variational_noise,
)
from speech_models.modules.encoder.conformer.conv_subsampling import ConvSubSampling
from speech_models.modules.utils.positional_encoding import (
    RelPositionalEncoding,
    SinusoidalPositionalEncoding,
)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        num_mels: int,
        num_blocks: int,
        hidden_size: int,
        ffn_dropout_prob: float,
        attention_dropout_prob: float,
        convolution_dropout_prob: float,
        num_heads: int,
        add_trainable_params_to_glu: bool,
        use_rel_positional_attn: bool,
        kernel_size_in_depthwise_conv: int,
        posenc_dropout_prob: float,
        variational_noise_std: float = 0.0,
        use_subsampling: bool = True,
        conv_subsample_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.use_subsampling = use_subsampling
        self.use_rel_positional_attn = use_rel_positional_attn
        self.hidden_size = hidden_size

        if self.use_subsampling:
            self.input_proj = ConvSubSampling(
                conv_subsample_kernel_size, hidden_size, num_mels
            )
        else:
            self.input_proj = nn.Linear(num_mels, hidden_size)

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

        apply_variational_noise(self, variational_noise_std)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_subsampling:
            x, xlens = self.input_proj(x, xlens)
        else:
            x = self.input_proj(x)

        if self.use_rel_positional_attn:
            x, pos_emb = self.positional_encoding(x)
        else:
            x = self.positional_encoding(x)
            pos_emb = None

        for block in self.blocks:
            x, xlens = block(x, xlens, pos_emb)

        return x, xlens
