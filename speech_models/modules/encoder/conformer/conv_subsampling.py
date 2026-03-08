import torch
import torch.nn as nn


class ConvSubSampling(nn.Module):
    def __init__(self, kernel_size: int, hidden_size: int) -> None:
        self.mod = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size),
            nn.ReLU(),
        )
        self.kernel_size = kernel_size

    def _calc_out_seq_len(
        self,
        seq_len: torch.Tensor,
        padding: int = 0,
        dilation: int = 1,
        stride: int = 1,
    ) -> torch.Tensor:
        # calculate output length after conv subsampling module.
        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        return (
            seq_len + 2 * padding - dilation * (self.kernel_size - 1) - 1
        ) // stride + 1

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mod(x)

        # 2x conv
        out_lens = self._calc_out_seq_len(self._calc_out_seq_len(xlens))

        return out, out_lens
