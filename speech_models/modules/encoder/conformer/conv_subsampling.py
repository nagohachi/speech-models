import torch
import torch.nn as nn


class ConvSubSampling(nn.Module):
    def __init__(self, kernel_size: int, hidden_size: int, num_mels: int) -> None:
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2),
            nn.ReLU(),
        )
        self.kernel_size = kernel_size

        freq_dim = self._calc_out_seq_len(torch.tensor(num_mels), stride=2).item()
        freq_dim = self._calc_out_seq_len(torch.tensor(freq_dim), stride=2).item()

        self.out_proj = nn.Linear(hidden_size * int(freq_dim), hidden_size)

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

        # -> (batch_size, 1, h, w)
        x = x.unsqueeze(1)
        out = self.mod(x)

        batch_size, channels, time_out, freq_out = out.size()

        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, time_out, channels * freq_out)
        out = self.out_proj(out)

        # 2x conv
        out_lens = self._calc_out_seq_len(
            self._calc_out_seq_len(xlens, stride=2), stride=2
        )

        return out, out_lens
