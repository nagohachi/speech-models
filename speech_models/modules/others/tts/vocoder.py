import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm as _remove_weight_norm
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: tuple[int, ...] = (1, 3, 5)
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=(kernel_size * d - d) // 2,
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                for _ in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for c in self.convs1:
            _remove_weight_norm(c)
        for c in self.convs2:
            _remove_weight_norm(c)


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN Generator (V1 compatible)."""

    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channel: int = 512,
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ) -> None:
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        ch // (2**i),
                        ch // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch_out = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch_out, k, tuple(d)))

        self.conv_post = weight_norm(nn.Conv1d(ch_out, 1, 7, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            x (torch.Tensor): Mel spectrogram of shape (batch_size, n_mels, mel_len).

        Returns:
            torch.Tensor: Waveform of shape (batch_size, 1, wav_len).
        """
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs: torch.Tensor = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        for up in self.ups:
            _remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        _remove_weight_norm(self.conv_pre)
        _remove_weight_norm(self.conv_post)


def load_hifigan(
    config_path: str | Path, checkpoint_path: str | Path
) -> HiFiGANGenerator:
    """Load a pretrained HiFi-GAN generator.

    Args:
        config_path: Path to config.json.
        checkpoint_path: Path to generator checkpoint.

    Returns:
        HiFiGANGenerator loaded with pretrained weights.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    generator = HiFiGANGenerator(
        upsample_initial_channel=config["upsample_initial_channel"],
        upsample_rates=config["upsample_rates"],
        upsample_kernel_sizes=config["upsample_kernel_sizes"],
        resblock_kernel_sizes=config["resblock_kernel_sizes"],
        resblock_dilation_sizes=config["resblock_dilation_sizes"],
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(state_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()

    return generator
