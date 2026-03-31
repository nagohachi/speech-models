import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """Predicts log-scaled durations per phoneme token.

    Architecture: Conv1d -> ReLU -> LayerNorm -> Dropout (x2) -> Conv1d(1x1)
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int = 256,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Encoder output of shape (B, T_text, in_channels).
            x_mask (torch.Tensor): Float mask of shape (B, T_text). 1=valid, 0=pad.

        Returns:
            torch.Tensor: Log durations of shape (B, T_text).
        """
        # (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x_mask_conv = x_mask.unsqueeze(1)  # (B, 1, T)

        x = self.conv_1(x * x_mask_conv)
        x = torch.relu(x)
        x = self.norm_1(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(x)

        x = self.conv_2(x * x_mask_conv)
        x = torch.relu(x)
        x = self.norm_2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(x)

        x = self.proj(x * x_mask_conv)  # (B, 1, T)
        return (x * x_mask_conv).squeeze(1)  # (B, T)
