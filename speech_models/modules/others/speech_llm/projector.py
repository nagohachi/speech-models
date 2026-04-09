import torch
import torch.nn as nn


class LinearProjector(nn.Module):
    """Simple linear projection from encoder to LLM embedding space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            (B, T, output_dim)
        """
        return self.proj(x)


class MLPProjector(nn.Module):
    """Two-layer MLP projector with GELU activation and optional frame stacking.

    When ``downsample_k > 1``, adjacent frames are concatenated before the MLP,
    reducing the sequence length by a factor of ``downsample_k`` (following
    ESPnet's ``FrameStackingMLP2PostEncoder``).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        downsample_k: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.ds_k = downsample_k
        self.fc1 = nn.Linear(input_dim * downsample_k, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            (B, T', output_dim) where T' = T // downsample_k
        """
        if self.ds_k > 1:
            B, T, D = x.shape
            T_new = T // self.ds_k
            x = x[:, : T_new * self.ds_k, :].reshape(B, T_new, D * self.ds_k)
        return self.fc2(self.act(self.fc1(x)))


class ConvProjector(nn.Module):
    """1D convolution for temporal downsampling followed by linear projection.

    Reduces the number of speech prefix tokens fed to the LLM.
    For example, with kernel_size=5 and stride=5, 1500 whisper frames become 300 tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 5,
        stride: int = 5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim, output_dim, kernel_size, stride, padding=kernel_size // 2
        )
        self.act = nn.GELU()
        self.proj = nn.Linear(output_dim, output_dim)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            (B, T', output_dim) where T' ≈ T / stride
        """
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T', output_dim)
        return self.proj(self.act(x))
