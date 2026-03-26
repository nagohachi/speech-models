from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class GlobalMVN(nn.Module):
    """Apply global mean and variance normalization.

    Pre-computed statistics (mean/var) are loaded from an .npy or .npz file
    and registered as non-trainable buffers.

    Args:
        stats_file: Path to .npy (Kaldi-style) or .npz file containing statistics.
        norm_means: Whether to apply mean normalization.
        norm_vars: Whether to apply variance normalization.
        eps: Floor value for variance to avoid division by zero.
    """

    def __init__(
        self,
        stats_file: str | Path,
        norm_means: bool = True,
        norm_vars: bool = True,
        eps: float = 1.0e-20,
    ) -> None:
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps
        self.stats_file = str(stats_file)

        stats = np.load(stats_file)
        if isinstance(stats, np.ndarray):
            # Kaldi-style: row 0 = [sum, count], row 1 = [sum_square, ...]
            count = stats[0].flatten()[-1]
            mean = stats[0, :-1] / count
            var = stats[1, :-1] / count - mean * mean
        else:
            # npz with keys: count, sum, sum_square
            count = stats["count"]
            sum_v = stats["sum"]
            sum_square_v = stats["sum_square"]
            mean = sum_v / count
            var = sum_square_v / count - mean * mean

        std = np.sqrt(np.maximum(var, eps))

        self.register_buffer("mean", torch.from_numpy(np.asarray(mean, dtype=np.float32)))
        self.register_buffer("std", torch.from_numpy(np.asarray(std, dtype=np.float32)))

    def extra_repr(self) -> str:
        return (
            f"stats_file={self.stats_file}, "
            f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"
        )

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply global MVN.

        Args:
            x: Feature tensor of shape (B, T, D).
            xlens: Lengths of shape (B,).

        Returns:
            Normalized features and lengths (unchanged).
        """
        if self.norm_means:
            x = x - self.mean
        if self.norm_vars:
            x = x / self.std
        return x, xlens
