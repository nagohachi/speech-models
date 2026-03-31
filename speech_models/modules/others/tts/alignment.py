import numpy as np
import torch


def maximum_path(log_prior: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Find maximum probability monotonic alignment path via dynamic programming.

    Pure Python implementation (no Cython dependency).

    Args:
        log_prior (torch.Tensor): Log probability matrix of shape (B, T_text, T_mel).
        mask (torch.Tensor): Float mask of shape (B, T_text, T_mel). 1=valid, 0=pad.

    Returns:
        torch.Tensor: Binary alignment path of shape (B, T_text, T_mel).
    """
    value = log_prior * mask
    device = value.device
    dtype = value.dtype

    value_np = value.detach().cpu().numpy().astype(np.float32)
    path_np = np.zeros_like(value_np, dtype=np.int32)
    mask_np = mask.detach().cpu().numpy()

    t_x_max = mask_np.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask_np.sum(2)[:, 0].astype(np.int32)

    for b in range(value_np.shape[0]):
        _maximum_path_each(path_np[b], value_np[b], t_x_max[b], t_y_max[b])

    return torch.from_numpy(path_np).to(device=device, dtype=dtype)


def _maximum_path_each(
    path: np.ndarray, value: np.ndarray, t_x: int, t_y: int
) -> None:
    """DP for a single sample. Modifies path and value in-place."""
    max_neg_val = -1e9

    # forward pass
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[x, y - 1]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]

    # backtrack
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1


def generate_path(durations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert per-token durations to an alignment matrix.

    Args:
        durations (torch.Tensor): Duration per token of shape (B, T_text).
        mask (torch.Tensor): Validity mask of shape (B, T_text, T_mel).

    Returns:
        torch.Tensor: Alignment matrix of shape (B, T_text, T_mel).
    """
    device = durations.device
    b, t_x, t_y = mask.shape

    cum_duration = torch.cumsum(durations, dim=1)  # (B, T_text)
    cum_duration_flat = cum_duration.view(b * t_x)

    # sequence_mask: True where index < length
    indices = torch.arange(t_y, device=device).unsqueeze(0)
    path = (indices < cum_duration_flat.unsqueeze(1)).to(mask.dtype)  # (B*T_text, T_mel)
    path = path.view(b, t_x, t_y)

    # subtract shifted version to get per-token segments
    path = path - torch.nn.functional.pad(path, [0, 0, 1, 0])[:, :-1]
    return path * mask


def duration_loss(
    logw: torch.Tensor, logw_target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """MSE loss on log durations.

    Args:
        logw (torch.Tensor): Predicted log durations (B, T_text).
        logw_target (torch.Tensor): Target log durations from MAS (B, T_text).
        lengths (torch.Tensor): Text lengths (B,).

    Returns:
        torch.Tensor: Scalar loss.
    """
    return torch.sum((logw - logw_target) ** 2) / torch.sum(lengths)
