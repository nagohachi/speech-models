import torch


def lens_to_mask(lens: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(lens.max(), device=lens.device)  # type: ignore[arg-type]
    return lens.unsqueeze(1) <= indices
