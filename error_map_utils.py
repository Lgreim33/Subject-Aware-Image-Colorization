import torch


def invert_error_map(error_map: torch.Tensor) -> torch.Tensor:
    """Inverts an error map so that high error values become low and vice versa.

    Args:
        error_map (torch.Tensor): The input error map tensor.

    Returns:
        torch.Tensor: The inverted error map tensor.
    """
    return 1.0 - error_map


def normalize_error_map(error_map: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalizes an error map to the range [0, 1].

    Args:
        error_map (torch.Tensor): The input error map tensor.
        epsilon (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized error map tensor.
    """
    min_val = torch.min(error_map)
    max_val = torch.max(error_map)
    normalized_map = (error_map - min_val) / (max_val - min_val + epsilon)
    return normalized_map
