"""Utility functions for Simply PyTorch optimizers.

This module provides helper functions for the Muon optimizer, particularly
the Newton-Schulz orthogonalization algorithm.
"""

import torch


def newton_schulz_orthogonalize(
    x: torch.Tensor,
    num_steps: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Orthogonalize a matrix using Newton-Schulz iterations.

    This function implements the Newton-Schulz algorithm for matrix orthogonalization,
    which iteratively refines a matrix to make it orthogonal. This is used in the
    Muon optimizer for gradient preconditioning.

    Reference: JAX implementation lines 272-301 in optimizers.py

    Args:
        x: Input tensor of shape (..., M, N) where ... represents optional batch dimensions
        num_steps: Number of Newton-Schulz iterations (default: 5)
        eps: Small constant for numerical stability (default: 1e-8)

    Returns:
        Orthogonalized and scaled tensor of the same shape as input

    Mathematical formulation:
        For iteration i:
            A = X^T @ X
            B = muon_b * A + muon_c * A^2
            X = muon_a * X + B @ X

        where muon_a = 3.4445, muon_b = -4.7750, muon_c = 2.0315
    """
    # Constants from Muon paper
    muon_a = 3.4445
    muon_b = -4.7750
    muon_c = 2.0315

    # Ensure more columns than rows for efficiency
    transposed = x.shape[-1] < x.shape[-2]
    if transposed:
        x = x.transpose(-2, -1)

    # Normalize
    x_norm = torch.linalg.norm(x, dim=(-2, -1), keepdim=True) + eps
    x = x / x_norm

    # Newton-Schulz iterations
    for _ in range(num_steps):
        # A = X @ X^T (not X^T @ X)
        a = torch.matmul(x, x.transpose(-2, -1))
        # A^2
        a_squared = torch.matmul(a, a)
        # B = muon_b * A + muon_c * A^2
        b = muon_b * a + muon_c * a_squared
        # X = muon_a * X + B @ X
        x = muon_a * x + torch.matmul(b, x)

    # Restore original orientation
    if transposed:
        x = x.transpose(-2, -1)

    # Scale by 0.2 * sqrt(max(rows, cols))
    scale = 0.2 * torch.sqrt(
        torch.tensor(max(x.shape[-1], x.shape[-2]), dtype=x.dtype, device=x.device)
    )

    return scale * x


def merge_repeated_dims(
    tensor: torch.Tensor,
    dim_annotation: list,
) -> tuple[torch.Tensor, dict | None]:
    """Merge repeated dimensions in a tensor.

    This function identifies dimensions with the same label in dim_annotation
    and merges them into a single dimension. Currently returns the tensor unchanged
    as this is primarily used in the JAX version for handling specific tensor layouts.

    For the PyTorch implementation, we handle simpler cases directly in the Muon
    optimizer without needing complex dimension merging.

    Reference: JAX implementation lines 303-389 in optimizers.py

    Args:
        tensor: Input tensor
        dim_annotation: List of dimension labels

    Returns:
        Tuple of (tensor, recipe) where recipe can be used to reconstruct original shape
    """
    # For PyTorch implementation, we don't need complex dimension merging
    # as we handle tensors more directly. Return tensor unchanged.
    return tensor, None


def reconstruct_from_merged(
    merged_tensor: torch.Tensor,
    recipe: dict | None,
) -> torch.Tensor:
    """Reconstruct original tensor shape from merged tensor.

    Companion function to merge_repeated_dims. Since we don't perform merging
    in the PyTorch implementation, this simply returns the tensor unchanged.

    Reference: JAX implementation lines 381-389 in optimizers.py

    Args:
        merged_tensor: Tensor with merged dimensions
        recipe: Recipe dictionary from merge_repeated_dims

    Returns:
        Reconstructed tensor
    """
    if recipe is None:
        return merged_tensor
    # In PyTorch implementation, no reconstruction needed
    return merged_tensor
