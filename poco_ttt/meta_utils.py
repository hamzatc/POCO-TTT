"""
Utility functions for meta-learning in POCO-TTT
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


def clone_params(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clone a dictionary of parameters.

    Args:
        params: Dictionary mapping parameter names to tensors

    Returns:
        New dictionary with cloned tensors
    """
    return {k: v.clone() for k, v in params.items()}


def get_param_dict(module: nn.Module, param_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """
    Get a dictionary of named parameters from a module.

    Args:
        module: PyTorch module
        param_names: Optional list of parameter name patterns to include.
                    If None, includes all parameters.

    Returns:
        Dictionary mapping parameter names to tensors
    """
    if param_names is None:
        return dict(module.named_parameters())

    return {
        name: param
        for name, param in module.named_parameters()
        if any(pattern in name for pattern in param_names)
    }


def compute_loss_with_params(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    input_data: Tuple,
    criterion: nn.Module,
) -> torch.Tensor:
    """
    Compute loss using substituted parameters via functional_call.

    Args:
        model: The model to use
        params: Dictionary of parameters to substitute
        input_data: Tuple of (input_list, target_list, info_list)
        criterion: Loss function

    Returns:
        Loss tensor
    """
    # This is a placeholder - actual implementation depends on model structure
    # For POCO models, we use the standard forward pass
    input_list, target_list, _ = input_data

    # Forward pass
    output = model(input_list, disable_unit_dropout=True)

    # Compute loss
    total_loss = torch.zeros(1, device=input_list[0].device)
    total_count = 0

    for out, tar in zip(output, target_list):
        if tar.numel() == 0:
            continue
        # Use last pred_length frames
        pred_len = out.shape[0]
        out_pred = out[-pred_len:]
        tar_pred = tar[-pred_len:]
        total_loss += criterion(out_pred, tar_pred) * tar_pred.numel()
        total_count += tar_pred.numel()

    if total_count > 0:
        return total_loss / total_count
    return total_loss


def update_params_with_grads(
    params: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
    lr: float,
) -> Dict[str, torch.Tensor]:
    """
    Update parameters using gradients (SGD step).

    Args:
        params: Current parameter values
        grads: Gradients for each parameter
        lr: Learning rate

    Returns:
        Updated parameter dictionary
    """
    return {
        k: params[k] - lr * grads[k]
        for k in params.keys()
        if k in grads
    }


def accumulate_grads(
    accumulated: Dict[str, torch.Tensor],
    new_grads: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Accumulate gradients across meta-batch.

    Args:
        accumulated: Previously accumulated gradients (or None)
        new_grads: New gradients to add

    Returns:
        Updated accumulated gradients
    """
    if accumulated is None:
        return {k: g.clone() for k, g in new_grads.items()}

    return {
        k: accumulated.get(k, torch.zeros_like(g)) + g
        for k, g in new_grads.items()
    }
