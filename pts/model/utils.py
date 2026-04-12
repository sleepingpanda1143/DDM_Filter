# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import inspect
from typing import Optional

import torch
import torch.nn as nn


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    return [k for k, v in params.items() if not str(v).startswith("*")]


def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    return x.mean(dim=dim)
