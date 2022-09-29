from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


def weighted_l1(inputs: Tensor,
                targets: Tensor,
                weights: Optional[Tensor] = None,
                normalize_weights: bool = True,
                reduction: str = 'mean',
                ignore_value: Optional[Union[int, float]] = None) -> Tensor:
    assert reduction in ['mean', 'sum', 'none'], "Invalid reduction"
    if weights is None:
        return F.l1_loss(inputs, targets, reduction=reduction)
    if normalize_weights:
        weights = weights / torch.sum(weights)
    wl1 = weights * torch.abs(inputs - targets)
    if ignore_value is not None:
        wl1[targets == ignore_value] = 0.0
    res = torch.sum(wl1, dim=-1)
    if reduction == 'mean':
        return res.mean()
    elif reduction == 'sum':
        return res.sum()
    else:  # reduction == 'none'
        return res


def weighted_mse(inputs: Tensor,
                 targets: Tensor,
                 weights: Optional[Tensor] = None,
                 normalize_weights: bool = True,
                 reduction: str = 'mean',
                 ignore_value: Optional[Union[int, float]] = None) -> Tensor:
    assert reduction in ['mean', 'sum', 'none'], "Invalid reduction"
    if weights is None:
        return F.mse_loss(inputs, targets, reduction=reduction)
    if normalize_weights:
        weights = weights / torch.sum(weights)
    wmse = weights * ((inputs - targets) ** 2)
    if ignore_value is not None:
        wmse[targets == ignore_value] = 0.0
    res = torch.sum(wmse, dim=-1)
    if reduction == 'mean':
        return res.mean()
    elif reduction == 'sum':
        return res.sum()
    else:  # reduction == 'none'
        return res


class WeightedL1Loss(_WeightedLoss):

    def __init__(self,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 normalize_weights: bool = True,
                 ignore_value: Optional[Union[float, int]] = None) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.normalize_weights = normalize_weights
        self.ignore_value = ignore_value

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return weighted_l1(inputs,
                           targets,
                           weights=self.weight,
                           normalize_weights=self.normalize_weights,
                           reduction=self.reduction,
                           ignore_value=self.ignore_value)


class WeightedMSELoss(_WeightedLoss):

    def __init__(self,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 normalize_weights: bool = True,
                 ignore_value: Optional[Union[float, int]] = None) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.normalize_weights = normalize_weights
        self.ignore_value = ignore_value

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return weighted_mse(inputs,
                            targets,
                            weights=self.weight,
                            normalize_weights=self.normalize_weights,
                            reduction=self.reduction,
                            ignore_value=self.ignore_value)
