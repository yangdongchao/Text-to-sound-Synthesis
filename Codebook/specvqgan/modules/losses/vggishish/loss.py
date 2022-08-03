import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WeightedCrossEntropy(nn.CrossEntropyLoss):

    def __init__(self, weights, **pytorch_ce_loss_args) -> None:
        super().__init__(reduction='none', **pytorch_ce_loss_args)
        self.weights = weights

    def __call__(self, outputs, targets, to_weight=True):
        loss = super().__call__(outputs, targets)
        if to_weight:
            return (loss * self.weights[targets]).sum() / self.weights[targets].sum()
        else:
            return loss.mean()

""" Binary Cross Entropy w/ a few extras
Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropy(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self, smoothing=0.1, target_threshold: Optional[float] = None, weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        super(BinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (target.size()[0], num_classes),
                off_value,
                device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        return F.binary_cross_entropy_with_logits(
            x, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)



if __name__ == '__main__':
    x = torch.randn(10, 5)
    target = torch.randint(0, 5, (10,))
    weights = torch.tensor([1., 2., 3., 4., 5.])

    # criterion_weighted = nn.CrossEntropyLoss(weight=weights)
    # loss_weighted = criterion_weighted(x, target)

    # criterion_weighted_manual = nn.CrossEntropyLoss(reduction='none')
    # loss_weighted_manual = criterion_weighted_manual(x, target)
    # print(loss_weighted, loss_weighted_manual.mean())
    # loss_weighted_manual = (loss_weighted_manual * weights[target]).sum() / weights[target].sum()
    # print(loss_weighted, loss_weighted_manual)
    # print(torch.allclose(loss_weighted, loss_weighted_manual))

    pytorch_weighted = nn.CrossEntropyLoss(weight=weights)
    pytorch_unweighted = nn.CrossEntropyLoss()
    custom = WeightedCrossEntropy(weights)

    assert torch.allclose(pytorch_weighted(x, target), custom(x, target, to_weight=True))
    assert torch.allclose(pytorch_unweighted(x, target), custom(x, target, to_weight=False))
    print(custom(x, target, to_weight=True), custom(x, target, to_weight=False))
