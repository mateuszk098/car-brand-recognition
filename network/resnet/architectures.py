from typing import Self

import torch
import torch.nn as nn
import torchinfo
from torch import Tensor
from torch.nn import Module

from . import layers


class SeResNet(nn.Module):
    """Squeeze and Excitation Residual Network for classification tasks."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.feed_forward = nn.Sequential(
            #
            nn.LazyConv2d(32, kernel_size=5, stride=2, padding=2),
            nn.LazyBatchNorm2d(),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            layers.LazySeResidualBlock(64, kernel_size=3, stride=1, squeeze_active=True),
            layers.LazySeResidualBlock(64, kernel_size=3, stride=1, squeeze_active=True),
            layers.MaxDepthPool2d(pool_size=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            layers.LazySeResidualBlock(96, kernel_size=5, stride=1, squeeze_active=True),
            layers.LazySeResidualBlock(96, kernel_size=5, stride=1, squeeze_active=True),
            layers.MaxDepthPool2d(pool_size=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            layers.LazySeResidualBlock(128, kernel_size=3, stride=1, squeeze_active=True),
            layers.LazySeResidualBlock(128, kernel_size=3, stride=1, squeeze_active=True),
            layers.MaxDepthPool2d(pool_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Flatten(),
            #
            nn.LazyLinear(256, bias=False),
            nn.LazyBatchNorm1d(),
            nn.Mish(),
            nn.Dropout1d(0.4),
            #
            nn.LazyLinear(256, bias=False),
            nn.LazyBatchNorm1d(),
            nn.Mish(),
            nn.Dropout1d(0.4),
            #
            nn.LazyLinear(num_classes),
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.predict(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)

    def warmup(self, x: Tensor) -> Self:
        self.feed_forward(x)
        return self

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)


def architecture_summary(model: Module) -> str:
    """Returns the architecture summary of the model."""
    return str(torchinfo.summary(model, verbose=0))
