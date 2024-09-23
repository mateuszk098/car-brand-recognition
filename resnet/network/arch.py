"""Squeeze and Excitation ResNet architecture for classification tasks."""

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Self

import torch
import torch.nn as nn
import torchinfo
from torch import Tensor
from torch.nn import Module

from ..config import ConfigFile
from .aliases import InputShape, Neck, Residuals, Shrinkage
from .layers import LazySEResidualBlock, MaxDepthPool2d


@unique
class ArchType(StrEnum):
    """Available architecture types for the Squeeze and Excitation ResNet model."""

    SEResNet2 = "SEResNet2"
    SEResNet3 = "SEResNet3"
    SEResNet4 = "SEResNet4"

    @classmethod
    def content(cls) -> set[str]:
        """Returns a set of all available architecture types."""
        return set(member for member in cls)


@dataclass(frozen=True, kw_only=True)
class SEResNetArch:
    """Represents the architecture of the Squeeze and Excitation ResNet model."""

    INPUT_SHAPE: InputShape
    SHRINKAGE: Shrinkage
    RESIDUALS: Residuals
    NECK: Neck


class SEResNet(nn.Module):
    """Squeeze and Excitation Residual Network for classification tasks."""

    def __init__(self, num_classes: int, arch: SEResNetArch) -> None:
        super().__init__()
        self._architecture = arch

        shrinkage = nn.Sequential()
        residuals = nn.Sequential()
        neck = nn.Sequential()
        flatten = nn.Flatten()
        classifier = nn.LazyLinear(num_classes)

        for conv_params, pool_params in arch.SHRINKAGE:
            shrinkage.append(
                nn.Sequential(
                    nn.LazyConv2d(*conv_params),
                    nn.LazyBatchNorm2d(),
                    nn.Mish(),
                    nn.MaxPool2d(*pool_params),
                ),
            )

        for res_params1, res_params2, depth_pool_params, max_pool_params in arch.RESIDUALS:
            residuals.append(
                nn.Sequential(
                    LazySEResidualBlock(*res_params1),
                    LazySEResidualBlock(*res_params2),
                    MaxDepthPool2d(*depth_pool_params),
                    nn.MaxPool2d(*max_pool_params),
                )
            )

        for units, dropout in arch.NECK:
            neck.append(
                nn.Sequential(
                    nn.LazyLinear(units, bias=False),
                    nn.LazyBatchNorm1d(),
                    nn.Mish(),
                    nn.Dropout1d(dropout),
                ),
            )

        self.feed_forward = nn.Sequential(shrinkage, residuals, flatten, neck, classifier)

    def __call__(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)

    @property
    def architecture(self) -> SEResNetArch:
        return self._architecture

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)

    def warmup(self) -> Self:
        input_shape = self.architecture.INPUT_SHAPE
        x = torch.randn(10, 3, *input_shape, generator=torch.manual_seed(42))
        self.feed_forward(x)
        return self


def get_se_resnet_arch(arch_type: str | ArchType) -> SEResNetArch:
    arch_type = ArchType(arch_type)
    arch = ConfigFile.ARCH.load().get(arch_type)
    return SEResNetArch(**arch)


def init_se_resnet(arch_type: str | ArchType, num_classes: int) -> SEResNet:
    model = SEResNet(num_classes, get_se_resnet_arch(arch_type))
    return model.warmup()


def arch_summary(model: Module) -> str:
    """Returns the architecture summary of the model."""
    return str(torchinfo.summary(model, verbose=0, depth=4))
