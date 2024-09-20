from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Self, TypeAlias

import torch
import torch.nn as nn
import torchinfo
from torch import Tensor
from torch.nn import Module

from ..config.files import Files
from ..utils.common import load_config, retrieve_config_file
from . import layers

InputType: TypeAlias = tuple[int, int]
ShrinkageType: TypeAlias = list[
    tuple[
        tuple[int, int, int, int],
        tuple[int, int],
    ]
]
ResidualsType: TypeAlias = list[
    tuple[
        tuple[int, int, int, int, bool],
        tuple[int, int, int, int, bool],
        tuple[int],
        tuple[int, int],
    ]
]
NeckType: TypeAlias = list[tuple[int, int]]


@dataclass(frozen=True, kw_only=True)
class SeResNetArch:
    """Represents the architecture of the Squeeze and Excitation ResNet model."""

    INPUT_SHAPE: InputType
    SHRINKAGE: ShrinkageType
    RESIDUALS: ResidualsType
    NECK: NeckType


@unique
class ArchType(StrEnum):
    """Available architecture types for the Squeeze and Excitation ResNet model."""

    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"

    @classmethod
    def types(cls) -> set[str]:
        return set(key.value for key in cls)


class SeResNet(nn.Module):
    """Squeeze and Excitation Residual Network for classification tasks."""

    def __init__(self, num_classes: int, arch: SeResNetArch) -> None:
        super().__init__()

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
                    layers.LazySeResidualBlock(*res_params1),
                    layers.LazySeResidualBlock(*res_params2),
                    layers.MaxDepthPool2d(*depth_pool_params),
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

        self._input_shape = arch.INPUT_SHAPE
        self.feed_forward = nn.Sequential(shrinkage, residuals, flatten, neck, classifier)

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    def __call__(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)

    def warmup(self) -> Self:
        self.feed_forward(torch.randn(10, 3, *self.input_shape))
        return self


def get_se_resnet_arch(arch_type: str | ArchType) -> SeResNetArch:
    arch_type = ArchType(arch_type.upper())
    architectures = load_config(retrieve_config_file(Files.ARCH))
    return SeResNetArch(**architectures[arch_type])


def init_se_resnet(num_classes: int, arch: SeResNetArch, pretrained: bool = False) -> SeResNet:
    model = SeResNet(num_classes, arch)
    return model.warmup()


def arch_summary(model: Module) -> str:
    """Returns the architecture summary of the model."""
    return str(torchinfo.summary(model, verbose=0, depth=4))
