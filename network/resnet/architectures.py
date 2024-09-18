from enum import StrEnum, unique
from importlib.resources import files
from types import SimpleNamespace
from typing import Any, Self

import torch
import torch.nn as nn
import torchinfo
from torch import Tensor
from torch.nn import Module

from ..utils.common import load_config
from . import layers


@unique
class ArchType(StrEnum):
    """Available architecture types for the Squeeze and Excitation ResNet model."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @classmethod
    def types(cls) -> set[str]:
        return set(key.value for key in cls)


class SeResNet(nn.Module):
    """Squeeze and Excitation Residual Network for classification tasks."""

    def __init__(self, num_classes: int, arch: SimpleNamespace) -> None:
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


def load_arch() -> dict[str, Any]:
    config_file = files("network.config").joinpath("arch.yaml")
    return load_config(str(config_file))


def init_se_resnet(num_classes: int, arch_type: str | ArchType, pretrained: bool = False) -> SeResNet:
    arch_type = ArchType(arch_type).upper()
    arch = load_arch()[arch_type]
    model = SeResNet(num_classes, SimpleNamespace(**arch))
    return model.warmup()


def architecture_summary(model: Module) -> str:
    """Returns the architecture summary of the model."""
    return str(torchinfo.summary(model, verbose=0, depth=4))
