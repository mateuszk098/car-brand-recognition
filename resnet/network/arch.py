"""
This module defines the architecture and initialization of the Squeeze and Excitation ResNet model.

Classes:
    - ArchType (StrEnum): Enum representing available architecture types for 
        the Squeeze and Excitation ResNet model.
    - SEResNetArch (dataclass): Represents the architecture of the Squeeze and Excitation ResNet model.
    - SEResNet (nn.Module): Squeeze and Excitation Residual Network for classification tasks.
    
Functions:
    - get_se_resnet_arch(arch_type: str | ArchType) -> SEResNetArch:
        Returns the architecture of the Squeeze and Excitation ResNet model.
    - init_se_resnet(arch_type: str | ArchType, num_classes: int) -> SEResNet:
        Initializes the Squeeze and Excitation ResNet model.
    - arch_summary(model: Module) -> str:
        Returns the architecture summary of the model.
"""

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
        """
        Returns a set of all available architecture types.
        This method iterates over all members of the class and collects them into a set.
        Returns:
            A set containing all architecture types as strings.
        """
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
        """
        Initializes the network architecture.
        Args:
            num_classes (int): The number of output classes for the classifier.
            arch (SEResNetArch): An instance of SEResNetArch containing the architecture parameters.
        """
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
        """
        Invokes the feed_forward method on the input tensor.
        Args:
            x (Tensor): The input tensor to be processed.
        Returns:
            The output tensor after applying the feed_forward method.
        """
        return self.feed_forward(x)

    @property
    def architecture(self) -> SEResNetArch:
        return self._architecture

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.
        Args:
            x (Tensor): Input tensor to the network.
        Returns:
            Output tensor after applying the feed-forward network.
        """
        return self.feed_forward(x)

    def warmup(self) -> Self:
        """
        Warm up the network by initializing layer dimensions with random data.
        This method passes random data through the network to initialize the dimensions
        of the layers. It uses a fixed seed for reproducibility.
        Returns:
            The instance of the network after warming up.
        """
        input_shape = self.architecture.INPUT_SHAPE
        x = torch.randn(10, 3, *input_shape, generator=torch.manual_seed(42))
        self.feed_forward(x)
        return self


def get_se_resnet_arch(arch_type: str | ArchType) -> SEResNetArch:
    """
    Returns the architecture of the Squeeze and Excitation ResNet model.
    Args:
        arch_type (str | ArchType): The type of architecture to retrieve. This can be either a string
            representing the architecture type or an instance of ArchType.
    Returns:
        An instance of the SEResNetArch class initialized with the architecture configuration.
    """
    arch_type = ArchType(arch_type)
    arch = ConfigFile.ARCH.load().get(arch_type)
    return SEResNetArch(**arch)


def init_se_resnet(arch_type: str | ArchType, num_classes: int) -> SEResNet:
    """
    Initializes the Squeeze and Excitation ResNet (SE-ResNet) model.
    Args:
        arch_type (str | ArchType): The architecture type of the SE-ResNet model.
            It can be a string or an instance of ArchType.
        num_classes (int): The number of output classes for the SE-ResNet model.
    Returns:
        An instance of the SE-ResNet model after performing a warmup.
    """
    model = SEResNet(num_classes, get_se_resnet_arch(arch_type))
    return model.warmup()


def arch_summary(model: Module) -> str:
    """
    Returns the architecture summary of the model.
    Args:
        model (Module): The neural network model to summarize.
    Returns:
        A string representation of the model's architecture summary.
    """
    return str(torchinfo.summary(model, verbose=0, depth=4))
