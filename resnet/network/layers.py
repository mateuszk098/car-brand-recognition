"""
This module defines custom neural network layers for vehicle recognition using PyTorch.

Classes:
    - MaxDepthPool2d (nn.Module)
    - LazySqueezeExcitation (nn.Module)
    - LazyResidualBlock (nn.Module)
    - LazySEResidualBlock (nn.Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity, Sequential


class MaxDepthPool2d(nn.Module):
    """Max pooling op over the channel dimension."""

    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.pool_size = int(pool_size)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        old_shape = x.shape
        new_shape = (old_shape[0], old_shape[1] // self.pool_size, self.pool_size, *old_shape[2:])
        return torch.amax(x.view(new_shape), dim=2)


class LazySqueezeExcitation(nn.Module):
    """Lazy squeeze and excitation block."""

    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = int(factor)
        self.feed_forward: Sequential | None = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.feed_forward is None:
            self.feed_forward = self._create_feed_forward(x.shape[1], x.shape[1] // self.factor)
        return x * self.feed_forward(x).view(x.shape[0], x.shape[1], 1, 1)

    def _create_feed_forward(self, in_channels: int, squeezed_channels: int) -> Sequential:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, squeezed_channels),
            nn.Mish(),
            nn.Linear(squeezed_channels, in_channels),
            nn.Sigmoid(),
        )


class LazyResidualBlock(nn.Module):
    """Lazy residual connection block."""

    def __init__(self, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.padding = int(kernel_size // 2)
        self.stride = int(stride)
        self.feed_forward = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride, self.padding, bias=False),
            nn.LazyBatchNorm2d(),
            nn.Mish(),
            nn.LazyConv2d(out_channels, kernel_size, 1, self.padding, bias=False),
            nn.LazyBatchNorm2d(),
        )
        self.shortcut: Identity | Sequential | None = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        x_residual = self.feed_forward(x)
        if self.shortcut is None:
            self.shortcut = self._create_shortcut(x.shape[1], x_residual.shape[1], self.stride)
        x_shortcut = self.shortcut(x)
        return F.mish(x_residual + x_shortcut)

    def _create_shortcut(self, in_channels: int, out_channels: int, stride: int) -> Sequential | Identity:
        shortcut_connection = nn.Identity()
        if in_channels != out_channels or stride > 1:
            return nn.Sequential(nn.LazyConv2d(out_channels, 1, stride, bias=False), nn.LazyBatchNorm2d())
        return shortcut_connection


class LazySEResidualBlock(nn.Module):
    """Lazy residual connection with squeeze and excitation block."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        squeeze_factor: int = 8,
        squeeze_active: bool = True,
    ) -> None:
        super().__init__()
        self.stride = int(stride)
        self.squeeze_active = bool(squeeze_active)
        self.residual_block = LazyResidualBlock(out_channels, kernel_size, stride)
        self.squeeze_excitation = LazySqueezeExcitation(squeeze_factor)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        x_residual = self.residual_block.feed_forward(x)
        if self.residual_block.shortcut is None:
            self.residual_block.shortcut = self.residual_block._create_shortcut(
                x.shape[1], x_residual.shape[1], self.stride
            )
        x_shortcut = self.residual_block.shortcut(x)
        residual_output = F.mish(x_residual + x_shortcut)
        if self.squeeze_active:
            return x_shortcut + self.squeeze_excitation(residual_output)
        return residual_output
