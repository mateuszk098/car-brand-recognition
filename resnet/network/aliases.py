"""Type aliases for the SEResNet network."""

from typing import TypeAlias

# Each alias represents the first parameters in init signature of nn.Module modules.

# Output channels, kernel size, stride, padding
LazyConv: TypeAlias = tuple[int, int, int, int]
# Kernel size, stride
MaxPool: TypeAlias = tuple[int, int]
# Pool size
DepthPool: TypeAlias = tuple[int]
# Output channels, kernel size, stride, squeeze factor, squeeze active
SEResBlock: TypeAlias = tuple[int, int, int, int, bool]

# Input shape (height, width)
InputShape: TypeAlias = tuple[int, int]
Shrinkage: TypeAlias = list[tuple[LazyConv, MaxPool]]
Residuals: TypeAlias = list[tuple[SEResBlock, SEResBlock, DepthPool, MaxPool]]
# Output features, dropout rate.
Neck: TypeAlias = list[tuple[int, float]]
