"""Custom Data Loaders"""

from typing import Any, Callable, Protocol

from numpy.typing import NDArray
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class SupportsTransform(Protocol):
    # Takes an image or an array and returns a tensor.
    transform: Callable[[Image | NDArray], Tensor]


class VehicleDataLoader(DataLoader):
    """DataLoader that supports setting different transforms for training and evaluation."""

    def __init__(
        self,
        *args: Any,
        train_transform: Compose | None = None,
        eval_transform: Compose | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dataset: SupportsTransform
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def train(self) -> None:
        """Sets train transformation for the dataset."""
        if self.train_transform is None:
            raise ValueError("Train transform is not set.")
        self.dataset.transform = self.train_transform

    def eval(self) -> None:
        """Sets evaluation transformation for the dataset."""
        if self.eval_transform is None:
            raise ValueError("Eval transform is not set.")
        self.dataset.transform = self.eval_transform
