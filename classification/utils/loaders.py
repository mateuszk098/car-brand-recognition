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
        if self.train_transform is None:
            raise ValueError("Train transform is not set.")
        self.dataset.transform = self.train_transform

    def eval(self) -> None:
        if self.eval_transform is None:
            raise ValueError("Eval transform is not set.")
        self.dataset.transform = self.eval_transform
