from numpy.typing import NDArray
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms


class VehicleTransform:
    def __init__(
        self,
        size: tuple[int, int],
        train: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.train = bool(train)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(contrast=0.2, saturation=0.2, brightness=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image | NDArray) -> Tensor:
        if self.train:
            return self.train_transform(image)  # type: ignore
        return self.eval_transform(image)  # type: ignore
