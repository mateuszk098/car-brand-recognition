"""Transforms for vehicle recognition."""

from torchvision import transforms
from torchvision.transforms import Compose

MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def train_transform(img_size: tuple[int, int]) -> Compose:
    """Returns augmentation and normalization transformations for training."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(contrast=0.2, saturation=0.2, brightness=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def eval_transform(img_size: tuple[int, int]) -> Compose:
    """Returns normalization transformations for evaluation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=img_size),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
