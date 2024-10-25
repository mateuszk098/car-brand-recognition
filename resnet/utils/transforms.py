"""
This module provides transformation utilities for training and evaluation of 
images using torchvision.

Functions:
    - train_transform(): Returns a composition of augmentation and normalization 
        transformations for training images.
    - eval_transform(): Returns a composition of normalization transformations 
        for evaluation images.
    
Constants:
    - MEAN: The mean values for normalization.
    - STD: The standard deviation values for normalization.
"""

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
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
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
