from functools import lru_cache
from inspect import signature
from os import PathLike
from pathlib import Path

import gdown
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from resnet.network.arch import ArchType, SEResNet, init_se_resnet
from resnet.utils.transforms import eval_transform

PRETRAINED_URLS: dict[ArchType, str] = {
    ArchType.SEResNet2: "",
    ArchType.SEResNet3: "",
    ArchType.SEResNet3: "",
}

CLASS_TO_IDX: dict[str, int] = {
    "Acura": 0,
    "Alfa Romeo": 1,
    "Audi": 2,
    "BMW": 3,
    "Bentley": 4,
    "Bugatti Veyron": 5,
    "Cadillac Escalade": 6,
    "Cadillac SRX": 7,
    "Chevrolet Camaro": 8,
    "Chevrolet Corvette": 9,
    "Dodge": 10,
    "Ferrari": 11,
    "Ford Mustang": 12,
    "Hyundai": 13,
    "Jeep Liberty": 14,
    "Kia": 15,
    "Lamborghini": 16,
    "Lexus": 17,
    "Maserati": 18,
    "Mercedes": 19,
    "Mitsubishi Lancer": 20,
    "Porsche": 21,
    "Renault": 22,
    "Rolls Royce": 23,
    "Toyota": 24,
}


def model_cache(func):
    wrapper = lru_cache(maxsize=1)(func)
    wrapper.__signature__ = signature(func)  # type: ignore
    return wrapper


@model_cache
def load_se_resnet(arch_type: str | ArchType, weights_file: str | PathLike | None = None) -> SEResNet:
    """Loads pre-trained SE-ResNet model."""
    model = init_se_resnet(arch_type, len(CLASS_TO_IDX))
    if weights_file is None:
        weights_file = download_pretrained_weights(arch_type)
    weights = torch.load(weights_file, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(weights)
    return model.eval()


def download_pretrained_weights(arch_type: str | ArchType) -> PathLike:
    """Downloads pre-trained weights for the specified architecture."""
    arch_type = ArchType(arch_type)
    weights_file = Path(f"{arch_type.lower()}-weights").with_suffix(".pt")
    if not weights_file.is_file():
        gdown.download(PRETRAINED_URLS[arch_type], str(weights_file), quiet=False)
    return weights_file


def predict(image: ArrayLike, model: SEResNet, topk: int = 5) -> list[dict[str, float]]:
    processed_img = preprocess(image, model.architecture.INPUT_SHAPE)
    processed_img = processed_img.unsqueeze(0)
    with torch.inference_mode():
        logits = model(processed_img)
    return postprocess(logits, topk)


def preprocess(image: ArrayLike, output_shape: tuple[int, int]) -> Tensor:
    img = np.asarray(image, dtype=np.uint8)
    transform = eval_transform(output_shape)
    return transform(img)  # type: ignore


def postprocess(logits: Tensor, topk: int = 5) -> list[dict[str, float]]:
    n_classes = len(CLASS_TO_IDX)
    topk = n_classes if topk > n_classes else topk

    probs = torch.softmax(logits, dim=1)
    top_ids = torch.topk(probs, topk).indices.numpy()
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    classes_with_probs: list[dict[str, float]] = list()
    for sample_probs, sample_top_ids in zip(probs, top_ids):
        for idx in sample_top_ids:
            cls = idx_to_class[idx]
            conf = f"{sample_probs[idx].item():.3f}"
            classes_with_probs.append({cls: float(conf)})

    return classes_with_probs
