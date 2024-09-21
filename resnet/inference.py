from functools import lru_cache
from os import PathLike
from pathlib import Path

import gdown
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

from resnet.network.arch import ArchType, SeResNet, get_se_resnet_arch, init_se_resnet
from resnet.utils.transforms import eval_transform

PRETRAINED_URLS: dict[ArchType, str] = {
    ArchType.SeResNet2SR: "",
    ArchType.SeResNet3SR: "https://drive.google.com/uc?export=download&id=1Z_2117kTcOljktYtG9zd69gtyE_KQfnj",
    ArchType.SeResNet2SR: "",
}

CLASS_TO_IDX: dict[str, int] = {
    "Bike": 0,
    "Car": 1,
    "Motorcycle": 2,
    "Plane": 3,
    "Ship": 4,
    "Train": 5,
}


def download_pretrained_weights(arch_type: str | ArchType) -> PathLike:
    arch_type = ArchType(arch_type)
    weights_file = Path(f"{arch_type.lower()}-weights").with_suffix(".pt")
    if not weights_file.is_file():
        gdown.download(PRETRAINED_URLS[arch_type], str(weights_file), quiet=False)
    return weights_file


@lru_cache(maxsize=1)
def load_se_resnet(arch_type: str | ArchType, weights_file: str | PathLike | None = None) -> SeResNet:
    arch_type = ArchType(arch_type)
    model = init_se_resnet(len(CLASS_TO_IDX), get_se_resnet_arch(arch_type))
    if weights_file is None:
        weights_file = download_pretrained_weights(arch_type)
    weights = torch.load(weights_file, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(weights)
    return model.eval()


def predict(image: ArrayLike, model: SeResNet, topk: int = 5) -> list[dict[str, float]]:
    processed_img = preprocess(image, model.input_shape)
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
