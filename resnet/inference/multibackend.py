"""Inference module with support for multiple backends PyTorch and OpenVINO."""

from os import PathLike
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from numpy.typing import ArrayLike
from openvino import CompiledModel
from torch import Tensor

from ..config import ConfigFile
from ..network.arch import ArchType, SEResNet, init_se_resnet
from ..utils.transforms import eval_transform
from .utils import BackendType, WeightsDownloader

CLASS_TO_IDX: dict[str, int] = ConfigFile.CLASSES.load()


def load_se_resnet(arch_type: str | ArchType, weights: str | PathLike) -> SEResNet:
    """Loads pre-trained SE-ResNet model."""
    model = init_se_resnet(arch_type, len(CLASS_TO_IDX))
    state_dict = torch.load(weights, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()


def torch2openvino(arch_type: str | ArchType, weights: str | PathLike, batch_size: int = -1) -> None:
    """Converts PyTorch model to OpenVINO format."""
    model = load_se_resnet(arch_type, weights)
    input_shape = model.architecture.INPUT_SHAPE
    example_input = torch.randn(1, 3, *input_shape)
    traced_model = torch.jit.trace(model, example_input)
    ov_model = ov.convert_model(
        traced_model,
        input=[batch_size, 3, *input_shape],  # Dynamic when batch_size=-1.
        example_input=example_input,
    )
    ov.save_model(ov_model, Path(weights).with_suffix(".xml"))


class PyTorchBackend:
    """Native PyTorch backend for inference."""

    def __init__(self, arch_type: str | ArchType, weights: str | PathLike) -> None:
        self._model = load_se_resnet(arch_type, weights)
        self._input_shape = self.model.architecture.INPUT_SHAPE

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def model(self) -> SEResNet:
        return self._model

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x).cpu()


class OpenVINOBackend:
    """OpenVINO backend for inference."""

    def __init__(self, model: str | PathLike, weights: str | PathLike) -> None:
        self._model = self._init_model(model, weights)
        # Last two dimensions are the image dimensions - always constant.
        self._input_shape = tuple(self.model.input().partial_shape[-2:].to_shape())

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def model(self) -> CompiledModel:
        return self._model

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    def forward(self, x: Tensor) -> Tensor:
        return torch.from_numpy(self.model(x)[0])

    def _init_model(self, model: str | PathLike, weights: str | PathLike) -> CompiledModel:
        core = ov.Core()
        ov_model = core.read_model(Path(model), Path(weights))
        ov_model = core.compile_model(ov_model, "CPU")
        return ov_model


class CarClassifier:
    """Car brand classifier with support for multiple backends."""

    def __init__(self, arch_type: str | ArchType, backend_type: str | BackendType) -> None:
        self._arch_type = ArchType(arch_type)
        self._backend_type = BackendType(backend_type)
        self._downloader = WeightsDownloader()
        self._backend = self._get_backend()

    def __call__(self, image: ArrayLike, topk: int = 5) -> list[tuple[str, float]]:
        return self.predict(image, topk)

    @property
    def backend(self) -> PyTorchBackend | OpenVINOBackend:
        return self._backend

    def predict(self, image: ArrayLike, topk: int = 5) -> list[tuple[str, float]]:
        """Predicts the car brand from the image."""
        topk = self._check_topk(topk)
        processed_img = self._preprocess(image)
        logits = self.backend(processed_img)
        return self._postprocess(logits, topk)

    def _preprocess(self, image: ArrayLike) -> Tensor:
        """Preprocesses the image for inference."""
        img = np.asarray(image, dtype=np.uint8)
        transform = eval_transform(self.backend.input_shape)
        return transform(img).unsqueeze(0)  # type: ignore

    def _postprocess(self, logits: Tensor, topk: int = 5) -> list[tuple[str, float]]:
        """Postprocesses the logits to get the topk classes with probabilities."""
        probs = torch.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, topk).indices.numpy()
        idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

        classes_with_probs: list[tuple[str, float]] = list()
        for sample_probs, sample_top_ids in zip(probs, top_ids):
            for idx in sample_top_ids:
                cls = idx_to_class[idx]
                conf = f"{sample_probs[idx].item():.3f}"
                classes_with_probs.append((cls, float(conf)))

        return classes_with_probs

    def _get_backend(self) -> PyTorchBackend | OpenVINOBackend:
        """Initializes the backend based on the backend type."""
        if self._backend_type == BackendType.PYTORCH:
            weights = self._downloader.download_pytorch(self._arch_type)
            return PyTorchBackend(self._arch_type, weights)

        if self._backend_type == BackendType.OPENVINO:
            weights, model = self._downloader.download_openvino(self._arch_type)
            return OpenVINOBackend(model, weights)

        raise ValueError(f"Unknown backend type: {self._backend_type!r}")

    def _check_topk(self, topk: int) -> int:
        """Checks if topk is valid - greater than 0 and less than the number of classes."""
        if topk < 1:
            raise ValueError("Topk must be greater than 0.")
        n_classes = len(CLASS_TO_IDX)
        return n_classes if topk > n_classes else topk
