"""
This module provides functionality to load and run inference using SE-ResNet models
with either PyTorch or OpenVINO backends. It includes utilities for model conversion
and preprocessing, as well as a unified interface for car brand classification.

Classes:
    - PyTorchBackend: Native PyTorch backend for inference.
    - OpenVINOBackend: OpenVINO backend for inference.
    - CarClassifier: Car brand classifier with support for multiple backends.
    
Functions:
    - load_se_resnet(): Loads pre-trained SE-ResNet model.
    - torch2openvino(): Converts PyTorch model to OpenVINO format.
"""

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
    """
    Loads a pre-trained SE-ResNet model.
    Args:
        arch_type (str | ArchType): The architecture type of the SE-ResNet model.
        weights (str | PathLike): The path to the pre-trained weights file.
    Returns:
        The SE-ResNet model loaded with the specified pre-trained weights.
    """
    model = init_se_resnet(arch_type, len(CLASS_TO_IDX))
    state_dict = torch.load(weights, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()


def torch2openvino(arch_type: str | ArchType, weights: str | PathLike, batch_size: int = -1) -> None:
    """
    Converts a PyTorch model to OpenVINO format.
    Args:
        arch_type (str | ArchType): The architecture type of the model.
        weights (str | PathLike): The path to the model weights file.
        batch_size (int, optional): The batch size. Defaults to -1, which indicates a dynamic batch size.
    """
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
        """
        Initializes the model with the specified architecture type and weights.
        Args:
            arch_type (str | ArchType): The type of architecture to use for the model.
            weights (str | PathLike): The path to the weights file to load into the model.
        """
        self._model = load_se_resnet(arch_type, weights)
        self._input_shape = self.model.architecture.INPUT_SHAPE

    def __call__(self, x: Tensor) -> Tensor:
        """
        Invokes the forward method on the input tensor.
        Args:
            x (Tensor): The input tensor to be processed.
        Returns:
            The output tensor after processing.
        """
        return self.forward(x)

    @property
    def model(self) -> SEResNet:
        return self._model

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the model and return the output tensor on the CPU.
        Args:
            x (Tensor): The input tensor to be passed through the model.
        Returns:
            The output tensor after being processed by the model, moved to the CPU.
        """
        return self.model.forward(x).cpu()


class OpenVINOBackend:
    """OpenVINO backend for inference."""

    def __init__(self, model: str | PathLike, weights: str | PathLike) -> None:
        """
        Initialize the inference model with the given model and weights.
        Args:
            model (str | PathLike): Path to the model file.
            weights (str | PathLike): Path to the weights file.
        """
        self._model = self._init_model(model, weights)
        # Last two dimensions are the image dimensions - always constant.
        self._input_shape = tuple(self.model.input().partial_shape[-2:].to_shape())

    def __call__(self, x: Tensor) -> Tensor:
        """
        Invokes the forward method on the given input tensor.
        Args:
            x (Tensor): The input tensor to be processed.
        Returns:
            The output tensor after applying the forward method.
        """
        return self.forward(x)

    @property
    def model(self) -> CompiledModel:
        return self._model

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (Tensor): Input tensor to be passed through the model.
        Returns:
            Output tensor obtained from the model's forward pass.
        """
        return torch.from_numpy(self.model(x)[0])

    def _init_model(self, model: str | PathLike, weights: str | PathLike) -> CompiledModel:
        """
        Initializes and compiles an OpenVINO model.
        Args:
            model (str | PathLike): The path to the model file.
            weights (str | PathLike): The path to the weights file.
        Returns:
            The compiled OpenVINO model ready for inference.
        """
        core = ov.Core()
        ov_model = core.read_model(Path(model), Path(weights))
        ov_model = core.compile_model(ov_model, "CPU")
        return ov_model


class CarClassifier:
    """Car brand classifier with support for multiple backends."""

    def __init__(self, arch_type: str | ArchType, backend_type: str | BackendType) -> None:
        """
        Initializes the MultiBackend class with the specified architecture and backend types.
        Args:
            arch_type (str | ArchType): The architecture type, either as a string or an ArchType enum.
            backend_type (str | BackendType): The backend type, either as a string or a BackendType enum.
        """
        self._arch_type = ArchType(arch_type)
        self._backend_type = BackendType(backend_type)
        self._downloader = WeightsDownloader()
        self._backend = self._get_backend()

    def __call__(self, image: ArrayLike, topk: int = 5) -> list[tuple[str, float]]:
        """
        Call the predict method on the given image.
        Args:
            image (ArrayLike): The input image to be processed.
            topk (int, optional): The number of top predictions to return. Defaults to 5.
        Returns:
            A list of tuples containing the predicted class labels and their corresponding probabilities.
        """
        return self.predict(image, topk)

    @property
    def backend(self) -> PyTorchBackend | OpenVINOBackend:
        return self._backend

    def predict(self, image: ArrayLike, topk: int = 5) -> list[tuple[str, float]]:
        """
        Predicts the car brand from the given image.
        Args:
            image (ArrayLike): The input image for which the car brand needs to be predicted.
            topk (int, optional): The number of top predictions to return. Defaults to 5.
        Returns:
            A list of tuples where each tuple contains a class label and its corresponding probability.
        """
        topk = self._check_topk(topk)
        processed_img = self._preprocess(image)
        logits = self.backend(processed_img)
        return self._postprocess(logits, topk)

    def _preprocess(self, image: ArrayLike) -> Tensor:
        """
        Preprocesses the image for inference.
        Args:
            image (ArrayLike): The input image to be preprocessed.
        Returns:
            The preprocessed image tensor ready for inference.
        """
        img = np.asarray(image, dtype=np.uint8)
        transform = eval_transform(self.backend.input_shape)
        return transform(img).unsqueeze(0)  # type: ignore

    def _postprocess(self, logits: Tensor, topk: int = 5) -> list[tuple[str, float]]:
        """
        Post-processes the logits to return the top-k classes with their probabilities.
        Args:
            logits (Tensor): The logits output from the model.
            topk (int, optional): The number of top classes to return. Defaults to 5.
        Returns:
            A list of tuples where each tuple contains a class label and its corresponding probability.
        """
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
        """
        Initializes and returns the appropriate backend based on the backend type.
        Returns:
            An instance of the backend class corresponding to the backend type.
        Raises:
            ValueError: If the backend type is unknown.
        """
        if self._backend_type == BackendType.PYTORCH:
            weights = self._downloader.download_pytorch(self._arch_type)
            return PyTorchBackend(self._arch_type, weights)

        if self._backend_type == BackendType.OPENVINO:
            weights, model = self._downloader.download_openvino(self._arch_type)
            return OpenVINOBackend(model, weights)

        raise ValueError(f"Unknown backend type: {self._backend_type!r}")

    def _check_topk(self, topk: int) -> int:
        """
        Checks if the provided topk value is valid.
        This method ensures that the topk value is greater than 0 and less than or equal to the
        number of classes. If the topk value is greater than the number of classes, it returns
        the number of classes.
        Args:
            topk (int): The topk value to be checked.
        Returns:
            The valid topk value, which is either the provided topk
                or the number of classes, whichever is smaller.
        Raises:
            ValueError: If the topk value is less than 1.
        """
        if topk < 1:
            raise ValueError("Topk must be greater than 0.")
        n_classes = len(CLASS_TO_IDX)
        return n_classes if topk > n_classes else topk
