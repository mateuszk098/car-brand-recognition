"""
This module provides utilities for downloading pre-trained weights for the 
Squeeze and Excitation ResNet model.

Classes:
    - BackendType: Enum representing available backends for inference.
    - WeightsDownloader: Class for downloading pre-trained weights for the 
        Squeeze and Excitation ResNet model.
"""

from enum import StrEnum, auto
from os import PathLike
from pathlib import Path
from typing import Any

import gdown

from ..network.arch import ArchType


class BackendType(StrEnum):
    """Available backends for inference."""

    PYTORCH = auto()
    OPENVINO = auto()


PRETRAINED_URLS: dict[BackendType, dict[ArchType, Any]] = {
    BackendType.PYTORCH: {
        ArchType.SEResNet2: "",
        ArchType.SEResNet3: "https://drive.google.com/uc?export=download&id=1FdyHb7qY7noFjbMNZUB21baABsk3_mL3",
        ArchType.SEResNet4: "",
    },
    BackendType.OPENVINO: {
        ArchType.SEResNet2: {
            "xml": "",
            "bin": "",
        },
        ArchType.SEResNet3: {
            "xml": "https://drive.google.com/uc?export=download&id=1BAbS6ON7I-PStcLr_phmcvzNYNYVkZB4",
            "bin": "https://drive.google.com/uc?export=download&id=1kCxksv5wMN-x1QMXJ7LxqAScetfMDxir",
        },
        ArchType.SEResNet4: {
            "xml": "",
            "bin": "",
        },
    },
}


class WeightsDownloader:
    """Downloads pre-trained weights for the Squeeze and Excitation ResNet model."""

    def __init__(self, directory: str | PathLike = "./checkpoints/") -> None:
        """
        Initializes the utility class with the specified directory.
        Args:
            directory (str | PathLike, optional): The directory path where checkpoints are stored.
                Defaults to ./checkpoints/.
        """
        self._directory = Path(directory)

    def download_pytorch(self, arch_type: str | ArchType) -> Path:
        """
        Downloads pre-trained PyTorch weights.
        Args:
            arch_type (str | ArchType): The architecture type for which to download the weights.
                It can be a string or an instance of ArchType.
        Returns:
            The path to the downloaded PyTorch weights file.
        """
        arch_type = ArchType(arch_type)
        self._create_dest_path()

        pt_file = self._directory.joinpath(arch_type.lower()).with_suffix(".pt")
        pt_file_url = PRETRAINED_URLS[BackendType.PYTORCH][arch_type]
        gdown.cached_download(pt_file_url, pt_file)

        return pt_file

    def download_openvino(self, arch_type: str | ArchType) -> tuple[Path, Path]:
        """
        Downloads the OpenVINO model files (XML and BIN) for the specified architecture type.
        Args:
            arch_type (str | ArchType): The architecture type for which to download the model files.
        Returns:
            A tuple containing the paths to the downloaded BIN and XML files.
        Raises:
            ValueError: If the provided architecture type is not valid.
        """
        arch_type = ArchType(arch_type)
        self._create_dest_path()

        xml_file = self._directory.joinpath(arch_type.lower()).with_suffix(".xml")
        bin_file = self._directory.joinpath(arch_type.lower()).with_suffix(".bin")

        xml_file_url = PRETRAINED_URLS[BackendType.OPENVINO][arch_type]["xml"]
        bin_file_url = PRETRAINED_URLS[BackendType.OPENVINO][arch_type]["bin"]

        gdown.cached_download(xml_file_url, xml_file)
        gdown.cached_download(bin_file_url, bin_file)

        return bin_file, xml_file

    def _create_dest_path(self) -> None:
        """Creates the destination directory path if it does not already exist."""
        self._directory.mkdir(parents=True, exist_ok=True)
