"""Utility functions for downloading pre-trained weights."""

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
        self._directory = Path(directory)

    def download_pytorch(self, arch_type: str | ArchType) -> Path:
        """Downloads pre-trained PyTorch weights."""
        arch_type = ArchType(arch_type)
        self._create_dest_path()

        pt_file = self._directory.joinpath(arch_type.lower()).with_suffix(".pt")
        pt_file_url = PRETRAINED_URLS[BackendType.PYTORCH][arch_type]
        gdown.cached_download(pt_file_url, pt_file)

        return pt_file

    def download_openvino(self, arch_type: str | ArchType) -> tuple[Path, Path]:
        """Downloads pre-trained OpenVINO weights."""
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
        self._directory.mkdir(parents=True, exist_ok=True)
