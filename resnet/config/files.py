"""Configuration files model."""

from dataclasses import dataclass
from enum import Enum, unique
from importlib.resources import files
from typing import Any, Self

import yaml


@dataclass(frozen=True, kw_only=True)
class FileDetails:
    """Represents configuration file details. It's name and root."""

    NAME: str
    ROOT: str = "resnet.config"


@unique
class ConfigFile(Enum):
    """Represents available configuration files. Example usage: `ConfigFile.ARCH.load()`."""

    ARCH = FileDetails(NAME="arch.yaml")
    LOGGING = FileDetails(NAME="logging.yaml")
    TRAIN = FileDetails(NAME="train.yaml")
    EVAL = FileDetails(NAME="eval.yaml")
    CLASSES = FileDetails(NAME="classes.yaml")

    def __init__(self, file_data: FileDetails) -> None:
        self.file_data = file_data

    def load(self) -> dict[str, Any]:
        path = str(files(self.file_data.ROOT).joinpath(self.file_data.NAME))
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def content(cls) -> set[Self]:
        """Returns a set of all available configuration files."""
        return set(member for member in cls)
