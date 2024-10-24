"""
This module defines the structure and handling of configuration files for the vehicle 
recognition system. It includes classes and methods to represent and load various
configuration files.

Classes:
    - FileDetails: Represents configuration file details, including its name and root directory.
    - ConfigFile: Enum representing available configuration files with methods to load their content.
    
Usage:
    - Use `ConfigFile.<CONFIG_NAME>.load()` to load the content of a specific configuration file.
    - Use `ConfigFile.content()` to get a set of all available configuration files.
"""

from dataclasses import dataclass
from enum import Enum, unique
from importlib.resources import files
from typing import Any, Self

import yaml


@dataclass(frozen=True, kw_only=True)
class FileDetails:
    """Represents configuration file details."""

    NAME: str
    ROOT: str = "resnet.config"


@unique
class ConfigFile(Enum):
    """Represents available configuration files."""

    ARCH = FileDetails(NAME="arch.yaml")
    LOGGING = FileDetails(NAME="logging.yaml")
    TRAIN = FileDetails(NAME="train.yaml")
    EVAL = FileDetails(NAME="eval.yaml")
    CLASSES = FileDetails(NAME="classes.yaml")

    def __init__(self, file_data: FileDetails) -> None:
        """
        Initializes the Files class with the given file data.
        Args:
            file_data (FileDetails): An instance of FileDetails containing information about the file.
        """
        self.file_data = file_data

    def load(self) -> dict[str, Any]:
        """
        Load the YAML configuration file specified by the file_data attributes.
        Returns:
            A dictionary containing the configuration data loaded from the YAML file.
        """
        path = str(files(self.file_data.ROOT).joinpath(self.file_data.NAME))
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def content(cls) -> set[Self]:
        """
        Returns a set of all available configuration files.
        This method iterates over all members of the class and collects them into a set.
        Returns:
            A set containing all members of the class.
        """
        return set(member for member in cls)
