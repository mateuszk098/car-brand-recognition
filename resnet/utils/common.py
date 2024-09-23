"""Common utilities for the package."""

import logging
import logging.config
import os
from enum import StrEnum, unique
from logging import Logger
from os import PathLike
from typing import Any, TypeAlias

import yaml

from ..config import ConfigFile


@unique
class RecordedStats(StrEnum):
    """Available losses and metrics to monitor during training."""

    TRAIN_LOSS = "train_loss"
    TRAIN_ACCURACY = "train_accuracy"
    VAL_LOSS = "val_loss"
    VAL_ACCURACY = "val_accuracy"

    @classmethod
    def content(cls) -> set[str]:
        return set(member for member in cls)


History: TypeAlias = dict[RecordedStats, list[float]]


def remove_file(file: str | PathLike) -> None:
    """Removes a file if it exists. Otherwise, do nothing."""
    try:
        os.remove(file)
    except FileNotFoundError:
        pass


def load_yaml(file: str | PathLike) -> dict[str, Any]:
    """Loads YAML file and return its content."""
    with open(file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def init_logger(name: str | None = None) -> Logger:
    """Initializes a logger with the specified name."""
    content = ConfigFile.LOGGING.load()
    logging.config.dictConfig(content)
    return logging.getLogger(name)
