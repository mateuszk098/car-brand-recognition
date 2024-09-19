"""Common utilities for the package."""

import logging
import logging.config
import os
from enum import StrEnum, unique
from importlib.resources import files
from logging import Logger
from os import PathLike
from typing import Any, TypeAlias

import yaml

from ..config.files import FileData, Files


@unique
class RecordedStats(StrEnum):
    """Available losses and metrics to monitor during training."""

    TRAIN_LOSS = "train_loss"
    TRAIN_ACCURACY = "train_accuracy"
    VAL_LOSS = "val_loss"
    VAL_ACCURACY = "val_accuracy"

    @classmethod
    def values(cls) -> set[str]:
        return set(key.value for key in cls)


def remove_file(file: str | PathLike) -> None:
    """Remove a file if it exists. Otherwise, do nothing."""
    try:
        os.remove(file)
    except FileNotFoundError:
        pass


def load_config(config_file: str | PathLike) -> dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def retrieve_config_file(file: FileData) -> str:
    """Retrieve the path to the specified configuration file."""
    return str(files(file.ROOT).joinpath(file.NAME))


def init_logger(name: str | None = None) -> Logger:
    """Initialize a logger with the specified name."""
    content = load_config(retrieve_config_file(Files.LOGGING))
    logging.config.dictConfig(content)
    return logging.getLogger(name)


History: TypeAlias = dict[str | RecordedStats, list[float]]
