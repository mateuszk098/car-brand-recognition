"""Callbacks for the training loop."""

import os
from abc import ABCMeta, abstractmethod
from enum import StrEnum, unique
from os import PathLike
from pathlib import Path
from time import strftime
from typing import Any

import torch
from dotenv import find_dotenv, load_dotenv
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler

from .common import History, RecordedStats, init_logger, remove_file
from .visualize import save_learning_curves

assert load_dotenv(find_dotenv()), "The .env file is missing!"

torch.serialization.add_safe_globals([RecordedStats])

logger = init_logger(os.getenv("LOGGER"))


@unique
class Callbacks(StrEnum):
    """Available callbacks for the training loop."""

    EarlyStopping = "EarlyStopping"
    ModelCheckpoint = "ModelCheckpoint"
    LearningCurvesCheckpoint = "LearningCurvesCheckpoint"

    @classmethod
    def keys(cls) -> set[str]:
        return set(key.value for key in cls)


class EarlyStopping:
    """Early stopping callback with the given monitoring value."""

    def __init__(
        self,
        monitor_value: str | RecordedStats = RecordedStats.VAL_LOSS,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor_value = RecordedStats(monitor_value)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.min_value = float("inf")

    def __call__(self, history: History) -> bool:
        return self.should_stop(history)

    def should_stop(self, history: History) -> bool:
        current_values = history[self.monitor_value]
        current_value = 0.0 if not current_values else current_values[-1]

        if current_value < self.min_value:
            self.min_value = current_value
            self.counter = 0
            return False

        if current_value > self.min_value + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


class Checkpoint(metaclass=ABCMeta):
    """Abstract base class for saving and loading checkpoints."""

    def __init__(
        self,
        checkpoints_ext: str,
        checkpoints_type: str,
        checkpoints_dir: str | PathLike,
    ) -> None:
        self.checkpoints_ext = str(checkpoints_ext)
        self.checkpoints_type = str(checkpoints_type)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.run_dir: PathLike | None = None
        self.latest_file: PathLike | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.save(*args, **kwargs)

    @abstractmethod
    def save(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def load(self) -> None:
        checkpoints = self.checkpoints_dir.rglob("*" + self.checkpoints_ext)
        latest_file = max(checkpoints, key=os.path.getctime, default=None)
        if latest_file is None:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoints_dir!s}")

        self.latest_file = latest_file
        self.run_dir = latest_file.parent

    def create_run_storage(self) -> None:
        if self.run_dir is None:
            unique_run = strftime(f"%Y_%m_%d_%H_%M_{self.checkpoints_type}")
            self.run_dir = self.checkpoints_dir / unique_run
            self.run_dir.mkdir(parents=True, exist_ok=True)


class ModelCheckpoint(Checkpoint):
    """Model checkpoint callback for saving and loading model states."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        checkpoints_freq: int = 1,
        checkpoints_dir: str | PathLike = "./checkpoints/",
    ) -> None:
        checkpoints_type = "states"
        checkpoints_ext = ".tar"
        super().__init__(checkpoints_ext, checkpoints_type, checkpoints_dir)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoints_freq = int(checkpoints_freq)
        self.latest_epoch: int = 0
        self.history: History = dict()

    def save(self, history: History) -> None:
        super().create_run_storage()
        if self.run_dir is None:
            raise RuntimeError("Run directory not created.")

        epoch = max((len(lst) for lst in history.values()), default=0)
        if epoch % self.checkpoints_freq == 0:
            checkpoint = Path(f"state_epoch_{epoch:03d}").with_suffix(self.checkpoints_ext)
            current_file = self.run_dir / checkpoint
            logger.debug(f"Saving state to {current_file!s}...")
            torch.save(
                {
                    "epoch": epoch,
                    "history": dict(history),
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                current_file,
            )
            if self.latest_file is not None:
                remove_file(self.latest_file)
            self.latest_file = current_file

    def load(self, map_location: str | torch.device | None = None) -> None:
        super().load()
        if self.latest_file is None:
            raise RuntimeError("Cannot infer run storage.")

        logger.debug(f"Loading state from {self.latest_file!s}...")
        checkpoint = torch.load(self.latest_file, map_location, weights_only=True)
        self.latest_epoch = checkpoint["epoch"]
        self.history = checkpoint["history"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])


class LearningCurvesCheckpoint(Checkpoint):
    """Learning curves checkpoint callback for saving learning curves plots."""

    def __init__(
        self,
        smooth_out: bool = True,
        window: int = 5,
        order: int = 2,
        checkpoints_freq: int = 1,
        checkpoints_dir: str | PathLike = "./checkpoints/",
    ) -> None:
        checkpoints_type = "plots"
        checkpoints_ext = ".png"
        super().__init__(checkpoints_ext, checkpoints_type, checkpoints_dir)

        self.smooth_out = bool(smooth_out)
        self.window = int(window)
        self.order = int(order)
        self.checkpoints_freq = int(checkpoints_freq)

    def save(self, history: History) -> None:
        super().create_run_storage()
        if self.run_dir is None:
            raise RuntimeError("Run directory not created.")

        epoch = max((len(lst) for lst in history.values()), default=0)
        if epoch % self.checkpoints_freq == 0:
            plot_file = Path(f"learning_curves_epoch_{epoch:03d}").with_suffix(self.checkpoints_ext)
            current_file = self.run_dir / plot_file

            save_learning_curves(history, current_file, self.window, self.order)
            logger.debug(f"Saving learning curves to {current_file!s}...")

            if self.latest_file is not None:
                remove_file(self.latest_file)
            self.latest_file = current_file
