"""
This module provides various callback classes for handling model checkpoints, 
learning curves, and early stopping during training.

Classes:
    - Callback(Protocol): Protocol for callback functions.
    - Checkpoint(metaclass=ABCMeta): Abstract base class for saving and loading checkpoints.
    - ModelCheckpoint(Checkpoint): Model checkpoint callback for saving and loading model states.
    - LearningCurvesCheckpoint(Checkpoint): Learning curves checkpoint callback 
        for saving learning curves plots.
    - EarlyStopping: Early stopping callback with the given monitoring value.
    
Functions:
    - infer_latest_dir(directory) -> PathLike | None: Returns the latest created 
        subdirectory in the given directory.
    - infer_latest_file(directory, ext: str | None = None) -> PathLike | None: Returns the latest 
        created file with given extension in the given directory.
    - is_valid_ext(ext: str) -> bool: Checks if the given file extension is valid.
"""

import os
from abc import ABCMeta, abstractmethod
from os import PathLike
from pathlib import Path
from time import strftime
from typing import Any, ClassVar, Protocol

import torch
from dotenv import find_dotenv, load_dotenv
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler

from .common import History, RecordedStats, init_logger, remove_file
from .visualize import save_learning_curves

load_dotenv(find_dotenv())
torch.serialization.add_safe_globals([RecordedStats])

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

logger = init_logger("DEBUG" if DEBUG else "INFO")


class Callback(Protocol):
    """Each callback should implement the `__call__` method."""

    def __call__(self, history: History) -> Any: ...


class Checkpoint(metaclass=ABCMeta):
    """Abstract base class for saving and loading checkpoints."""

    _run_dir: ClassVar[PathLike | None] = None

    def __init__(self, checkpoints_dir: str | PathLike, checkpoints_freq: int, checkpoints_ext: str) -> None:
        """
        Initializes the callback with the specified parameters.
        Args:
            checkpoints_dir (str | PathLike): The directory where checkpoints will be saved.
            checkpoints_freq (int): The frequency (in epochs) at which checkpoints will be saved.
            checkpoints_ext (str): The file extension for the checkpoint files.
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_freq = int(checkpoints_freq)
        self.checkpoints_ext = str(checkpoints_ext)
        self.latest_file: PathLike | None = None
        self.run_dir: PathLike | None = None

    def __call__(self, history: History) -> None:
        """
        Invokes the callback with the given training history.
        Args:
            history (History): The training history object containing details of the training process.
        """
        self.save(history)

    @abstractmethod
    def save(self, history: History) -> None:
        """
        Save the training history.
        Args:
            history (History): The training history object to be saved.
        """

    def load(self) -> PathLike:
        """
        Returns the latest run directory.
        This method infers the latest run directory from the checkpoints directory.
        If no runs are found, it raises a FileNotFoundError.
        Returns:
            The path to the latest run directory.
        Raises:
            FileNotFoundError: If no runs are found in the checkpoints directory.
        """
        latest_run_dir = infer_latest_dir(self.checkpoints_dir)
        if latest_run_dir is None:
            raise FileNotFoundError(f"No runs found in {self.checkpoints_dir!s}")
        Checkpoint._run_dir = latest_run_dir
        return Checkpoint._run_dir

    def create_run(self) -> PathLike:
        """
        Creates a new run directory and returns its path.
        This method generates a new directory within the `checkpoints_dir` using the current
        timestamp formatted as "run_HH_MM_SS_DD_MM_YYYY". If the directory does not already
        exist, it is created. The path to this new run directory is then returned.
        Returns:
            The path to the newly created run directory.
        """
        if Checkpoint._run_dir is None:
            Checkpoint._run_dir = self.checkpoints_dir / strftime(f"run_%H_%M_%S_%d_%m_%Y")
            Checkpoint._run_dir.mkdir(parents=True, exist_ok=True)
        return Checkpoint._run_dir


class ModelCheckpoint(Checkpoint):
    """Model checkpoint callback for saving and loading model states."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        checkpoints_dir: str | PathLike = "./checkpoints/",
        checkpoints_freq: int = 1,
    ) -> None:
        """
        Initialize the callback with the given parameters.
        Args:
            model (Module): The model to be trained.
            optimizer (Optimizer): The optimizer for training the model.
            scheduler (LRScheduler): The learning rate scheduler.
            checkpoints_dir (str | PathLike, optional): Directory to save checkpoints.
                Defaults to "./checkpoints/".
            checkpoints_freq (int, optional): Frequency (in epochs) to save checkpoints.
                Defaults to 1.
        """
        super().__init__(checkpoints_dir, checkpoints_freq, ".tar")
        Checkpoint._run_dir = None

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.latest_epoch: int = 0
        self.history: History = dict()

    def save(self, history: History) -> None:
        if self.run_dir is None:
            self.run_dir = super().create_run()

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

            if self.latest_file is not None and current_file != self.latest_file:
                remove_file(self.latest_file)
            self.latest_file = current_file

    def load(self, map_location: str | torch.device | None = None) -> None:
        """
        Load the latest checkpoint from the run directory.
        This method loads the latest checkpoint file from the run directory, updating the state of the model,
        optimizer, and scheduler with the saved state. It also updates the latest epoch and training history.
        Args:
            map_location (str | torch.device | None, optional): The device to map the loaded tensors to.
                Can be a string specifying the device, a torch.device object, or None. Defaults to None.
        Raises:
            RuntimeError: If the run directory is empty and no checkpoint file is found.
        """
        self.run_dir = super().load()
        self.latest_file = infer_latest_file(self.run_dir, self.checkpoints_ext)
        if self.latest_file is None:
            raise RuntimeError("Run directory is empty.")

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
        checkpoints_dir: str | PathLike = "./checkpoints/",
        checkpoints_freq: int = 1,
    ) -> None:
        """
        Initialize the callback with optional smoothing and checkpoint parameters.
        Args:
            smooth_out (bool): Whether to apply smoothing to the output. Defaults to True.
            window (int): The window size for smoothing. Defaults to 5.
            order (int): The order of the smoothing filter. Defaults to 2.
            checkpoints_dir (str | PathLike): Directory where checkpoints will be saved.
                Defaults to "./checkpoints/".
            checkpoints_freq (int): Frequency of saving checkpoints. Defaults to 1.
        """
        super().__init__(checkpoints_dir, checkpoints_freq, ".png")
        Checkpoint._run_dir = None

        self.smooth_out = bool(smooth_out)
        self.window = int(window)
        self.order = int(order)

    def save(self, history: History) -> None:
        if self.run_dir is None:
            self.run_dir = super().create_run()

        epoch = max((len(lst) for lst in history.values()), default=0)
        if epoch % self.checkpoints_freq == 0:
            plot = Path(f"learning_curves_epoch_{epoch:03d}").with_suffix(self.checkpoints_ext)
            current_file = self.run_dir / plot

            logger.debug(f"Saving learning curves to {current_file!s}...")
            save_learning_curves(history, current_file, self.window, self.order)

            if self.latest_file is not None and current_file != self.latest_file:
                remove_file(self.latest_file)
            self.latest_file = current_file

    def load(self) -> None:
        """
        Loads the run directory and infers the latest checkpoint file.
        This method sets the `run_dir` attribute by calling the `load` method of the superclass.
        It then infers the latest checkpoint file in the run directory and sets
        the `latest_file` attribute.
        """
        self.run_dir = super().load()
        self.latest_file = infer_latest_file(self.run_dir, self.checkpoints_ext)


class EarlyStopping:
    """Early stopping callback with the given monitoring value."""

    def __init__(
        self,
        monitor_value: str | RecordedStats = RecordedStats.VAL_LOSS,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:
        """
        Initializes the callback with parameters to monitor a specific value, patience, and minimum delta.
        Args:
            monitor_value (str | RecordedStats, optional): The value to monitor.
                Defaults to RecordedStats.VAL_LOSS.
            patience (int, optional): Number of epochs with no improvement after which
                training will be stopped. Defaults to 10.
            min_delta (float, optional): Minimum change in the monitored value to qualify
                as an improvement. Defaults to 0.0.
        """
        self.monitor_value = RecordedStats(monitor_value)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.min_value = float("inf")

    def __call__(self, history: History) -> bool:
        """
        Invokes the callback with the given training history.
        Args:
            history (History): The training history object.
        Returns:
            A boolean indicating whether the training should stop.
        """
        return self.should_stop(history)

    def should_stop(self, history: History) -> bool:
        """
        Determines whether training should stop based on the monitored value and patience.
        Args:
            history (History): An object containing the history of monitored values.
        Returns:
            True if training should stop, False otherwise.
        """
        current_values = history[self.monitor_value]
        current_value = 0.0 if not current_values else current_values[-1]

        if current_value < self.min_value - self.min_delta:
            self.min_value = current_value
            self.counter = 0
            return False

        self.counter += 1
        if self.counter == self.patience:
            return True

        return False


def infer_latest_dir(directory) -> PathLike | None:
    """Returns the latest created subdirectory in the given directory."""
    dirs = (d for d in directory.glob("*") if d.is_dir())
    return max(dirs, key=os.path.getctime, default=None)


def infer_latest_file(directory, ext: str | None = None) -> PathLike | None:
    """Returns the latest created file with given extension in the given directory."""
    if ext is not None and is_valid_ext(ext):
        files = (f for f in directory.glob("*") if f.is_file() and f.suffix == ext)
    else:
        files = (f for f in directory.glob("*") if f.is_file())
    return max(files, key=os.path.getctime, default=None)


def is_valid_ext(ext: str) -> bool:
    """Checks if the given file extension is valid."""
    ext = str(ext)
    return ext.startswith(".") and len(ext) > 1
