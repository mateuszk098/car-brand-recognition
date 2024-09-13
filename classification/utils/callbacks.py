import os
from abc import ABCMeta, abstractmethod
from enum import StrEnum
from pathlib import Path
from time import strftime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler

from .common import init_logger

assert load_dotenv(find_dotenv()), "The .env file is missing!"

logger = init_logger(os.getenv("LOGGER"))

plt.style.use("ggplot")
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.color": "#4A4B52",
        "lines.linestyle": "--",
        "lines.linewidth": 1.25,
    }
)

TEXT = r"\textrm{{{}}}"
TEXT_COLOR = "#4A4B52"
TRAIN_COLOR = "#4A4B52"
VAL_COLOR = "#F78A1F"


def smooth(x: NDArray, y: NDArray, window: int = 7, order: int = 2) -> tuple[NDArray, NDArray]:
    if len(y) < window:
        return x, y
    y_filtered = savgol_filter(y, window, order)
    interpolator = CubicSpline(x, y_filtered)
    x_new = np.linspace(x[0], x[-1], len(x) * 10)
    y_new = interpolator(x_new)
    return x_new, y_new


def remove_file(file) -> None:
    try:
        os.remove(file)
    except FileNotFoundError:
        pass
    except TypeError:
        pass


class Callbacks(StrEnum):
    """Available callbacks for the training loop."""

    EarlyStopping = "EarlyStopping"
    ModelCheckpoint = "ModelCheckpoint"
    PlotCheckpoint = "PlotCheckpoint"


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss: float) -> bool:
        return self.should_stop(validation_loss)

    def should_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False

        if validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


class Checkpoint(metaclass=ABCMeta):
    def __init__(
        self,
        checkpoints_freq: int,
        checkpoints_ext: str,
        checkpoints_type: str,
        checkpoints_dir: str | Path,
    ) -> None:
        self.checkpoints_freq = int(checkpoints_freq)
        self.checkpoints_ext = str(checkpoints_ext)
        self.checkpoints_type = str(checkpoints_type)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.run_dir: Path | None = None
        self.latest_file: Path | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.save(*args, **kwargs)

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    def create_run_storage(self) -> None:
        if self.run_dir is None:
            unique_run = strftime(f"%Y_%m_%d_%H_%M_{self.checkpoints_type}")
            self.run_dir = self.checkpoints_dir / unique_run
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def infer_run_storage(self) -> None:
        checkpoints = self.checkpoints_dir.rglob("*" + self.checkpoints_ext)
        latest_file = max(checkpoints, key=os.path.getctime, default=None)
        if latest_file is None:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoints_dir!s}")

        self.latest_file = latest_file
        self.run_dir = latest_file.parent


class ModelCheckpoint(Checkpoint):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        checkpoints_freq: int = 1,
        checkpoints_dir: str | Path = "./checkpoints/",
    ) -> None:
        checkpoints_type = "states"
        checkpoints_ext = ".tar"
        super().__init__(checkpoints_freq, checkpoints_ext, checkpoints_type, checkpoints_dir)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history: dict[str, list[float]] = dict()

    def save(self, history: dict[str, list[float]]) -> None:
        super().create_run_storage()
        if self.run_dir is None:
            raise RuntimeError("Run directory not created.")

        epoch = len(history["loss"])
        if epoch % self.checkpoints_freq == 0:
            checkpoint = Path(f"state_epoch_{epoch:03d}").with_suffix(self.checkpoints_ext)
            current_file = self.run_dir / checkpoint
            logger.debug(f"Saving state to {current_file!s}...")
            torch.save(
                {
                    "history": dict(history),
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                current_file,
            )
            remove_file(self.latest_file)
            self.latest_file = current_file

    def load(self, map_location: str | torch.device | None = None) -> None:
        super().infer_run_storage()
        if self.latest_file is None:
            raise RuntimeError("Cannot infer run storage.")

        logger.debug(f"Loading state from {self.latest_file!s}...")
        checkpoint = torch.load(self.latest_file, map_location, weights_only=True)
        self.history = checkpoint["history"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])


class PlotCheckpoint(Checkpoint):
    def __init__(
        self,
        smooth_out: bool = True,
        window: int = 5,
        order: int = 2,
        checkpoints_freq: int = 1,
        checkpoints_dir: str | Path = "./checkpoints/",
    ) -> None:
        checkpoints_type = "plots"
        checkpoints_ext = ".png"
        super().__init__(checkpoints_freq, checkpoints_ext, checkpoints_type, checkpoints_dir)

        self.smooth_out = bool(smooth_out)
        self.window = int(window)
        self.order = int(order)

    def save(self, history: dict[str, list[float]]) -> None:
        super().create_run_storage()
        if self.run_dir is None:
            raise RuntimeError("Run directory not created.")

        epoch = len(history["loss"])
        if epoch % self.checkpoints_freq == 0:
            plot_file = Path(f"learning_curves_epoch_{epoch:03d}").with_suffix(self.checkpoints_ext)
            current_file = self.run_dir / plot_file

            data = pd.DataFrame(history).rename_axis("epoch").reset_index().assign(epoch=lambda x: x.epoch.add(1))
            fig = self._draw_learning_curves(data)
            logger.debug(f"Saving learning curves to {current_file!s}...")
            fig.savefig(current_file)

            remove_file(self.latest_file)
            self.latest_file = current_file

    def load(self) -> None:
        super().infer_run_storage()

    def _draw_learning_curves(self, data: DataFrame) -> Figure:
        x_loss, y_loss = smooth(data.epoch.to_numpy(), data.loss.to_numpy())
        x_val_loss, y_val_loss = smooth(data.epoch.to_numpy(), data.val_loss.to_numpy())
        x_acc, y_acc = smooth(data.epoch.to_numpy(), data.accuracy.to_numpy())
        x_val_acc, y_val_acc = smooth(data.epoch.to_numpy(), data.val_accuracy.to_numpy())

        fig = plt.figure(figsize=(9.0, 3.5), tight_layout=True)

        plt.subplot(1, 2, 1)
        plt.suptitle(TEXT.format("Training History"))
        plt.plot(x_loss, y_loss, label=TEXT.format("Train"), color=TRAIN_COLOR)
        plt.plot(x_val_loss, y_val_loss, label=TEXT.format("Valid"), color=VAL_COLOR)
        plt.xlabel(TEXT.format("Epoch"))
        plt.ylabel(TEXT.format("Loss"))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_acc, y_acc, label=TEXT.format("Train"), color=TRAIN_COLOR)
        plt.plot(x_val_acc, y_val_acc, label=TEXT.format("Valid"), color=VAL_COLOR)
        plt.xlabel(TEXT.format("Epoch"))
        plt.ylabel(TEXT.format("Accuracy"))
        plt.legend()

        return fig
