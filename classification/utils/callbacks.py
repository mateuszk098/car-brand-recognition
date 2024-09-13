import os
from enum import StrEnum
from pathlib import Path
from time import strftime

import torch
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler


class Callbacks(StrEnum):
    """Available callbacks for the training loop."""

    EarlyStopping = "EarlyStopping"
    ModelCheckpoint = "ModelCheckpoint"


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


class ModelCheckpoint:
    def __init__(
        self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, freq: int = 1
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.freq = int(freq)
        self.checkpoints_dir = Path("./checkpoints/")
        self.run_dir: Path | None = None
        self.history: dict[str, list[float]] = dict()

    def __call__(self, history: dict[str, list[float]]) -> None:
        self.save(history)

    def save(self, history: dict[str, list[float]]) -> None:
        if self.run_dir is None:
            self.run_dir = self.checkpoints_dir / strftime("run-%Y-%m-%d-%H-%M-%S")
            self.run_dir.mkdir(parents=True, exist_ok=True)

        epoch = len(history["loss"])
        if epoch % self.freq == 0:
            checkpoint = Path(f"checkpoint-epoch-{epoch:03d}").with_suffix(".pt")
            torch.save(
                {
                    "history": dict(history),
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                self.run_dir / checkpoint,
            )

    def load(self, map_location: str | None = None) -> None:
        latest_checkpoint_path = max(
            self.checkpoints_dir.rglob("*.pt"), key=os.path.getctime, default=None
        )
        if latest_checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoints_dir!r}")

        checkpoint = torch.load(
            latest_checkpoint_path, weights_only=True, map_location=map_location
        )
        self.history = checkpoint["history"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
