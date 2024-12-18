"""
This module provides utility functions for training and validating a vehicle recognition model using PyTorch.

Functions:
    - train_step(): Performs a single training step for the model.
    - valid_step(): Performs a single validation step for the model.
    - fit(): Executes the training loop for the model over a specified number of epochs.
"""

import os
import time
from collections import defaultdict

import mlflow
import torch
import torch.nn as nn
from dotenv import find_dotenv, load_dotenv
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric

from .callbacks import Callback, EarlyStopping, ModelCheckpoint, RecordedStats
from .common import init_logger
from .loaders import VehicleDataLoader

load_dotenv(find_dotenv())

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

logger = init_logger("DEBUG" if DEBUG else "INFO")


def train_step(
    model: Module,
    loader: VehicleDataLoader,
    loss: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str | torch.device,
    l1_weight: float = 0.0,
    l2_weight: float = 0.0,
) -> None:
    """Training step for the model."""
    model.train()
    loader.train()

    total_model_loss: Tensor = torch.tensor(0.0)
    total_grads_norm: Tensor = torch.tensor(0.0)
    total_l1_loss: Tensor = torch.tensor(0.0)
    total_l2_loss: Tensor = torch.tensor(0.0)

    for x, y in loader:
        grads: list[Tensor] = list()
        l1_loss: Tensor = torch.tensor(0.0, device=device)
        l2_loss: Tensor = torch.tensor(0.0, device=device)

        for param in model.parameters():
            l1_loss += param.abs().sum()
            l2_loss += param.square().sum()

            if param.grad is not None:
                grads.append(param.grad.detach().cpu().flatten())

        model_loss = loss.forward(model.forward(x.to(device)).squeeze(), y.to(device))
        l1_loss = l1_weight * l1_loss
        l2_loss = l2_weight * l2_loss
        total_loss = model_loss + l1_loss + l2_loss

        total_model_loss += model_loss.cpu()
        total_l1_loss += l1_loss.cpu()
        total_l2_loss += l2_loss.cpu()
        if grads:
            total_grads_norm += torch.cat(grads).norm(2)

        optimizer.zero_grad()
        total_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    mean_model_loss: float = total_model_loss.item() / len(loader)
    mean_l1_loss: float = total_l1_loss.item() / len(loader)
    mean_l2_loss: float = total_l2_loss.item() / len(loader)
    mean_grads_norm: float = total_grads_norm.item() / len(loader)

    pattern = "Mean Model Loss: {:3.6f} | Mean L1 Loss: {:3.6f} | Mean L2 Loss: {:3.6f} | Grads Norm: {:3.6f}"
    logger.debug(pattern.format(mean_model_loss, mean_l1_loss, mean_l2_loss, mean_grads_norm))


def valid_step(
    model: Module,
    loader: VehicleDataLoader,
    loss: Module,
    metric: Metric,
    device: str | torch.device,
) -> tuple[float, float]:
    """Validation step for the model."""
    model.eval()
    loader.eval()
    metric.reset()
    model_loss: Tensor = torch.tensor(0.0)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_logit = model.forward(x).squeeze()
            y_proba = torch.softmax(y_logit, dim=-1)
            model_loss += loss.forward(y_logit, y).item()
            metric.update(y_proba, y)

    return model_loss.item() / len(loader), metric.compute().item()


def fit(
    model: Module,
    train_loader: VehicleDataLoader,
    valid_loader: VehicleDataLoader,
    loss: Module,
    metric: Metric,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str | torch.device,
    l1_weight: float = 0.0,
    l2_weight: float = 0.0,
    epochs: int = 100,
    callbacks: list[Callback] | None = None,
) -> dict[RecordedStats, list[float]]:
    """Training loop for the model."""

    start_epoch: int = 1
    history = defaultdict(list)
    callbacks = callbacks or list()
    log = (
        "Epoch: {:3d} | Train Time: {:3.2f} s | Train Loss: {:6.4f} | "
        "Train Acc: {:6.4f} | Val Loss: {:6.4f} | Val Acc: {:6.4f}"
    )

    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            start_epoch = 1 + callback.latest_epoch
            history = defaultdict(list, callback.history)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_step(model, train_loader, loss, optimizer, scheduler, device, l1_weight, l2_weight)
        t1 = time.perf_counter()

        train_loss, train_acc = valid_step(model, train_loader, loss, metric, device)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss, metric, device)

        history[RecordedStats.TRAIN_LOSS].append(train_loss)
        history[RecordedStats.TRAIN_ACCURACY].append(train_acc)
        history[RecordedStats.VAL_LOSS].append(valid_loss)
        history[RecordedStats.VAL_ACCURACY].append(valid_acc)

        mlflow.log_metric(RecordedStats.TRAIN_LOSS, train_loss, step=epoch)
        mlflow.log_metric(RecordedStats.TRAIN_ACCURACY, train_acc, step=epoch)
        mlflow.log_metric(RecordedStats.VAL_LOSS, valid_loss, step=epoch)
        mlflow.log_metric(RecordedStats.VAL_ACCURACY, valid_acc, step=epoch)

        info = log.format(epoch, t1 - t0, train_loss, train_acc, valid_loss, valid_acc)
        logger.info(info)

        for callback in callbacks:
            if callback(history) and isinstance(callback, EarlyStopping):
                logger.info("Early stopping...")
                return history

    return history
