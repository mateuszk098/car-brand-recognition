import os
import time
from collections import defaultdict
from typing import Any, Callable

import torch
from dotenv import find_dotenv, load_dotenv
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torcheval.metrics import Metric

from classification.utils.common import init_logger

from .callbacks import Callbacks
from .loaders import VehicleDataLoader

assert load_dotenv(find_dotenv()), "The .env file is missing!"

logger = init_logger(os.getenv("LOGGER"))


def train_step(
    model: Module,
    loader: VehicleDataLoader,
    loss: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str | torch.device,
) -> None:
    model.train()
    loader.train()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss_value = loss.forward(model.forward(x).squeeze(), y)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        scheduler.step()


def valid_step(
    model: Module,
    loader: VehicleDataLoader,
    loss: Module,
    metric: Metric,
    device: str | torch.device,
) -> tuple[float, float]:
    model.eval()
    loader.eval()
    metric.reset()
    loss_value = torch.tensor(0.0)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_logit = model.predict(x).squeeze()
            y_proba = torch.softmax(y_logit, dim=-1)
            loss_value += loss.forward(y_logit, y).item()
            metric.update(y_proba, y)

    return loss_value.item() / len(loader), metric.compute().item()


def fit(
    model: Module,
    train_loader: VehicleDataLoader,
    valid_loader: VehicleDataLoader,
    loss: Module,
    metric: Metric,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: str | torch.device,
    epochs: int = 100,
    verbose_step: int = 1,
    callbacks: dict[Callbacks, Callable[..., Any]] | None = None,
) -> dict[str, list[float]]:

    start_epoch = 1
    history = defaultdict(list)
    log = (
        "Epoch: {:3d} | Train Time: {:3.2f}s | Train Loss: {:6.4f} | "
        "Train Acc: {:6.4f} | Val Loss: {:6.4f} | Val Acc: {:6.4f}"
    )

    callbacks = callbacks if callbacks is not None else dict()
    checkpoint_callback = callbacks.get(Callbacks.ModelCheckpoint)
    early_stopping_callback = callbacks.get(Callbacks.EarlyStopping)
    plot_checkpoint_callback = callbacks.get(Callbacks.PlotCheckpoint)

    if checkpoint_callback is not None:
        checkpoint_loss = checkpoint_callback.history.get("loss")
        if checkpoint_loss is not None:
            start_epoch = len(checkpoint_loss) + 1
            history = defaultdict(list, checkpoint_callback.history)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_step(model, train_loader, loss, optimizer, scheduler, device)
        t1 = time.perf_counter()

        train_loss, train_acc = valid_step(model, train_loader, loss, metric, device)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss, metric, device)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(valid_loss)
        history["val_accuracy"].append(valid_acc)

        if epoch % verbose_step == 0 or epoch == 1:
            info = log.format(epoch, t1 - t0, train_loss, train_acc, valid_loss, valid_acc)
            logger.info(info)

        if checkpoint_callback is not None:
            checkpoint_callback(history)

        if plot_checkpoint_callback is not None:
            plot_checkpoint_callback(history)

        if early_stopping_callback is not None and early_stopping_callback(valid_loss):
            logger.info("Early Stopping...")
            break

    return history
