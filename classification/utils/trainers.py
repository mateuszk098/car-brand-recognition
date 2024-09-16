import os
import time
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from dotenv import find_dotenv, load_dotenv
from torch import Tensor
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
    l1_weight: float = 0.0,
    l2_weight: float = 0.0,
) -> None:
    model.train()
    loader.train()

    total_model_loss = torch.tensor(0.0)
    total_grads_norm = torch.tensor(0.0)
    total_l1_loss = torch.tensor(0.0)
    total_l2_loss = torch.tensor(0.0)

    for x, y in loader:
        grads: list[Tensor] = list()
        l1_loss = torch.tensor(0.0, device=device)
        l2_loss = torch.tensor(0.0, device=device)

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

    mean_model_loss = total_model_loss.item() / len(loader)
    mean_l1_loss = total_l1_loss.item() / len(loader)
    mean_l2_loss = total_l2_loss.item() / len(loader)
    mean_grads_norm = total_grads_norm.item() / len(loader)

    pattern = "Mean Model Loss: {:3.6f} | Mean L1 Loss: {:3.6f} | Mean L2 Loss: {:3.6f} | Grads Norm: {:3.6f}"
    logger.debug(pattern.format(mean_model_loss, mean_l1_loss, mean_l2_loss, mean_grads_norm))


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
    model_loss = torch.tensor(0.0)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_logit = model.predict(x).squeeze()
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
    callbacks: dict[Callbacks, Callable[..., Any]] | None = None,
) -> dict[str, list[float]]:

    start_epoch = 1
    history = defaultdict(list)
    log = (
        "Epoch: {:3d} | Train Time: {:3.2f} s | Train Loss: {:6.4f} | "
        "Train Acc: {:6.4f} | Val Loss: {:6.4f} | Val Acc: {:6.4f}"
    )

    callbacks = callbacks if callbacks is not None else dict()
    checkpoint_callback = callbacks.get(Callbacks.ModelCheckpoint)
    early_stopping_callback = callbacks.get(Callbacks.EarlyStopping)
    learning_curves_checkpoint_callback = callbacks.get(Callbacks.LearningCurvesCheckpoint)

    if checkpoint_callback is not None:
        start_epoch = 1 + checkpoint_callback.latest_epoch
        history = defaultdict(list, checkpoint_callback.history)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_step(model, train_loader, loss, optimizer, scheduler, device, l1_weight, l2_weight)
        t1 = time.perf_counter()

        train_loss, train_acc = valid_step(model, train_loader, loss, metric, device)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss, metric, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(valid_loss)
        history["val_accuracy"].append(valid_acc)

        info = log.format(epoch, t1 - t0, train_loss, train_acc, valid_loss, valid_acc)
        logger.info(info)

        if checkpoint_callback is not None:
            checkpoint_callback(history)

        if learning_curves_checkpoint_callback is not None:
            learning_curves_checkpoint_callback(history)

        if early_stopping_callback is not None and early_stopping_callback(valid_loss):
            logger.info("Early Stopping...")
            break

    return history
