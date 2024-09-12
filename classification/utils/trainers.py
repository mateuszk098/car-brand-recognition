import time
from collections import defaultdict

import torch
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torcheval.metrics import Metric

from classification.utils.common import init_logger

from .callbacks import EarlyStopping
from .loaders import VehicleDataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = init_logger("TRAIN")


def train_step(
    model: Module,
    loader: VehicleDataLoader,
    loss: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> None:
    model.train()
    loader.train()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss_value = loss(model.forward(x).squeeze(), y)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        scheduler.step()


def valid_step(
    model: Module, loader: VehicleDataLoader, loss: Module, metric: Metric
) -> tuple[float, float]:
    model.eval()
    loader.eval()
    metric.reset()
    loss_value = torch.tensor(0.0)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_logit = model.predict(x).squeeze()
            y_proba = torch.softmax(y_logit, dim=-1)
            loss_value += loss.forward(y_logit, y).item()
            metric.update(y_proba, y)

    return loss_value.item() / len(loader), metric.compute().item()


def train_loop(
    model: Module,
    train_loader: VehicleDataLoader,
    valid_loader: VehicleDataLoader,
    loss: Module,
    metric: Metric,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    early_stopping: EarlyStopping,
    epochs: int = 100,
    verbose_step: int = 1,
) -> dict[str, list[float]]:

    model = model.to(DEVICE)
    history = defaultdict(list)
    log = "Epoch: {:3d} | Time: {:3.2f}s | Loss: {:8.5f} | ACC: {:8.5f} | Val Loss: {:8.5f} | Val ACC: {:8.5f}"

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_step(model, train_loader, loss, optimizer, scheduler)
        t1 = time.perf_counter()

        train_loss, train_acc = valid_step(model, train_loader, loss, metric)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss, metric)

        if early_stopping(valid_loss):
            logger.info("Early Stopping...")
            break

        if (epoch + 1) % verbose_step == 0 or epoch == 0:
            info = log.format(epoch + 1, t1 - t0, train_loss, train_acc, valid_loss, valid_acc)
            logger.info(info)

        history["Loss"].append(train_loss)
        history["ACC"].append(train_acc)
        history["Val Loss"].append(valid_loss)
        history["Val ACC"].append(valid_acc)

    return history
