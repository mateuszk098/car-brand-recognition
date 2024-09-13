import math
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import yaml
from torcheval import metrics
from torchvision.datasets import ImageFolder

from classification.network.resnet import SeResNet
from classification.utils.callbacks import Callbacks, EarlyStopping, ModelCheckpoint
from classification.utils.common import init_logger
from classification.utils.loaders import VehicleDataLoader
from classification.utils.trainers import fit
from classification.utils.transforms import VehicleTransform

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)

logger = init_logger("TRAIN")


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main(*, config_path: str | Path) -> None:
    config = SimpleNamespace(**load_config(config_path))

    train_dataset = ImageFolder(config.TRAIN_DATASET)
    valid_dataset = ImageFolder(config.VALID_DATASET)

    assert set(train_dataset.classes) == set(valid_dataset.classes)

    num_classes = len(train_dataset.classes)
    transform = VehicleTransform(size=config.INPUT_SIZE)

    train_loader = VehicleDataLoader(
        train_dataset,
        train_transform=transform.train_transform,
        eval_transform=transform.eval_transform,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    valid_loader = VehicleDataLoader(
        valid_dataset,
        eval_transform=transform.eval_transform,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = SeResNet(num_classes=num_classes).to(config.DEVICE)

    loss = nn.CrossEntropyLoss()
    metric = metrics.MulticlassAccuracy(average="macro", num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True, fused=True)  # type: ignore

    scheduler_steps = config.EPOCHS * int(math.ceil(len(train_dataset) / config.BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        div_factor=config.DIV_FACTOR,
        total_steps=scheduler_steps,
        anneal_strategy=config.ANNEAL_STRATEGY,
        cycle_momentum=config.CYCLE_MOMENTUM,
        base_momentum=config.BASE_MOMENTUM,
        max_momentum=config.MAX_MOMENTUM,
    )

    early_stopping_callback = EarlyStopping(config.PATIENCE, config.MIN_DELTA)
    model_checkpoint_callback = ModelCheckpoint(model, optimizer, scheduler, freq=1)

    if config.RESUME:
        model_checkpoint_callback.load(map_location=config.DEVICE)
        logger.info("Resuming training from the latest checkpoint...")

    history = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss=loss,
        metric=metric,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.EPOCHS,
        device=config.DEVICE,
        callbacks={
            Callbacks.EarlyStopping: early_stopping_callback,
            Callbacks.ModelCheckpoint: model_checkpoint_callback,
        },
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config_path", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
