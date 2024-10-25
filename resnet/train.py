"""Train a car braand recognition model using a ResNet architecture."""

import math
import os
from argparse import ArgumentParser
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from time import strftime
from types import SimpleNamespace

import mlflow
import mlflow.models.signature
import numpy as np
import torch
import torch.nn as nn
import torchmetrics.classification as metrics
from dotenv import find_dotenv, load_dotenv
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from torchvision.datasets import ImageFolder

from resnet.network.arch import arch_summary, init_se_resnet
from resnet.utils.callbacks import EarlyStopping, LearningCurvesCheckpoint, ModelCheckpoint
from resnet.utils.common import RecordedStats, init_logger, load_yaml
from resnet.utils.loaders import VehicleDataLoader
from resnet.utils.trainers import fit
from resnet.utils.transforms import eval_transform, train_transform

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available, aborting...")

load_dotenv(find_dotenv())

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
SEED = 42
DEVICE = torch.device("cuda")

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("network-training")

logger = init_logger("DEBUG" if DEBUG else "INFO")


def main(*, config_file: str | PathLike) -> None:
    """
    Main function to train a vehicle recognition model using a specified configuration file.
    The function performs the following steps:
    1. Loads the configuration from the specified file.
    2. Initializes the model, datasets, and data loaders.
    3. Sets up the loss function, metric, optimizer, and learning rate scheduler.
    4. Initializes training callbacks such as early stopping and model checkpointing.
    5. Optionally resumes training from the latest checkpoint if specified in the configuration.
    6. Logs configuration and model details to MLflow.
    7. Trains the model and logs training history.
    8. Saves the final trained model and logs it to MLflow with input and output schema signatures.
    Args:
        config_file (str | PathLike): Path to the YAML configuration file.
    """
    logger.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))

    logger.info("Initializing model, datasets and loaders...")
    train_dataset = ImageFolder(config.TRAIN_DATASET)
    valid_dataset = ImageFolder(config.VALID_DATASET)

    assert set(train_dataset.classes) == set(valid_dataset.classes)

    num_classes = len(train_dataset.classes)
    class_to_id = train_dataset.class_to_idx

    model = init_se_resnet(config.ARCH_TYPE, num_classes)
    model = model.to(DEVICE)
    input_shape = model.architecture.INPUT_SHAPE

    train_loader = VehicleDataLoader(
        train_dataset,
        train_transform=train_transform(input_shape),
        eval_transform=eval_transform(input_shape),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    valid_loader = VehicleDataLoader(
        valid_dataset,
        eval_transform=eval_transform(input_shape),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.PERSISTENT_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    logger.info("Initializing loss, metric, optimizer, and scheduler...")
    loss = nn.CrossEntropyLoss()
    metric = metrics.MulticlassAccuracy(average=config.METRIC_AVERAGE, num_classes=num_classes)
    metric = metric.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True, fused=True)  # type: ignore
    scheduler_steps = config.EPOCHS * int(math.ceil(len(train_dataset) / config.BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
        three_phase=config.THREE_PHASE,
        total_steps=scheduler_steps,
        anneal_strategy=config.ANNEAL_STRATEGY,
        cycle_momentum=config.CYCLE_MOMENTUM,
        base_momentum=config.BASE_MOMENTUM,
        max_momentum=config.MAX_MOMENTUM,
    )

    logger.info("Initializing callbacks...")
    early_stopping = EarlyStopping(
        monitor_value=RecordedStats(config.MONITOR),
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
    )
    model_checkpoint = ModelCheckpoint(
        model,
        optimizer,
        scheduler,
        checkpoints_freq=config.CHECKPOINTS_FREQ,
        checkpoints_dir=config.CHECKPOINTS_DIR,
    )
    learning_curves = LearningCurvesCheckpoint(
        checkpoints_freq=config.CHECKPOINTS_FREQ,
        checkpoints_dir=config.CHECKPOINTS_DIR,
    )
    if config.RESUME:
        model_checkpoint.load(map_location=DEVICE)
        learning_curves.load()
        logger.info("Resuming training from the latest checkpoint...")

    logger.info(f"Start training on {torch.cuda.get_device_name(DEVICE)}...")

    with mlflow.start_run():
        mlflow.log_artifact(str(config_file))
        mlflow.log_params(vars(config))
        mlflow.log_dict(asdict(model.architecture), "architecture.yaml")
        mlflow.log_dict(class_to_id, "class_to_idx.yaml")

        with TemporaryDirectory() as tmp_dir:
            tmp_f = Path(tmp_dir, "model_summary.txt")
            tmp_f.write_text(arch_summary(model))
            mlflow.log_artifact((str(tmp_f)))

        history = fit(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            loss=loss,
            metric=metric,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            l1_weight=config.L1_WEIGHT,
            l2_weight=config.L2_WEIGHT,
            epochs=config.EPOCHS,
            callbacks=[early_stopping, model_checkpoint, learning_curves],
        )

        final_model_id = Path(strftime(f"{config.ARCH_TYPE}_%H_%M_%S_%d_%m_%Y")).with_suffix(".pt")
        logger.info(f"Training completed, saving the final model to {final_model_id!s}...")
        torch.save(model.state_dict(), final_model_id)

        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3, *input_shape))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, num_classes))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pytorch.log_model(model, "model", signature=signature)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config_file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
