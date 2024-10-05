import os
from argparse import ArgumentParser
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics.classification as metrics
from dotenv import find_dotenv, load_dotenv
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module
from torchvision.datasets import ImageFolder

from resnet.inference import load_se_resnet
from resnet.utils.common import init_logger, load_yaml
from resnet.utils.loaders import VehicleDataLoader
from resnet.utils.transforms import eval_transform

load_dotenv(find_dotenv())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = init_logger("INFO" if os.getenv("MODE") == "PROD" else "DEBUG")


@dataclass(frozen=True, kw_only=True)
class EvaluationResult:
    Loss: float
    Accuracy: float
    Precision: float
    Recall: float
    F1Score: float
    ConfusionMatrix: NDArray


def evaluate(
    model: Module,
    loader: VehicleDataLoader,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    normalize: Literal["none", "true", "pred", "all"] | None = None,
) -> EvaluationResult:
    model.eval()
    loader.eval()

    model_loss: Tensor = torch.tensor(0.0)
    num_classes = len(loader.dataset.classes)  # type: ignore

    loss = nn.CrossEntropyLoss()
    accuracy = metrics.MulticlassAccuracy(average=average, num_classes=num_classes).to(DEVICE)
    precision = metrics.MulticlassPrecision(average=average, num_classes=num_classes).to(DEVICE)
    recall = metrics.MulticlassRecall(average=average, num_classes=num_classes).to(DEVICE)
    f1_score = metrics.MulticlassF1Score(average=average, num_classes=num_classes).to(DEVICE)
    confusion_matrix = metrics.MulticlassConfusionMatrix(num_classes=num_classes, normalize=normalize).to(DEVICE)

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_logit = model.forward(x).squeeze()
            y_proba = torch.softmax(y_logit, dim=-1)
            model_loss += loss.forward(y_logit, y).item()
            accuracy.update(y_proba, y)
            precision.update(y_proba, y)
            recall.update(y_proba, y)
            f1_score.update(y_proba, y)
            confusion_matrix.update(y_proba, y)

    plt.figure(figsize=(16, 16), dpi=300, tight_layout=True)
    confusion_matrix.plot(ax=plt.gca(), cmap="bone_r", labels=loader.dataset.classes)  # type: ignore
    for text in plt.gca().texts:
        text.set_fontsize(10)
    plt.gca().set_xlabel("Predicted")
    plt.gca().set_ylabel("Ground Truth")
    plt.savefig("confusion_matrix.png")

    return EvaluationResult(
        Loss=model_loss.item() / len(loader),
        Accuracy=accuracy.compute().item(),
        Precision=precision.compute().item(),
        Recall=recall.compute().item(),
        F1Score=f1_score.compute().item(),
        ConfusionMatrix=confusion_matrix.compute().cpu().numpy().round(2),
    )


def main(*, config_file: str | PathLike) -> None:
    logger.info(f"Loading configuration from {config_file!s}...")
    config = SimpleNamespace(**load_yaml(config_file))

    valid_dataset = ImageFolder(config.VALID_DATASET)

    model = load_se_resnet(config.ARCH_TYPE, config.WEIGHTS_FILE)
    model = model.to(DEVICE)

    valid_loader = VehicleDataLoader(
        valid_dataset,
        eval_transform=eval_transform(model.architecture.INPUT_SHAPE),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    res = evaluate(model, valid_loader, config.METRIC_AVERAGE, config.CONFUSION_MATRIX_NORMALIZE)

    log = "Accuracy: {:4.2%} | Precision: {:4.2%} | " "Recall: {:4.2%} | F1 Score: {:4.2%} | Loss {:6.4f}"
    np.set_printoptions(precision=2, linewidth=140, floatmode="fixed")

    logger.info(log.format(res.Accuracy, res.Precision, res.Recall, res.F1Score, res.Loss))
    logger.info(f"Confusion Matrix:\n{res.ConfusionMatrix}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config_file", type=Path, required=True)
    kwargs = vars(p.parse_args())
    main(**kwargs)
