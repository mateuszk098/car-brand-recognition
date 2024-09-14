import logging
import logging.config
import os
from importlib.resources import files
from logging import Logger

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

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


def draw_learning_curves(data: DataFrame) -> Figure:
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


def remove_file(file) -> None:
    try:
        os.remove(file)
    except FileNotFoundError:
        pass
    except TypeError:
        pass


def init_logger(name: str | None = None) -> Logger:
    config_path = files("classification.config").joinpath("logging.yaml")
    with open(str(config_path), "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    logging.config.dictConfig(content)
    return logging.getLogger(name)
