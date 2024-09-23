"""Utility functions for visualizing training history."""

from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from .callbacks import History, RecordedStats

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


def smooth_curve(x: NDArray, y: NDArray, window: int = 5, order: int = 2) -> tuple[NDArray, NDArray]:
    """Smooths the curve using Savitzky-Golay filter and interpolates it using cubic spline."""
    if len(y) < window:
        return x, y
    y_filtered = savgol_filter(y, window, order)
    interpolator = CubicSpline(x, y_filtered)
    x_new = np.linspace(x[0], x[-1], len(x) * 10)
    y_new = interpolator(x_new)
    return x_new, y_new


def save_learning_curves(history: History, path: str | PathLike, window: int = 5, order: int = 2) -> None:
    """Saves the learning curves plot to the given path."""
    if not RecordedStats.content().issubset(history.keys()):
        raise ValueError(f"Missing required keys in history: {RecordedStats.content()!r}")

    data = pd.DataFrame(history).rename_axis("epoch").reset_index().assign(epoch=lambda x: x.epoch.add(1))
    epoch = data.epoch.to_numpy()

    train_loss = data[RecordedStats.TRAIN_LOSS].to_numpy()
    train_accuracy = data[RecordedStats.TRAIN_ACCURACY].to_numpy()
    val_loss = data[RecordedStats.VAL_LOSS].to_numpy()
    val_accuracy = data[RecordedStats.VAL_ACCURACY].to_numpy()

    epoch_smooth, train_loss_smooth = smooth_curve(epoch, train_loss, window, order)
    _, train_accuracy_smooth = smooth_curve(epoch, train_accuracy, window, order)
    _, val_loss_smooth = smooth_curve(epoch, val_loss, window, order)
    _, val_accuracy_smooth = smooth_curve(epoch, val_accuracy, window, order)

    fig = plt.figure(figsize=(9.0, 3.5), tight_layout=True)

    plt.subplot(1, 2, 1)
    plt.suptitle(TEXT.format("Training History"))
    plt.plot(epoch_smooth, train_loss_smooth, label=TEXT.format("Train"), color=TRAIN_COLOR)
    plt.plot(epoch_smooth, val_loss_smooth, label=TEXT.format("Valid"), color=VAL_COLOR)
    plt.xlabel(TEXT.format("Epoch"))
    plt.ylabel(TEXT.format("Loss"))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_smooth, train_accuracy_smooth, label=TEXT.format("Train"), color=TRAIN_COLOR)
    plt.plot(epoch_smooth, val_accuracy_smooth, label=TEXT.format("Valid"), color=VAL_COLOR)
    plt.xlabel(TEXT.format("Epoch"))
    plt.ylabel(TEXT.format("Accuracy"))
    plt.legend()

    plt.savefig(path)
    plt.close(fig)
