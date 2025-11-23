"""Plotting helpers shared across notebooks."""
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(hist: dict) -> None:
    """Plot training loss and validation RMSE history."""
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train loss")
    plt.plot(hist["epoch"], hist["val_rmse"], label="val RMSE")
    plt.xlabel("epoch")
    plt.ylabel("loss / RMSE")
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parity_plot(y_true: Sequence[float], y_pred: Sequence[float], title: str = "Parity plot") -> None:
    """Scatter of predicted vs. ground-truth values."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, color="gray", linewidth=1.2)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def residual_hist(residuals: Sequence[float], bins: int = 40, title: str = "Residuals") -> None:
    """Histogram of residuals."""
    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.xlabel("Prediction error (y - ŷ)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def error_vs_descriptor(
    descriptor: Sequence[float],
    errors: Sequence[float],
    xlabel: str = "Descriptor",
    title: str = "Absolute error vs descriptor",
) -> None:
    """Quick scatter to inspect correlation between descriptor value and absolute error."""
    descriptor = np.array(descriptor)
    errors = np.abs(np.array(errors))
    plt.figure()
    plt.scatter(descriptor, errors, alpha=0.5, s=12)
    plt.xlabel(xlabel)
    plt.ylabel("|y - ŷ|")
    plt.title(title)
    plt.tight_layout()
    plt.show()
