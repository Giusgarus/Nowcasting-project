"""Diagnostic plots for current-level persistence models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_true_vs_predicted_duration(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a true-vs-predicted duration scatter plot."""

    import matplotlib.pyplot as plt

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(predictions["y_true_seconds"], predictions["y_pred_seconds"], s=8, alpha=0.6)
    max_value = float(
        max(predictions["y_true_seconds"].max(), predictions["y_pred_seconds"].max())
    )
    ax.plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("True remaining persistence (s)")
    ax.set_ylabel("Predicted remaining persistence (s)")
    ax.set_title("Current-Level Persistence: True vs Predicted")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_residuals(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """Save a residual histogram in seconds."""

    import matplotlib.pyplot as plt

    output = _prepare_output(output_path)
    residuals = predictions["y_pred_seconds"] - predictions["y_true_seconds"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=60)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction residual (s)")
    ax.set_ylabel("Count")
    ax.set_title("Current-Level Persistence Residuals")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_error_by_event(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """Save mean absolute error by event."""

    import matplotlib.pyplot as plt

    output = _prepare_output(output_path)
    by_event = (
        predictions.groupby("global_event_id", sort=False)["absolute_error_seconds"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(by_event)), by_event.to_numpy())
    ax.set_xlabel("Event rank")
    ax.set_ylabel("Mean absolute error (s)")
    ax.set_title("Current-Level Persistence Error by Event")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_duration_distribution(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save predicted and true duration distributions."""

    import matplotlib.pyplot as plt

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(predictions["y_true_seconds"], bins=60, alpha=0.5, label="True")
    ax.hist(predictions["y_pred_seconds"], bins=60, alpha=0.5, label="Predicted")
    ax.set_xlabel("Remaining persistence (s)")
    ax.set_ylabel("Count")
    ax.set_title("Current-Level Persistence Duration Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_learned_shapelets(
    shapelets_by_length: dict[int, np.ndarray],
    output_dir: str | Path,
) -> None:
    """Save learned shapelets grouped by shapelet length."""

    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for length, shapelets in sorted(shapelets_by_length.items()):
        values = np.asarray(shapelets)
        fig, ax = plt.subplots(figsize=(9, 5))
        for shapelet in values:
            ax.plot(shapelet, alpha=0.35)
        ax.set_title(f"Learned Shapelets, Length {length}")
        ax.set_xlabel("Shapelet sample")
        ax.set_ylabel("Relative signal")
        fig.tight_layout()
        fig.savefig(destination / f"learned_shapelets_L{length}.png", dpi=150)
        plt.close(fig)
