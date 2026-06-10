"""Interactive plotting helpers for signal-only exploratory analysis."""

import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "nowcasting-analysis-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def _set_figure_title(fig: plt.Figure, title: str) -> None:
    """Set the interactive window title when supported by the active backend."""

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def _show(fig: plt.Figure) -> None:
    """Render a figure in the active interactive environment."""

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_signal_over_time(
    dataframe: pd.DataFrame,
    title: str,
) -> None:
    """Display a full Signal-over-Time line plot."""

    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    ax.plot(dataframe["Time"], dataframe["Signal"], linewidth=0.5)
    ax.set(title=title, xlabel="Time", ylabel="Signal")
    ax.grid(alpha=0.2)
    _show(fig)


def plot_signal_zoom(
    dataframe: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    title: str,
) -> None:
    """Display a Signal-over-Time plot for a selected row interval."""

    zoom = dataframe.iloc[max(0, start_idx) : min(len(dataframe), end_idx)]
    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    ax.plot(zoom["Time"], zoom["Signal"], marker=".", linewidth=0.8, markersize=2)
    ax.set(title=title, xlabel="Time", ylabel="Signal")
    ax.grid(alpha=0.2)
    _show(fig)


def plot_sampling_histogram(
    dt_seconds: pd.Series,
    title: str,
    *,
    max_seconds: float = 600,
) -> None:
    """Display a histogram of positive sampling intervals up to max_seconds."""

    visible = dt_seconds[(dt_seconds > 0) & (dt_seconds <= max_seconds)]
    fig, ax = plt.subplots(figsize=(8, 4))
    display_title = f"{title} (0 < dt <= {max_seconds:g}s)"
    _set_figure_title(fig, display_title)
    ax.hist(visible, bins=60)
    ax.set(
        title=display_title,
        xlabel="Sampling interval (seconds)",
        ylabel="Count",
    )
    ax.grid(alpha=0.2)
    _show(fig)


def plot_signal_histogram(
    dataframe: pd.DataFrame,
    title: str,
) -> None:
    """Display a Signal histogram."""

    fig, ax = plt.subplots(figsize=(8, 4))
    _set_figure_title(fig, title)
    ax.hist(dataframe["Signal"], bins=80)
    ax.set(title=title, xlabel="Signal", ylabel="Count")
    ax.grid(alpha=0.2)
    _show(fig)


def plot_signal_with_quantile_lines(
    dataframe: pd.DataFrame,
    title: str,
) -> None:
    """Display full Signal with exploratory q05 and q95 reference lines."""

    q05 = dataframe["Signal"].quantile(0.05)
    q95 = dataframe["Signal"].quantile(0.95)
    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    ax.plot(dataframe["Time"], dataframe["Signal"], linewidth=0.5)
    ax.axhline(q05, color="tab:blue", linestyle="--", label=f"q05={q05:.3g}")
    ax.axhline(q95, color="tab:red", linestyle="--", label=f"q95={q95:.3g}")
    ax.set(title=title, xlabel="Time", ylabel="Signal")
    ax.legend()
    ax.grid(alpha=0.2)
    _show(fig)


def plot_combined_boxplot(
    signals: dict[str, pd.Series],
    title: str,
) -> None:
    """Display a combined horizontal boxplot for several datasets."""

    fig, ax = plt.subplots(figsize=(10, max(4, len(signals) * 0.6)))
    _set_figure_title(fig, title)
    ax.boxplot(list(signals.values()), tick_labels=list(signals), vert=False)
    ax.set(title=title, xlabel="Signal")
    ax.grid(alpha=0.2)
    _show(fig)
