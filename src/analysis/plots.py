"""Interactive plotting helpers for signal-only exploratory analysis."""

import os
import tempfile
from collections.abc import Mapping
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
    if plt.get_backend().lower() != "agg":
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


def plot_signal_with_rolling_means(
    dataframe: pd.DataFrame,
    windows: Mapping[str, int],
    title: str,
) -> None:
    """Display Signal with rolling-mean baseline overlays."""

    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    ax.plot(dataframe["Time"], dataframe["Signal"], linewidth=0.4, label="Signal")
    for window_name, window_samples in windows.items():
        rolling_mean = dataframe["Signal"].rolling(
            window_samples,
            min_periods=window_samples,
        ).mean()
        ax.plot(
            dataframe["Time"],
            rolling_mean,
            linewidth=1,
            label=f"{window_name} mean ({window_samples} samples)",
        )
    ax.set(title=title, xlabel="Time", ylabel="Signal")
    ax.legend()
    ax.grid(alpha=0.2)
    _show(fig)


def plot_rolling_std(
    dataframe: pd.DataFrame,
    windows: Mapping[str, int],
    title: str,
) -> None:
    """Display rolling Signal standard deviations."""

    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    for window_name, window_samples in windows.items():
        rolling_std = dataframe["Signal"].rolling(
            window_samples,
            min_periods=window_samples,
        ).std()
        ax.plot(
            dataframe["Time"],
            rolling_std,
            linewidth=0.8,
            label=f"{window_name} std ({window_samples} samples)",
        )
    ax.set(title=title, xlabel="Time", ylabel="Rolling Signal std")
    ax.legend()
    ax.grid(alpha=0.2)
    _show(fig)


def plot_hourly_pattern(hourly_summary: pd.DataFrame, title: str) -> None:
    """Display mean and median Signal by hour of day."""

    fig, ax = plt.subplots(figsize=(9, 4))
    _set_figure_title(fig, title)
    ax.plot(hourly_summary["hour"], hourly_summary["mean_signal"], marker="o", label="Mean")
    ax.plot(
        hourly_summary["hour"],
        hourly_summary["median_signal"],
        marker="o",
        label="Median",
    )
    ax.set(title=title, xlabel="Hour of day", ylabel="Signal", xticks=range(24))
    ax.legend()
    ax.grid(alpha=0.2)
    _show(fig)


def plot_delta_histogram(delta_signal: pd.Series, title: str) -> None:
    """Display the distribution of signed Signal changes."""

    fig, ax = plt.subplots(figsize=(8, 4))
    _set_figure_title(fig, title)
    ax.hist(delta_signal.dropna(), bins=100)
    ax.set(title=title, xlabel="Delta Signal", ylabel="Count")
    ax.grid(alpha=0.2)
    _show(fig)


def plot_abs_delta_histogram(abs_delta_signal: pd.Series, title: str) -> None:
    """Display the distribution of absolute Signal changes."""

    fig, ax = plt.subplots(figsize=(8, 4))
    _set_figure_title(fig, title)
    ax.hist(abs_delta_signal.dropna(), bins=100)
    ax.set(title=title, xlabel="Absolute delta Signal", ylabel="Count")
    ax.grid(alpha=0.2)
    _show(fig)


def plot_signal_with_spike_markers(
    dataframe: pd.DataFrame,
    spikes: pd.DataFrame,
    title: str,
) -> None:
    """Display full Signal with selected spike candidates marked."""

    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    ax.plot(dataframe["Time"], dataframe["Signal"], linewidth=0.4, label="Signal")
    ax.scatter(
        spikes["time_current"],
        spikes["signal_current"],
        color="tab:red",
        s=20,
        label="Top absolute changes",
        zorder=3,
    )
    ax.set(title=title, xlabel="Time", ylabel="Signal")
    ax.legend()
    ax.grid(alpha=0.2)
    _show(fig)


def plot_spike_zoom(
    dataframe: pd.DataFrame,
    spike_time: pd.Timestamp,
    radius: int,
    title: str,
) -> None:
    """Display a row-based zoom around a spike timestamp."""

    if radius < 1:
        raise ValueError("radius must be positive.")
    matches = dataframe.index[dataframe["Time"].eq(spike_time)]
    if matches.empty:
        return
    center = int(matches[0])
    zoom = dataframe.iloc[max(0, center - radius) : center + radius + 1]
    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    ax.plot(zoom["Time"], zoom["Signal"], marker=".", linewidth=0.8, markersize=2)
    ax.axvline(spike_time, color="tab:red", linestyle="--", label="Spike candidate")
    ax.set(title=title, xlabel="Time", ylabel="Signal")
    ax.legend()
    ax.grid(alpha=0.2)
    _show(fig)


def plot_normalization_timeseries(
    transformed: pd.DataFrame,
    transformation: str,
    title: str,
) -> None:
    """Display one temporary normalization variant over time."""

    fig, ax = plt.subplots(figsize=(12, 4))
    _set_figure_title(fig, title)
    ax.plot(transformed["Time"], transformed[transformation], linewidth=0.5)
    ax.set(title=title, xlabel="Time", ylabel=transformation)
    ax.grid(alpha=0.2)
    _show(fig)


def plot_normalization_histogram(
    transformed: pd.DataFrame,
    transformation: str,
    title: str,
) -> None:
    """Display one temporary normalization variant as a histogram."""

    fig, ax = plt.subplots(figsize=(8, 4))
    _set_figure_title(fig, title)
    ax.hist(transformed[transformation].dropna(), bins=100)
    ax.set(title=title, xlabel=transformation, ylabel="Count")
    ax.grid(alpha=0.2)
    _show(fig)
