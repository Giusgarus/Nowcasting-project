"""Multi-panel Matplotlib plots for switch-method comparisons."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_switch_methods_for_event(
    event_frame: pd.DataFrame,
    switch_methods: Mapping[str, Sequence[int] | np.ndarray],
    save_path: str | Path,
    *,
    threshold: float,
    event_title: str | None = None,
    show_event_timestamp: bool = True,
) -> None:
    """Plot one vertical signal-and-switch subplot for each supplied method."""

    if not switch_methods:
        raise ValueError("At least one switch method is required.")
    frame = event_frame.sort_values("Time", kind="stable")
    times = pd.to_datetime(frame["Time"])
    signal = frame["Signal_true"].to_numpy(dtype=float)
    event_timestamp = (
        pd.Timestamp(frame["event_timestamp"].dropna().iloc[0])
        if "event_timestamp" in frame and frame["event_timestamp"].notna().any()
        else None
    )

    figure, axes = plt.subplots(
        len(switch_methods),
        1,
        figsize=(15, 4.5 * len(switch_methods)),
        sharex=True,
        squeeze=False,
    )
    for subplot_index, (method_name, switch_values) in enumerate(
        switch_methods.items()
    ):
        switch = np.asarray(switch_values).ravel()
        if len(switch) != len(frame):
            raise ValueError(f"Switch length mismatch for method '{method_name}'.")

        switch_axis = axes[subplot_index, 0]
        switch_axis.plot(
            times,
            switch,
            color="steelblue",
            linewidth=2,
            drawstyle="steps-post",
            label=method_name,
        )
        switch_axis.set_title(method_name, fontsize=12, fontweight="bold")
        switch_axis.set_ylabel("Switch")
        switch_axis.set_yticks([0, 1])
        switch_axis.set_ylim([-0.1, 1.1])
        switch_axis.grid(True, which="major", axis="y", linestyle="--", alpha=0.7)

        signal_axis = switch_axis.twinx()
        signal_axis.plot(
            times,
            signal,
            color="gray",
            linewidth=1.5,
            alpha=0.9,
            label="True Signal",
        )
        signal_axis.axhline(
            y=threshold,
            color="orangered",
            linewidth=2,
            label=f"{threshold:g} Threshold",
        )
        if show_event_timestamp and subplot_index == 0 and event_timestamp is not None:
            signal_axis.axvline(
                x=event_timestamp,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label="Event Timestamp",
            )
        signal_axis.set_ylabel("Signal")

        lines, labels = switch_axis.get_legend_handles_labels()
        signal_lines, signal_labels = signal_axis.get_legend_handles_labels()
        switch_axis.legend(
            lines + signal_lines,
            labels + signal_labels,
            loc="upper right",
        )

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    figure.autofmt_xdate()
    if event_title:
        figure.suptitle(event_title, fontsize=13, fontweight="bold")
    figure.tight_layout()

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_perfect_switch_for_event(
    event_frame: pd.DataFrame,
    save_path: str | Path,
    *,
    threshold: float,
    switch_time: int,
    quality_flag: str,
    additional_switch_methods: Mapping[str, Sequence[int] | np.ndarray] | None = None,
) -> None:
    """Plot Perfect Switch variants and optional future methods in separate panels."""

    frame = event_frame.sort_values("Time", kind="stable")
    event_id = str(frame["event_id"].iloc[0])
    dataset_name = str(frame["dataset_name"].iloc[0])
    methods: dict[str, Sequence[int] | np.ndarray] = {
        "Perfect Switch": frame["perfect_switch"],
    }
    methods.update(additional_switch_methods or {})
    plot_switch_methods_for_event(
        frame,
        methods,
        save_path,
        threshold=threshold,
        event_title=(
            f"{event_id} | {dataset_name} | threshold={threshold:g} | "
            f"switch_time={switch_time} | quality={quality_flag}"
        ),
    )
