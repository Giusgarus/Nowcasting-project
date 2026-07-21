"""Multi-panel Matplotlib plots for switch-method comparisons."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_centered_event_display_frame(
    signal_source: pd.DataFrame,
    event_frame: pd.DataFrame,
    *,
    before_minutes: float,
    after_minutes: float,
    signal_column: str = "Signal",
) -> pd.DataFrame:
    """Return a signal-only display window centered on an event timestamp."""

    required_signal_columns = {"dataset_id", "dataset_name", "Time", signal_column}
    missing_signal = sorted(required_signal_columns - set(signal_source.columns))
    if missing_signal:
        raise ValueError(f"signal_source is missing columns: {missing_signal}")
    if event_frame.empty:
        raise ValueError("event_frame cannot be empty.")

    event = event_frame.sort_values("Time", kind="stable").iloc[0]
    center = (
        pd.Timestamp(event["event_timestamp"])
        if "event_timestamp" in event_frame.columns and pd.notna(event["event_timestamp"])
        else pd.Timestamp(event["Time"])
    )
    start = center - pd.Timedelta(minutes=float(before_minutes))
    end = center + pd.Timedelta(minutes=float(after_minutes))

    source = signal_source.copy()
    source["Time"] = pd.to_datetime(source["Time"], errors="raise")
    selected = source.loc[
        source["dataset_id"].astype(str).eq(str(event["dataset_id"]))
        & source["dataset_name"].astype(str).eq(str(event["dataset_name"]))
        & source["Time"].ge(start)
        & source["Time"].le(end),
        ["dataset_id", "dataset_name", "Time", signal_column],
    ].copy()
    if selected.empty:
        raise ValueError(
            "No signal samples found for centered display window "
            f"around {center}."
        )
    selected = selected.sort_values("Time", kind="stable").reset_index(drop=True)
    selected["Signal_true"] = selected[signal_column].astype(float)
    if signal_column != "Signal_true":
        selected = selected.drop(columns=signal_column)

    metadata_columns = [
        "event_id",
        "global_event_id",
        "split",
        "quality_flag",
        "event_timestamp",
        "threshold",
        "switch_time",
    ]
    for column in metadata_columns:
        if column in event_frame.columns:
            selected[column] = event_frame[column].dropna().iloc[0]
    if "event_timestamp" not in selected.columns:
        selected["event_timestamp"] = center
    return selected


def map_switch_to_display_frame(
    display_frame: pd.DataFrame,
    switch_frame: pd.DataFrame,
    switch_column: str,
    *,
    fill_value: int = 0,
) -> np.ndarray:
    """Map native-task switch values to a display frame, filling missing times."""

    if switch_column not in switch_frame.columns:
        raise ValueError(f"switch_frame is missing column '{switch_column}'.")
    if "Time" not in display_frame.columns or "Time" not in switch_frame.columns:
        raise ValueError("Both display_frame and switch_frame must contain Time.")

    lookup = switch_frame.copy()
    lookup["Time"] = pd.to_datetime(lookup["Time"], errors="raise")
    lookup = lookup.drop_duplicates(subset=["Time"], keep="last")
    series = lookup.set_index("Time")[switch_column]
    times = pd.to_datetime(display_frame["Time"], errors="raise")
    mapped = times.map(series).fillna(fill_value)
    return mapped.to_numpy(dtype=np.int8)


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
