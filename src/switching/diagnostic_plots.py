"""Diagnostic plots for task-local and cross-task switch comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path

_PLOT_CACHE_DIR = Path.cwd() / ".matplotlib-cache"
_XDG_CACHE_DIR = Path.cwd() / ".cache"
_PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.switch_metrics import compute_switch_metrics
from src.switching.cross_task import SwitchComparisonSource, read_method_frame

TASK_COLORS = {
    "autoregressive": "#1f77b4",
    "current_level_persistence": "#2ca02c",
    "long_fade_detection": "#ff7f0e",
    "survival_persistence": "#9467bd",
}


def save_figure(figure: plt.Figure, path: str | Path) -> None:
    """Save and close one figure."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def compact_method_label(row: pd.Series | Mapping[str, object]) -> str:
    """Return a readable task-method label."""

    task_name = str(row["task_name"])
    task_label = {
        "autoregressive": "AR",
        "current_level_persistence": "CLP",
        "long_fade_detection": "LFD",
        "survival_persistence": "SURV",
    }.get(task_name, task_name)
    return f"{task_label} | {row['method_name']}"


def plot_metric_bars(
    metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    metric_columns: Sequence[str] = ("f1", "precision", "recall", "balanced_accuracy"),
) -> None:
    """Plot horizontal bar charts for core switch metrics."""

    if metrics.empty:
        return
    frame = metrics.sort_values("f1", ascending=True).copy()
    labels = [compact_method_label(row) for _, row in frame.iterrows()]
    colors = [TASK_COLORS.get(str(task), "gray") for task in frame["task_name"]]

    if len(metric_columns) == 4:
        figure, axes_grid = plt.subplots(
            2,
            2,
            figsize=(15.0, max(9.0, 0.72 * len(frame))),
            sharey=True,
        )
        axes = list(axes_grid.ravel())
    else:
        figure, axes_row = plt.subplots(
            1,
            len(metric_columns),
            figsize=(5.0 * len(metric_columns), max(6.0, 0.42 * len(frame))),
            sharey=True,
        )
        axes = [axes_row] if len(metric_columns) == 1 else list(axes_row)
    y_positions = np.arange(len(frame))
    for axis, metric in zip(axes, metric_columns, strict=True):
        values = frame[metric].astype(float).to_numpy()
        axis.barh(y_positions, values, color=colors, alpha=0.85)
        axis.set_title(metric)
        axis.set_xlim(0.0, 1.05)
        axis.grid(axis="x", linestyle="--", alpha=0.4)
    for index, axis in enumerate(axes):
        axis.set_yticks(y_positions)
        if len(metric_columns) == 4 and index % 2 == 1:
            axis.tick_params(axis="y", labelleft=False)
        elif len(metric_columns) != 4 and index > 0:
            axis.tick_params(axis="y", labelleft=False)
        else:
            axis.set_yticklabels(labels, fontsize=8)
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_precision_recall_scatter(
    metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    annotate_top_n: int = 8,
) -> None:
    """Plot precision against recall, with marker size representing coverage."""

    if metrics.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 7))
    for task_name, group in metrics.groupby("task_name", sort=True):
        coverage = group["coverage_pct"].fillna(0).clip(lower=0)
        sizes = 35 + 3.0 * coverage
        axis.scatter(
            group["recall"],
            group["precision"],
            s=sizes,
            alpha=0.75,
            color=TASK_COLORS.get(str(task_name), "gray"),
            label=str(task_name),
            edgecolor="white",
            linewidth=0.8,
        )

    top = metrics.sort_values("f1", ascending=False).head(annotate_top_n)
    for _, row in top.iterrows():
        axis.annotate(
            compact_method_label(row),
            (float(row["recall"]), float(row["precision"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Recall vs Perfect Switch")
    axis.set_ylabel("Precision vs Perfect Switch")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(title="Task", loc="lower left")
    axis.set_title(title, fontsize=13, fontweight="bold")
    save_figure(figure, output_path)


def plot_coverage(
    coverage: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot native decision coverage against the common reference grid."""

    if coverage.empty:
        return
    frame = coverage.sort_values("coverage_pct", ascending=True).copy()
    labels = [compact_method_label(row) for _, row in frame.iterrows()]
    colors = [TASK_COLORS.get(str(task), "gray") for task in frame["task_name"]]

    figure, axis = plt.subplots(figsize=(12, max(6.0, 0.42 * len(frame))))
    y_positions = np.arange(len(frame))
    axis.barh(y_positions, frame["coverage_pct"], color=colors, alpha=0.85)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Native decision coverage on reference grid (%)")
    axis.set_xlim(0, 105)
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    for y_pos, (_, row) in zip(y_positions, frame.iterrows(), strict=True):
        axis.text(
            float(row["coverage_pct"]) + 1.0,
            y_pos,
            f"missed perfect+={int(row['perfect_positive_points_without_native_decision'])}",
            va="center",
            fontsize=7,
        )
    axis.set_title(title, fontsize=13, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_raw_vs_postprocessed_counts(
    metrics: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot raw and post-processed positive switch counts."""

    if metrics.empty:
        return
    frame = metrics.sort_values("num_model_positive", ascending=True).copy()
    labels = [compact_method_label(row) for _, row in frame.iterrows()]
    y_positions = np.arange(len(frame))
    height = 0.36

    figure, axis = plt.subplots(figsize=(13, max(6.0, 0.45 * len(frame))))
    axis.barh(
        y_positions - height / 2,
        frame["num_model_positive_raw"],
        height=height,
        color="#9ecae1",
        label="Raw switch positives",
    )
    axis.barh(
        y_positions + height / 2,
        frame["num_model_positive"],
        height=height,
        color="#08519c",
        label="Post-processed switch positives",
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Positive switch samples")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    axis.legend(loc="lower right")
    axis.set_title(title, fontsize=13, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output_path)


def select_reference_events(
    reference_grid: pd.DataFrame,
    *,
    max_events: int,
    event_column: str = "event_id",
) -> list[str]:
    """Select chronologically ordered events with Perfect Switch activity."""

    frame = reference_grid.copy()
    frame["Time"] = pd.to_datetime(frame["Time"], errors="raise")
    grouped = (
        frame.groupby(event_column, dropna=False)
        .agg(
            first_time=("Time", "min"),
            perfect_positive=("perfect_switch", "sum"),
            samples=("perfect_switch", "size"),
        )
        .reset_index()
    )
    positives = grouped.loc[grouped["perfect_positive"].gt(0)].copy()
    if positives.empty:
        positives = grouped.copy()
    selected = positives.sort_values("first_time", kind="stable").head(max_events)
    return [str(value) for value in selected[event_column].tolist()]


def compute_reference_event_f1_matrix(
    reference_grid: pd.DataFrame,
    sources: Sequence[SwitchComparisonSource],
    *,
    event_ids: Sequence[str],
    missing_model_switch: int,
    event_column: str = "event_id",
) -> pd.DataFrame:
    """Compute per-event F1 on a common reference grid for a set of methods."""

    reference = reference_grid[["Time", event_column, "perfect_switch"]].copy()
    reference["Time"] = pd.to_datetime(reference["Time"], errors="raise")
    reference[event_column] = reference[event_column].astype(str)
    reference = reference.loc[reference[event_column].isin([str(item) for item in event_ids])]

    rows: list[dict[str, float | str]] = []
    for source in sources:
        method = read_method_frame(source)
        aligned = reference.merge(method, on="Time", how="left", validate="one_to_one")
        aligned["model_switch"] = aligned["model_switch"].fillna(missing_model_switch)
        row: dict[str, float | str] = {"method": compact_method_label(source.__dict__)}
        for event_id, event_frame in aligned.groupby(event_column, sort=False):
            metrics = compute_switch_metrics(
                event_frame["perfect_switch"].astype("int8").tolist(),
                event_frame["model_switch"].astype("int8").tolist(),
            )
            row[str(event_id)] = metrics.f1
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def plot_event_metric_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot a method-by-event F1 heatmap."""

    if matrix.empty:
        return
    values = matrix.to_numpy(dtype=float)
    figure, axis = plt.subplots(
        figsize=(max(10.0, 0.45 * len(matrix.columns)), max(6.0, 0.38 * len(matrix)))
    )
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels([str(col).split("_")[-1] for col in matrix.columns], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index, fontsize=8)
    axis.set_xlabel("Event")
    axis.set_title(title, fontsize=13, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("F1 vs Perfect Switch")
    figure.tight_layout()
    save_figure(figure, output_path)


def map_switch_to_times(
    times: pd.Series,
    method_frame: pd.DataFrame,
    column: str,
    *,
    fill_value: int = 0,
) -> np.ndarray:
    """Map one method switch column onto a target timestamp grid."""

    frame = method_frame[["Time", column]].copy()
    frame["Time"] = pd.to_datetime(frame["Time"], errors="raise")
    series = frame.drop_duplicates("Time", keep="last").set_index("Time")[column]
    mapped = pd.to_datetime(times, errors="raise").map(series).fillna(fill_value)
    return mapped.to_numpy(dtype=np.int8)


def plot_switch_timeline_rows(
    frame: pd.DataFrame,
    switches: Mapping[str, Sequence[int] | np.ndarray],
    output_path: str | Path,
    *,
    threshold: float,
    title: str,
    raw_switches: Mapping[str, Sequence[int] | np.ndarray] | None = None,
) -> None:
    """Plot signal and many switch rows in a compact timeline."""

    if not switches:
        return
    data = frame.sort_values("Time", kind="stable").copy()
    times = pd.to_datetime(data["Time"], errors="raise")
    signal = data["Signal_true"].astype(float).to_numpy()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15, max(6.0, 0.45 * len(switches) + 3.5)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, max(1.0, 0.25 * len(switches))]},
    )
    signal_axis, switch_axis = axes
    signal_axis.plot(times, signal, color="gray", linewidth=1.5, label="Signal")
    signal_axis.axhline(threshold, color="orangered", linewidth=1.8, label=f"{threshold:g} threshold")
    signal_axis.set_ylabel("Signal")
    signal_axis.grid(True, linestyle="--", alpha=0.35)
    signal_axis.legend(loc="upper right")

    labels = list(switches.keys())
    y_positions = np.arange(len(labels))
    raw_switches = raw_switches or {}
    for y_pos, label in zip(y_positions, labels, strict=True):
        values = np.asarray(switches[label]).astype(int)
        color = "#444444" if label == "Perfect Switch" else None
        switch_axis.fill_between(
            times,
            y_pos - 0.35,
            y_pos + 0.35,
            where=values > 0,
            step="post",
            alpha=0.55,
            color=color,
        )
        if label in raw_switches:
            raw = np.asarray(raw_switches[label]).astype(int)
            raw_times = times[raw > 0]
            if len(raw_times):
                switch_axis.scatter(
                    raw_times,
                    np.full(len(raw_times), y_pos + 0.42),
                    marker="|",
                    color="black",
                    s=18,
                    linewidths=0.8,
                )
    switch_axis.set_yticks(y_positions)
    switch_axis.set_yticklabels(labels, fontsize=8)
    switch_axis.set_ylim(-0.75, len(labels) - 0.25)
    switch_axis.set_xlabel("Time")
    switch_axis.set_title("Filled bands = post-processed switch; black ticks = raw switch")
    switch_axis.grid(axis="x", linestyle="--", alpha=0.35)
    switch_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.autofmt_xdate()
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_intra_task_switch_event(
    event_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot one method's diagnostic signal, score, raw switch, and final switch."""

    frame = event_frame.sort_values("Time", kind="stable").copy()
    times = pd.to_datetime(frame["Time"], errors="raise")
    threshold = float(frame["threshold"].dropna().iloc[0]) if "threshold" in frame else 10.0

    diagnostic_columns = [
        ("predicted_remaining_persistence_seconds", "Predicted persistence seconds"),
        ("true_remaining_persistence_seconds", "True persistence seconds"),
        ("prob_long_fade", "P(long fade)"),
        ("survival_probability_300s", "S(300s)"),
        ("survival_probability", "Survival probability"),
        ("min_predicted_signal", "Min predicted future signal"),
        ("Signal_predicted", "Selected predicted signal"),
    ]
    available_diagnostics = [
        (column, label) for column, label in diagnostic_columns if column in frame.columns
    ]

    figure, axes = plt.subplots(
        3 if available_diagnostics else 2,
        1,
        figsize=(15, 9 if available_diagnostics else 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0] if available_diagnostics else [1.2, 1.0]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    signal_axis = axes[0]
    signal_axis.plot(times, frame["Signal_true"], color="gray", linewidth=1.5, label="Signal")
    signal_axis.axhline(threshold, color="orangered", linewidth=1.8, label=f"{threshold:g} threshold")
    signal_axis.set_ylabel("Signal")
    signal_axis.grid(True, linestyle="--", alpha=0.35)
    signal_axis.legend(loc="upper right")

    switch_axis = axes[-1]
    switch_rows = {
        "Perfect Switch": frame["perfect_switch"].to_numpy(),
        "Model raw": frame["model_switch_raw"].to_numpy(),
        "Model postprocessed": frame["model_switch"].to_numpy(),
    }
    for y_pos, (label, values) in enumerate(switch_rows.items()):
        switch_axis.fill_between(
            times,
            y_pos - 0.3,
            y_pos + 0.3,
            where=np.asarray(values).astype(int) > 0,
            step="post",
            alpha=0.6,
        )
    switch_axis.set_yticks(np.arange(len(switch_rows)))
    switch_axis.set_yticklabels(list(switch_rows.keys()))
    switch_axis.set_ylim(-0.6, len(switch_rows) - 0.4)
    switch_axis.grid(axis="x", linestyle="--", alpha=0.35)
    switch_axis.set_xlabel("Time")
    switch_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    if available_diagnostics:
        diagnostic_axis = axes[1]
        for column, label in available_diagnostics[:3]:
            diagnostic_axis.plot(times, frame[column], linewidth=1.4, label=label)
        if "duration_threshold_seconds" in frame.columns:
            diagnostic_axis.axhline(
                float(frame["duration_threshold_seconds"].dropna().iloc[0]),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Duration threshold",
            )
            diagnostic_axis.set_ylabel("Seconds")
        elif "probability_threshold" in frame.columns:
            diagnostic_axis.axhline(
                float(frame["probability_threshold"].dropna().iloc[0]),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Probability threshold",
            )
            diagnostic_axis.set_ylabel("Probability")
            diagnostic_axis.set_ylim(-0.05, 1.05)
        elif "survival_probability_threshold" in frame.columns:
            diagnostic_axis.axhline(
                float(frame["survival_probability_threshold"].dropna().iloc[0]),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Survival threshold",
            )
            diagnostic_axis.set_ylabel("Probability")
            diagnostic_axis.set_ylim(-0.05, 1.05)
        else:
            diagnostic_axis.axhline(threshold, color="orangered", linestyle="--", linewidth=1.2)
            diagnostic_axis.set_ylabel("Diagnostic")
        diagnostic_axis.grid(True, linestyle="--", alpha=0.35)
        diagnostic_axis.legend(loc="upper right")

    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.autofmt_xdate()
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_autoregressive_horizon_errors(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot MAE and RMSE by autoregressive horizon."""

    if predictions.empty or "horizon_step" not in predictions.columns:
        return
    grouped = predictions.groupby("horizon_step", as_index=False).agg(
        mae=("absolute_error", "mean"),
        rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(grouped["horizon_step"], grouped["mae"], marker="o", label="MAE")
    axis.plot(grouped["horizon_step"], grouped["rmse"], marker="o", label="RMSE")
    axis.set_xlabel("Forecast horizon step")
    axis.set_ylabel("Raw-scale error")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    axis.set_title(title, fontsize=13, fontweight="bold")
    save_figure(figure, output_path)


def plot_autoregressive_forecast_example(
    predictions: pd.DataFrame,
    reference_grid: pd.DataFrame,
    window_id: str,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot one autoregressive context/future forecast example."""

    window = predictions.loc[predictions["window_id"].astype(str).eq(str(window_id))].copy()
    if window.empty:
        return
    window = window.sort_values("horizon_step", kind="stable")
    first = window.iloc[0]
    input_start = pd.Timestamp(first["input_start_time"])
    input_end = pd.Timestamp(first["input_end_time"])
    target_start = pd.Timestamp(first["target_start_time"])
    sample_interval = (
        pd.Timestamp(first["target_end_time"]) - target_start
    ) / max(int(window["horizon_step"].max()) - 1, 1)
    target_times = [
        target_start + int(step - 1) * sample_interval
        for step in window["horizon_step"].astype(int)
    ]
    ref = reference_grid.copy()
    ref["Time"] = pd.to_datetime(ref["Time"], errors="raise")
    context = ref.loc[ref["Time"].between(input_start, input_end)].copy()

    figure, axis = plt.subplots(figsize=(12, 5))
    if not context.empty:
        axis.plot(
            context["Time"],
            context["Signal_true"],
            color="gray",
            linewidth=1.5,
            label="Past context",
        )
    axis.plot(target_times, window["y_true_raw"], color="black", marker="o", label="True future")
    axis.plot(target_times, window["y_pred_raw"], color="#1f77b4", marker="o", label="Predicted future")
    axis.axhline(10.0, color="orangered", linewidth=1.4, label="10 threshold")
    axis.axvline(input_end, color="black", linestyle=":", linewidth=1.0, label="Decision time")
    axis.set_ylabel("Signal")
    axis.set_xlabel("Time")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="upper right")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.autofmt_xdate()
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_duration_scatter(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot predicted versus true current-level persistence durations."""

    required = {"y_true_seconds", "y_pred_seconds"}
    if predictions.empty or not required.issubset(predictions.columns):
        return
    frame = predictions.dropna(subset=list(required)).copy()
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        frame["y_true_seconds"] / 60.0,
        frame["y_pred_seconds"] / 60.0,
        s=10,
        alpha=0.35,
    )
    limit = float(np.nanpercentile(frame[["y_true_seconds", "y_pred_seconds"]].to_numpy(), 99) / 60.0)
    limit = max(limit, 5.0)
    axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1.2, label="Ideal")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("True duration (minutes)")
    axis.set_ylabel("Predicted duration (minutes)")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    axis.set_title(title, fontsize=13, fontweight="bold")
    save_figure(figure, output_path)


def plot_duration_error_distribution(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot distribution of absolute duration errors."""

    if predictions.empty or "absolute_error_minutes" not in predictions.columns:
        return
    errors = predictions["absolute_error_minutes"].dropna()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(errors.clip(upper=errors.quantile(0.99)), bins=40, color="#2ca02c", alpha=0.8)
    axis.set_xlabel("Absolute error (minutes, clipped at p99)")
    axis.set_ylabel("Windows")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_title(title, fontsize=13, fontweight="bold")
    save_figure(figure, output_path)


def plot_probability_calibration(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    probability_column: str,
    target_column: str,
    title: str,
    bins: int = 10,
) -> None:
    """Plot a simple reliability curve for binary probability predictions."""

    required = {probability_column, target_column}
    if frame.empty or not required.issubset(frame.columns):
        return
    data = frame.dropna(subset=[probability_column, target_column]).copy()
    data[probability_column] = data[probability_column].clip(0, 1)
    data["probability_bin"] = pd.cut(
        data[probability_column],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )
    grouped = data.groupby("probability_bin", observed=False).agg(
        mean_probability=(probability_column, "mean"),
        observed_frequency=(target_column, "mean"),
        count=(target_column, "size"),
    )
    grouped = grouped.dropna(subset=["mean_probability", "observed_frequency"])

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Ideal")
    axis.scatter(
        grouped["mean_probability"],
        grouped["observed_frequency"],
        s=30 + grouped["count"].astype(float),
        alpha=0.75,
        color="#ff7f0e",
        label="Bins",
    )
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    axis.set_title(title, fontsize=13, fontweight="bold")
    save_figure(figure, output_path)


def plot_probability_distribution(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    probability_column: str,
    title: str,
) -> None:
    """Plot a probability distribution."""

    if frame.empty or probability_column not in frame.columns:
        return
    values = frame[probability_column].dropna().clip(0, 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=np.linspace(0, 1, 31), color="#9467bd", alpha=0.8)
    axis.set_xlabel(probability_column)
    axis.set_ylabel("Windows")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_title(title, fontsize=13, fontweight="bold")
    save_figure(figure, output_path)


def plot_survival_curves(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    max_curves: int = 12,
) -> None:
    """Plot representative saved survival curves across the prediction range."""

    probability_columns = [
        column
        for column in predictions.columns
        if column.startswith("survival_probability_") and column.endswith("s")
    ]
    if predictions.empty or not probability_columns:
        return
    horizons = [float(column.removeprefix("survival_probability_").removesuffix("s")) for column in probability_columns]
    order = np.argsort(horizons)
    horizons = np.asarray(horizons)[order]
    probability_columns = [probability_columns[index] for index in order]

    frame = predictions.copy().reset_index(drop=True)
    time_column = "sample_time" if "sample_time" in frame.columns else "Time"
    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    selection_column = (
        "survival_probability_300s"
        if "survival_probability_300s" in frame.columns
        else probability_columns[len(probability_columns) // 2]
    )
    valid = frame.loc[frame[selection_column].notna()].copy()
    if valid.empty:
        return
    n_curves = min(max(1, int(max_curves)), len(valid))
    quantiles = np.linspace(0.05, 0.95, n_curves) if n_curves > 1 else np.asarray([0.5])
    score_values = valid[selection_column].to_numpy(dtype=float)
    selected_positions: list[int] = []
    selected_quantiles: list[float] = []
    for quantile in quantiles:
        target = float(np.quantile(score_values, quantile))
        candidates = np.argsort(np.abs(score_values - target), kind="stable")
        position = next(int(value) for value in candidates if int(value) not in selected_positions)
        selected_positions.append(position)
        selected_quantiles.append(float(quantile))
    selected = valid.iloc[selected_positions].copy()
    selected["_selection_quantile"] = selected_quantiles

    figure, axis = plt.subplots(figsize=(10.0, 8.2))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(selected)))
    for color, (_, row) in zip(colors, selected.iterrows(), strict=True):
        quantile_pct = int(round(float(row["_selection_quantile"]) * 100))
        survival_300s = float(row[selection_column])
        median_seconds = float(row.get("predicted_median_remaining_seconds", np.nan))
        observed_seconds = float(row.get("y_time_seconds", np.nan))
        observed = bool(row.get("y_event_observed", True))
        median_label = f"{median_seconds / 60.0:.1f} min" if np.isfinite(median_seconds) else "n/a"
        if np.isfinite(observed_seconds):
            observed_label = f"{observed_seconds / 60.0:.1f} min"
            if not observed:
                observed_label = f">={observed_label}"
        else:
            observed_label = "n/a"
        label = (
            f"q{quantile_pct:02d} | S(5 min)={survival_300s:.2f}\n"
            f"pred. median={median_label} | observed={observed_label}"
        )
        axis.plot(
            horizons / 60.0,
            row[probability_columns].to_numpy(dtype=float),
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=label,
        )
    axis.axvline(5.0, color="black", linestyle=":", linewidth=1.2, label="Decision horizon (5 min)")
    axis.set_xlabel("Horizon (minutes)")
    axis.set_ylabel("Survival probability")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(
        fontsize=8.0,
        title="Representative test windows",
        title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        columnspacing=1.5,
        handlelength=2.5,
    )
    axis.set_title(title, fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.25, 1.0, 1.0))
    save_figure(figure, output_path)
