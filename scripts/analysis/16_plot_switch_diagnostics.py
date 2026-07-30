"""Generate intra-task and cross-task switch diagnostic plots."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.switching.cross_task import (  # noqa: E402
    SwitchComparisonSource,
    build_reference_grid_metrics,
    build_strict_intersection_metrics,
    discover_switch_sources,
    load_reference_grid,
    read_method_frame,
)
from src.switching.diagnostic_plots import (  # noqa: E402
    compact_method_label,
    compute_reference_event_f1_matrix,
    map_switch_to_times,
    plot_autoregressive_forecast_example,
    plot_autoregressive_horizon_errors,
    plot_coverage,
    plot_duration_error_distribution,
    plot_duration_scatter,
    plot_event_metric_heatmap,
    plot_intra_task_switch_event,
    plot_metric_bars,
    plot_precision_recall_scatter,
    plot_probability_calibration,
    plot_probability_distribution,
    plot_raw_vs_postprocessed_counts,
    plot_survival_curves,
    plot_switch_timeline_rows,
    select_reference_events,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    COMPARISON_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_results_index_dir,
    relative_project_path,
    sanitize_id,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/switch_diagnostics.yaml"


def project_path(value: str | Path) -> Path:
    """Resolve a possibly relative path inside the project."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def cross_task_output_dir(cross_config: dict) -> Path:
    """Return the configured cross-task comparison output directory."""

    comparison = cross_config["comparison"]
    comparison_id = sanitize_id(str(comparison["comparison_id"]))
    normalized_selection_id = sanitize_id(str(comparison["normalized_selection_id"]))
    return (
        project_path(cross_config["output"]["results_root"])
        / normalized_selection_id
        / comparison_id
    )


def select_switch_event_ids(frame: pd.DataFrame, *, max_events: int) -> list[str]:
    """Select chronologically ordered event IDs that are useful for timeline plots."""

    if "event_id" not in frame.columns:
        return []
    data = frame.copy()
    data["Time"] = pd.to_datetime(data["Time"], errors="raise")
    grouped = (
        data.groupby("event_id", dropna=False)
        .agg(
            first_time=("Time", "min"),
            perfect_positive=("perfect_switch", "sum"),
            model_positive=("model_switch", "sum"),
        )
        .reset_index()
    )
    useful = grouped.loc[
        grouped["perfect_positive"].gt(0) | grouped["model_positive"].gt(0)
    ].copy()
    if useful.empty:
        useful = grouped.copy()
    selected = useful.sort_values("first_time", kind="stable").head(max_events)
    return [str(value) for value in selected["event_id"].tolist()]


def read_run_index(path: Path) -> pd.DataFrame:
    """Read an optional run index."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_predictions_path(
    source: SwitchComparisonSource,
    switch_frame: pd.DataFrame,
    *,
    run_indexes: dict[str, pd.DataFrame],
) -> Path | None:
    """Resolve the best available test-prediction parquet for one source."""

    run_id = source.method_id
    if "run_id" in switch_frame.columns and switch_frame["run_id"].notna().any():
        run_id = str(switch_frame["run_id"].dropna().iloc[0])

    if source.task_name == "autoregressive":
        index = run_indexes.get("autoregressive", pd.DataFrame())
        if not index.empty:
            match = index.loc[index["run_id"].astype(str).eq(run_id)]
            if not match.empty and "predictions_path" in match.columns:
                candidate = project_path(str(match.iloc[0]["predictions_path"]))
                if candidate.exists():
                    return candidate

    task_candidates = [
        PROJECT_ROOT
        / "results"
        / "runs"
        / source.task_name
        / source.selection_id
        / run_id
        / "predictions"
        / "test_predictions.parquet",
        PROJECT_ROOT
        / "results"
        / "runs"
        / source.task_name
        / source.selection_id
        / run_id.replace("__", "_")
        / "predictions"
        / "test_predictions.parquet",
    ]
    for candidate in task_candidates:
        if candidate.exists():
            return candidate
    return None


def read_predictions(path: Path | None) -> pd.DataFrame:
    """Read a prediction parquet when it exists."""

    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def survival_binary_target_at_horizon(frame: pd.DataFrame, horizon_seconds: float) -> pd.Series:
    """Return known binary survival-at-horizon labels, with unknown censoring as NaN."""

    if not {"y_time_seconds", "y_event_observed"}.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    time = frame["y_time_seconds"].astype(float)
    observed = frame["y_event_observed"].astype(int)
    target = pd.Series(np.nan, index=frame.index, dtype=float)
    target.loc[time.gt(horizon_seconds)] = 1.0
    target.loc[observed.eq(1) & time.le(horizon_seconds)] = 0.0
    return target


def generate_intra_task_plots(
    *,
    sources: list[SwitchComparisonSource],
    reference_grid: pd.DataFrame,
    config: dict,
    run_indexes: dict[str, pd.DataFrame],
) -> list[dict[str, str]]:
    """Generate task-local diagnostic plots from saved switch comparisons."""

    manifest: list[dict[str, str]] = []
    plots_config = config["plots"]
    root = project_path(config["output"]["switch_diagnostics_root"])

    for source in sources:
        switch_frame = pd.read_parquet(source.predictions_path)
        task_dir = (
            root
            / source.task_name
            / sanitize_id(source.selection_id)
            / sanitize_id(source.method_id)
        )
        event_timelines_dir = task_dir / "figures" / "event_timelines"
        task_specific_dir = task_dir / "figures" / "task_specific"
        forecast_examples_dir = task_dir / "figures" / "forecast_examples"

        event_ids = select_switch_event_ids(
            switch_frame,
            max_events=int(plots_config["max_events_per_method"]),
        )
        for position, event_id in enumerate(event_ids, start=1):
            event_frame = switch_frame.loc[
                switch_frame["event_id"].astype(str).eq(str(event_id))
            ].copy()
            path = (
                event_timelines_dir
                / f"event_{position:03d}_{sanitize_id(event_id)}_raw_vs_postprocessed.png"
            )
            plot_intra_task_switch_event(
                event_frame,
                path,
                title=f"{source.task_name} | {source.method_name} | {event_id}",
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "event_raw_vs_postprocessed_timeline",
                    "path": relative_project_path(path),
                }
            )

        prediction_path = run_predictions_path(
            source,
            switch_frame,
            run_indexes=run_indexes,
        )
        predictions = read_predictions(prediction_path)

        if source.task_name == "autoregressive" and not predictions.empty:
            path = task_specific_dir / "horizon_error.png"
            plot_autoregressive_horizon_errors(
                predictions,
                path,
                title=f"{source.method_name} | horizon error",
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "autoregressive_horizon_error",
                    "path": relative_project_path(path),
                }
            )
            candidate_windows = (
                switch_frame.loc[
                    switch_frame["perfect_switch"].gt(0) | switch_frame["model_switch"].gt(0),
                    "window_id",
                ]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(int(plots_config["max_forecast_examples_per_method"]))
            )
            for position, window_id in enumerate(candidate_windows, start=1):
                path = (
                    forecast_examples_dir
                    / f"forecast_{position:03d}_{sanitize_id(window_id)}.png"
                )
                plot_autoregressive_forecast_example(
                    predictions,
                    reference_grid,
                    window_id,
                    path,
                    title=f"{source.method_name} | forecast window {window_id}",
                )
                manifest.append(
                    {
                        "scope": "intra_task",
                        "task_name": source.task_name,
                        "method_name": source.method_name,
                        "plot_type": "autoregressive_forecast_example",
                        "path": relative_project_path(path),
                    }
                )

        if source.task_name == "current_level_persistence":
            duration_frame = predictions
            if duration_frame.empty and {
                "true_remaining_persistence_seconds",
                "predicted_remaining_persistence_seconds",
            }.issubset(switch_frame.columns):
                duration_frame = switch_frame.rename(
                    columns={
                        "true_remaining_persistence_seconds": "y_true_seconds",
                        "predicted_remaining_persistence_seconds": "y_pred_seconds",
                    }
                ).copy()
                duration_frame["absolute_error_minutes"] = (
                    duration_frame["y_pred_seconds"].sub(duration_frame["y_true_seconds"]).abs()
                    / 60.0
                )
            path = task_specific_dir / "duration_predicted_vs_true.png"
            plot_duration_scatter(
                duration_frame,
                path,
                title=f"{source.method_name} | predicted vs true duration",
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "duration_predicted_vs_true",
                    "path": relative_project_path(path),
                }
            )
            path = task_specific_dir / "duration_absolute_error_distribution.png"
            plot_duration_error_distribution(
                duration_frame,
                path,
                title=f"{source.method_name} | duration absolute error",
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "duration_absolute_error_distribution",
                    "path": relative_project_path(path),
                }
            )

        if source.task_name == "long_fade_detection":
            path = task_specific_dir / "probability_calibration.png"
            plot_probability_calibration(
                switch_frame,
                path,
                probability_column="prob_long_fade",
                target_column="y_long_fade",
                title=f"{source.method_name} | long-fade calibration",
                bins=int(plots_config["probability_calibration_bins"]),
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "long_fade_probability_calibration",
                    "path": relative_project_path(path),
                }
            )
            path = task_specific_dir / "probability_distribution.png"
            plot_probability_distribution(
                switch_frame,
                path,
                probability_column="prob_long_fade",
                title=f"{source.method_name} | long-fade probability distribution",
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "long_fade_probability_distribution",
                    "path": relative_project_path(path),
                }
            )

        if source.task_name == "survival_persistence":
            horizon = float(
                switch_frame["survival_horizon_seconds"].dropna().iloc[0]
                if "survival_horizon_seconds" in switch_frame
                and switch_frame["survival_horizon_seconds"].notna().any()
                else 300.0
            )
            calibration_frame = switch_frame.copy()
            calibration_frame["known_survival_at_horizon"] = survival_binary_target_at_horizon(
                calibration_frame,
                horizon,
            )
            path = task_specific_dir / "survival_probability_calibration.png"
            probability_column = (
                "survival_probability_300s"
                if "survival_probability_300s" in calibration_frame.columns
                else "survival_probability"
            )
            plot_probability_calibration(
                calibration_frame,
                path,
                probability_column=probability_column,
                target_column="known_survival_at_horizon",
                title=f"{source.method_name} | survival probability calibration at {horizon:g}s",
                bins=int(plots_config["probability_calibration_bins"]),
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "survival_probability_calibration",
                    "path": relative_project_path(path),
                }
            )
            path = task_specific_dir / "survival_probability_distribution.png"
            plot_probability_distribution(
                calibration_frame,
                path,
                probability_column=probability_column,
                title=f"{source.method_name} | survival probability distribution",
            )
            manifest.append(
                {
                    "scope": "intra_task",
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "plot_type": "survival_probability_distribution",
                    "path": relative_project_path(path),
                }
            )
            if not predictions.empty:
                path = task_specific_dir / "survival_curves.png"
                plot_survival_curves(
                    predictions,
                    path,
                    title=f"{source.method_name} | selected survival curves",
                    max_curves=int(plots_config["max_survival_curves"]),
                )
                manifest.append(
                    {
                        "scope": "intra_task",
                        "task_name": source.task_name,
                        "method_name": source.method_name,
                        "plot_type": "survival_curves",
                        "path": relative_project_path(path),
                    }
                )

    return manifest


def select_cross_task_timeline_sources(
    metrics: pd.DataFrame,
    sources: list[SwitchComparisonSource],
    *,
    max_methods: int,
) -> list[SwitchComparisonSource]:
    """Select top cross-task sources while keeping at least one method per task."""

    by_comparison = {source.comparison_id: source for source in sources}
    selected_ids: list[str] = []
    for _, row in (
        metrics.sort_values("f1", ascending=False)
        .drop_duplicates("task_name", keep="first")
        .iterrows()
    ):
        selected_ids.append(str(row["comparison_id"]))
    for comparison_id in metrics.sort_values("f1", ascending=False)["comparison_id"]:
        value = str(comparison_id)
        if value not in selected_ids:
            selected_ids.append(value)
        if len(selected_ids) >= max_methods:
            break
    return [by_comparison[item] for item in selected_ids if item in by_comparison]


def generate_cross_task_plots(
    *,
    sources: list[SwitchComparisonSource],
    reference_grid: pd.DataFrame,
    cross_config: dict,
    config: dict,
) -> list[dict[str, str]]:
    """Generate cross-task diagnostic plots in the cross-task comparison folder."""

    manifest: list[dict[str, str]] = []
    output_dir = cross_task_output_dir(cross_config)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_paths = ensure_results_subdirs(
        figures_dir,
        (
            "global_metrics",
            "precision_recall",
            "coverage",
            "raw_vs_postprocessed",
            "event_heatmaps",
            "event_timelines/selected_methods",
            "event_timelines/all_methods",
        ),
    )
    tables_dir.mkdir(parents=True, exist_ok=True)

    denominators = list(cross_config["evaluation"]["denominators"])
    missing_model_switch = int(cross_config["evaluation"].get("missing_model_switch", 0))
    reference_metrics = pd.DataFrame()
    coverage = pd.DataFrame()
    strict_metrics = pd.DataFrame()
    if "reference_grid_missing_as_zero" in denominators:
        reference_metrics, coverage = build_reference_grid_metrics(
            reference_grid=reference_grid,
            sources=sources,
            missing_model_switch=missing_model_switch,
        )
    if "strict_common_timestamp_intersection" in denominators:
        strict_metrics = build_strict_intersection_metrics(
            reference_grid=reference_grid,
            sources=sources,
        )

    metric_sets = {
        "reference_grid": reference_metrics,
        "strict_intersection": strict_metrics,
    }
    for name, metrics in metric_sets.items():
        if metrics.empty:
            continue
        path = output_paths["global_metrics"] / f"{name}_core_switch_metrics.png"
        plot_metric_bars(
            metrics,
            path,
            title=f"Cross-task core switch metrics | {name}",
        )
        manifest.append(
            {
                "scope": "cross_task",
                "task_name": "all",
                "method_name": "all",
                "plot_type": f"{name}_core_switch_metrics",
                "path": relative_project_path(path),
            }
        )
        path = output_paths["precision_recall"] / f"{name}_precision_recall_coverage.png"
        plot_precision_recall_scatter(
            metrics,
            path,
            title=f"Cross-task precision/recall with coverage | {name}",
        )
        manifest.append(
            {
                "scope": "cross_task",
                "task_name": "all",
                "method_name": "all",
                "plot_type": f"{name}_precision_recall_coverage",
                "path": relative_project_path(path),
            }
        )

    if not coverage.empty:
        path = output_paths["coverage"] / "reference_grid_native_decision_coverage.png"
        plot_coverage(
            coverage,
            path,
            title="Cross-task native decision coverage on reference grid",
        )
        manifest.append(
            {
                "scope": "cross_task",
                "task_name": "all",
                "method_name": "all",
                "plot_type": "reference_grid_native_decision_coverage",
                "path": relative_project_path(path),
            }
        )

    if not reference_metrics.empty:
        path = output_paths["raw_vs_postprocessed"] / "reference_grid_raw_vs_postprocessed_counts.png"
        plot_raw_vs_postprocessed_counts(
            reference_metrics,
            path,
            title="Cross-task raw vs post-processed switch positives",
        )
        manifest.append(
            {
                "scope": "cross_task",
                "task_name": "all",
                "method_name": "all",
                "plot_type": "reference_grid_raw_vs_postprocessed_counts",
                "path": relative_project_path(path),
            }
        )

        event_ids = select_reference_events(
            reference_grid,
            max_events=int(config["plots"]["max_cross_task_events"]),
        )
        matrix = compute_reference_event_f1_matrix(
            reference_grid,
            sources,
            event_ids=event_ids,
            missing_model_switch=missing_model_switch,
        )
        matrix.to_csv(tables_dir / "cross_task_reference_grid_event_f1_matrix.csv")
        path = output_paths["event_heatmaps"] / "reference_grid_event_f1_heatmap.png"
        plot_event_metric_heatmap(
            matrix,
            path,
            title="Cross-task per-event F1 on common reference grid",
        )
        manifest.append(
            {
                "scope": "cross_task",
                "task_name": "all",
                "method_name": "all",
                "plot_type": "reference_grid_event_f1_heatmap",
                "path": relative_project_path(path),
            }
        )

        selected_timeline_sources = select_cross_task_timeline_sources(
            reference_metrics,
            sources,
            max_methods=int(config["plots"]["max_cross_task_timeline_methods"]),
        )
        selected_method_frames = {
            source.comparison_id: read_method_frame(source)
            for source in selected_timeline_sources
        }
        selected_timeline_rows = []
        for source in selected_timeline_sources:
            metric = reference_metrics.loc[
                reference_metrics["comparison_id"].astype(str).eq(source.comparison_id)
            ].iloc[0]
            selected_timeline_rows.append(
                {
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "comparison_id": source.comparison_id,
                    "f1_reference_grid": float(metric["f1"]),
                }
            )
        pd.DataFrame(selected_timeline_rows).to_csv(
            tables_dir / "cross_task_selected_event_timeline_methods.csv",
            index=False,
        )

        all_timeline_sources = sources
        all_method_frames = {
            source.comparison_id: read_method_frame(source)
            for source in all_timeline_sources
        }
        all_timeline_rows = []
        for source in all_timeline_sources:
            metric = reference_metrics.loc[
                reference_metrics["comparison_id"].astype(str).eq(source.comparison_id)
            ]
            all_timeline_rows.append(
                {
                    "task_name": source.task_name,
                    "method_name": source.method_name,
                    "comparison_id": source.comparison_id,
                    "f1_reference_grid": (
                        float(metric.iloc[0]["f1"]) if not metric.empty else np.nan
                    ),
                }
            )
        pd.DataFrame(all_timeline_rows).to_csv(
            tables_dir / "cross_task_all_event_timeline_methods.csv",
            index=False,
        )

        threshold = float(reference_grid["threshold"].dropna().iloc[0]) if "threshold" in reference_grid else 10.0
        for position, event_id in enumerate(event_ids, start=1):
            event_frame = reference_grid.loc[
                reference_grid["event_id"].astype(str).eq(str(event_id))
            ].copy()
            switches: dict[str, np.ndarray] = {
                "Perfect Switch": event_frame["perfect_switch"].to_numpy(dtype=np.int8)
            }
            raw_switches: dict[str, np.ndarray] = {}
            for source in selected_timeline_sources:
                method_frame = selected_method_frames[source.comparison_id]
                label = compact_method_label(source.__dict__)
                switches[label] = map_switch_to_times(
                    event_frame["Time"],
                    method_frame,
                    "model_switch",
                    fill_value=missing_model_switch,
                )
                raw_switches[label] = map_switch_to_times(
                    event_frame["Time"],
                    method_frame,
                    "model_switch_raw",
                    fill_value=missing_model_switch,
                )
            path = (
                output_paths["event_timelines/selected_methods"]
                / f"event_{position:03d}_{sanitize_id(event_id)}_cross_task_switches.png"
            )
            plot_switch_timeline_rows(
                event_frame,
                switches,
                path,
                threshold=threshold,
                title=f"Cross-task switch timeline | {event_id}",
                raw_switches=raw_switches,
            )
            manifest.append(
                {
                    "scope": "cross_task",
                    "task_name": "all",
                    "method_name": "selected_timeline_methods",
                    "plot_type": "cross_task_event_timeline_selected_methods",
                    "path": relative_project_path(path),
                }
            )

            if config["plots"].get("plot_all_cross_task_timeline_methods", False):
                all_switches: dict[str, np.ndarray] = {
                    "Perfect Switch": event_frame["perfect_switch"].to_numpy(dtype=np.int8)
                }
                all_raw_switches: dict[str, np.ndarray] = {}
                for source in all_timeline_sources:
                    method_frame = all_method_frames[source.comparison_id]
                    label = compact_method_label(source.__dict__)
                    all_switches[label] = map_switch_to_times(
                        event_frame["Time"],
                        method_frame,
                        "model_switch",
                        fill_value=missing_model_switch,
                    )
                    all_raw_switches[label] = map_switch_to_times(
                        event_frame["Time"],
                        method_frame,
                        "model_switch_raw",
                        fill_value=missing_model_switch,
                    )
                path = (
                    output_paths["event_timelines/all_methods"]
                    / f"event_{position:03d}_{sanitize_id(event_id)}_cross_task_switches_all_methods.png"
                )
                plot_switch_timeline_rows(
                    event_frame,
                    all_switches,
                    path,
                    threshold=threshold,
                    title=f"Cross-task switch timeline | all methods | {event_id}",
                    raw_switches=all_raw_switches,
                )
                manifest.append(
                    {
                        "scope": "cross_task",
                        "task_name": "all",
                        "method_name": "all_timeline_methods",
                        "plot_type": "cross_task_event_timeline_all_methods",
                        "path": relative_project_path(path),
                    }
                )

    return manifest


def write_report(path: Path, manifest: pd.DataFrame) -> None:
    """Write a compact plot navigation report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Switch Diagnostic Plot Navigation\n\n")
        stream.write(
            "Generated plots are organized by scope. Intra-task figures live under "
            "`results/comparisons/switch_diagnostics/<task>/<selection>/<method>/`. "
            "Cross-task figures live inside the corresponding "
            "`results/comparisons/cross_task_switch/<selection>/<comparison>/figures/` folder.\n\n"
        )
        stream.write("## Plot Counts\n\n")
        if manifest.empty:
            stream.write("No plots were generated.\n")
            return
        counts = (
            manifest.groupby(["scope", "task_name", "plot_type"], dropna=False)
            .size()
            .reset_index(name="num_plots")
            .sort_values(["scope", "task_name", "plot_type"])
        )
        stream.write("```text\n")
        stream.write(counts.to_string(index=False))
        stream.write("\n```\n")


def main() -> None:
    """Generate configured switch diagnostic plots."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("diagnostics_type") != "switch_diagnostics":
        raise ValueError("diagnostics_type must be switch_diagnostics.")

    cross_config_path = project_path(config["cross_task"]["config_path"])
    cross_config = load_yaml_config(cross_config_path)
    comparisons_index = pd.read_csv(project_path(cross_config["sources"]["comparison_index_path"]))
    sources = discover_switch_sources(
        comparisons_index,
        project_root=PROJECT_ROOT,
        include_sources=list(cross_config["sources"]["include"]),
    )
    if not sources:
        raise RuntimeError("No switch-evaluation sources matched the diagnostics config.")
    reference_grid = load_reference_grid(
        project_path(cross_config["reference"]["perfect_switch_timeseries_path"])
    )

    run_indexes = {
        "autoregressive": read_run_index(PROJECT_ROOT / "results" / "index" / "runs.csv"),
        "current_level_persistence": read_run_index(
            PROJECT_ROOT / "results" / "index" / "current_level_persistence_runs.csv"
        ),
        "survival_persistence": read_run_index(
            PROJECT_ROOT / "results" / "index" / "survival_persistence_runs.csv"
        ),
    }

    print("=== Switch Diagnostic Plot Generation ===")
    print(f"Config: {relative_project_path(config_path)}")
    print(f"Sources: {len(sources)}")
    print("Prints: generated plot counts and root paths.")
    print("Displays: no interactive figures.")
    print("Saves: intra-task diagnostic PNGs, cross-task diagnostic PNGs, manifests, metadata.")

    manifest_rows: list[dict[str, str]] = []
    manifest_rows.extend(
        generate_intra_task_plots(
            sources=sources,
            reference_grid=reference_grid,
            config=config,
            run_indexes=run_indexes,
        )
    )
    manifest_rows.extend(
        generate_cross_task_plots(
            sources=sources,
            reference_grid=reference_grid,
            cross_config=cross_config,
            config=config,
        )
    )

    manifest = pd.DataFrame(manifest_rows)
    switch_diagnostics_root = project_path(config["output"]["switch_diagnostics_root"])
    switch_diagnostics_root.mkdir(parents=True, exist_ok=True)
    manifest_path = switch_diagnostics_root / "plot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    report_path = switch_diagnostics_root / "reports" / "switch_diagnostic_plot_navigation.md"
    write_report(report_path, manifest)

    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "diagnostics_type": "switch_diagnostics",
        "config_path": relative_project_path(config_path),
        "cross_task_config_path": relative_project_path(cross_config_path),
        "config_fingerprint": config_fingerprint(config),
        "num_sources": len(sources),
        "num_plots": int(len(manifest)),
        "switch_diagnostics_root": relative_project_path(switch_diagnostics_root),
        "cross_task_output_dir": relative_project_path(cross_task_output_dir(cross_config)),
        "created_at": created_at,
    }
    save_yaml(switch_diagnostics_root / "metadata.yaml", metadata)

    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": "switch_diagnostics_externalHoldout_test_fc_uplink_fade",
            "method_id": "multiple",
            "reference_id": str(cross_config["reference"]["selection_id"]),
            "comparison_type": "switch_diagnostics",
            "selection_id": str(cross_config["comparison"]["normalized_selection_id"]),
            "results_path": relative_project_path(switch_diagnostics_root),
            "metrics_path": relative_project_path(switch_diagnostics_root / "plot_manifest.csv"),
            "figures_path": relative_project_path(switch_diagnostics_root),
            "status": "complete",
            "created_at": created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )

    print(f"Generated plots: {len(manifest):,}")
    print(f"Intra-task root: {relative_project_path(switch_diagnostics_root)}")
    print(f"Cross-task root: {relative_project_path(cross_task_output_dir(cross_config) / 'figures')}")
    print(f"Manifest: {relative_project_path(manifest_path)}")
    if not manifest.empty:
        counts = manifest.groupby(["scope", "task_name"], dropna=False).size()
        print(counts.to_string())


if __name__ == "__main__":
    main()
