"""Compare long-fade probability-derived switches against Perfect Switch."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.switching.comparison import compute_model_vs_perfect_metrics  # noqa: E402
from src.switching.metrics import build_switch_behavior_metrics_tables  # noqa: E402
from src.switching.plots import (  # noqa: E402
    build_centered_event_display_frame,
    map_switch_to_display_frame,
    plot_switch_methods_for_event,
)
from src.tasks.long_fade_detection.evaluation.switch_from_probability import (  # noqa: E402
    build_probability_switch_timeseries,
)
from src.tasks.long_fade_detection.utils.paths import (  # noqa: E402
    TASK_NAME,
    run_dir,
    switch_comparison_dir,
    switch_summary_dir,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    COMPARISON_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_perfect_switch_dir,
    get_results_index_dir,
    make_comparison_id,
    relative_project_path,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/long_fade_detection/switch_comparison.yaml"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--refresh-plots-only",
        action="store_true",
        help=(
            "Regenerate event plots from saved switch-comparison parquet files "
            "without recomputing metrics or checking model/dataset fingerprints."
        ),
    )
    return parser.parse_args()


def load_display_signal_source(config: dict) -> tuple[pd.DataFrame | None, dict]:
    """Load the optional full-signal source used only for event plot display."""

    display_config = dict(config.get("plots", {}).get("display_window", {}))
    if not bool(display_config.get("enabled", False)):
        return None, display_config
    source_path = project_path(display_config["signal_source_path"])
    return pd.read_parquet(source_path), display_config


def plot_event_switch_comparisons(
    *,
    perfect_timeseries: pd.DataFrame,
    comparison: pd.DataFrame,
    figures_dir: Path,
    method_name: str,
    threshold: float,
    max_events: int,
    title_suffix: str,
    display_signal_source: pd.DataFrame | None,
    display_config: dict,
) -> list[str]:
    """Save event switch plots, optionally using a wider display-only window."""

    saved_figures = []
    event_order = (
        perfect_timeseries.groupby("event_id", sort=False)["event_timestamp"]
        .first()
        .sort_values()
        .index[:max_events]
    )
    for event_id in event_order:
        perfect_event = perfect_timeseries.loc[
            perfect_timeseries["event_id"].eq(event_id)
        ].sort_values("Time", kind="stable")
        model_event = comparison.loc[
            comparison["event_id"].eq(event_id),
            ["Time", "model_switch_min_time"],
        ].sort_values("Time", kind="stable")

        if display_signal_source is not None:
            plot_frame = build_centered_event_display_frame(
                display_signal_source,
                perfect_event,
                before_minutes=float(display_config.get("before_minutes", 90)),
                after_minutes=float(display_config.get("after_minutes", 90)),
                signal_column=str(display_config.get("signal_column", "Signal")),
            )
            fill_value = int(display_config.get("fill_missing_switch_with", 0))
            switch_methods = {
                "Perfect Switch": map_switch_to_display_frame(
                    plot_frame,
                    perfect_event,
                    "perfect_switch",
                    fill_value=fill_value,
                ),
                method_name: map_switch_to_display_frame(
                    plot_frame,
                    model_event,
                    "model_switch_min_time",
                    fill_value=fill_value,
                ),
            }
        else:
            plot_frame = perfect_event
            model_lookup = model_event.set_index("Time")["model_switch_min_time"]
            switch_methods = {
                "Perfect Switch": perfect_event["perfect_switch"].to_numpy(dtype=float),
                method_name: perfect_event["Time"].map(model_lookup).to_numpy(dtype=float),
            }

        figure_path = figures_dir / f"{event_id}_switch_comparison.png"
        plot_switch_methods_for_event(
            plot_frame,
            switch_methods,
            figure_path,
            threshold=threshold,
            event_title=f"{event_id} | Perfect vs {method_name} | {title_suffix}",
        )
        saved_figures.append(relative_project_path(figure_path))
    return saved_figures


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")

    selection_id = str(config["data"]["selection_id"])
    dataset_path = project_path(config["data"]["dataset_path"])
    dataset_metadata = load_yaml_config(dataset_path / "dataset_metadata.yaml")
    test_metadata_path = dataset_path / "test_metadata.parquet"
    test_metadata = pd.read_parquet(test_metadata_path)
    perfect_root = get_perfect_switch_dir(selection_id, task_name=TASK_NAME)
    perfect_timeseries_path = perfect_root / "tables/perfect_switch_timeseries.parquet"
    perfect_timeseries = pd.read_parquet(perfect_timeseries_path)
    perfect_metadata = load_yaml_config(perfect_root / "metadata.yaml")
    reference_id = str(perfect_metadata["reference_id"])

    conversion = config["conversion"]
    probability_threshold = float(conversion["probability_threshold"])
    signal_threshold = float(conversion["signal_threshold"])
    signal_condition = str(conversion["signal_condition"])
    require_signal_gate = bool(
        conversion.get("require_current_signal_above_threshold", True)
    )
    switch_time = int(conversion["switch_time_samples"])
    apply_min_island = bool(conversion["apply_min_island_length"])
    overwrite = bool(config["output"].get("overwrite", False))
    display_signal_source, display_config = load_display_signal_source(config)

    print("=== Long-Fade Detection Switch Comparison ===")
    print(f"Selection: {selection_id}")
    print(f"Probability decision threshold: {probability_threshold:g}")
    print(f"Signal threshold: {signal_threshold:g}")
    print(f"Require current signal above threshold: {require_signal_gate}")
    print(f"Post-processing switch time: {switch_time} samples")
    print("Uses saved test predictions only; no model is trained or loaded.")

    if args.refresh_plots_only:
        print("Refresh-plots-only mode: metrics and predictions are not modified.")
        for method in config["methods"]:
            run_id = str(method["run_id"])
            comparison_id = make_comparison_id(run_id)
            output_dir = switch_comparison_dir(
                selection_id=selection_id,
                comparison_id=comparison_id,
            )
            comparison_path = output_dir / "predictions/model_vs_perfect_switch.parquet"
            if not comparison_path.exists():
                raise FileNotFoundError(
                    f"Saved switch comparison not found: {comparison_path}"
                )
            comparison = pd.read_parquet(comparison_path)
            figures_dir = output_dir / "figures/event_switch_plots"
            if overwrite and figures_dir.exists():
                shutil.rmtree(figures_dir)
            saved_figures = plot_event_switch_comparisons(
                perfect_timeseries=perfect_timeseries,
                comparison=comparison,
                figures_dir=figures_dir,
                method_name=str(method["name"]),
                threshold=signal_threshold,
                max_events=int(config["plots"]["max_events"]),
                title_suffix=f"probability threshold={probability_threshold:g}",
                display_signal_source=display_signal_source,
                display_config=display_config,
            )
            print(f"{method['name']}: refreshed {len(saved_figures)} plots")
        return

    summary_rows = []
    behavior_rows = []
    perfect_behavior_row = None
    for method in config["methods"]:
        run_id = str(method["run_id"])
        comparison_id = make_comparison_id(run_id)
        method_run_dir = run_dir(run_id)
        run_metadata = load_yaml_config(method_run_dir / "metadata.yaml")
        run_dataset_metadata = run_metadata.get("dataset_metadata", {})
        current_dataset_fingerprint = dataset_metadata.get("config_fingerprint")
        refresh_metadata = run_metadata.get("prediction_refresh", {})
        trained_on_current_dataset = (
            run_dataset_metadata.get("config_fingerprint")
            == current_dataset_fingerprint
        )
        refreshed_on_current_dataset = (
            refresh_metadata.get("dataset_config_fingerprint")
            == current_dataset_fingerprint
        )
        if not trained_on_current_dataset and not refreshed_on_current_dataset:
            raise RuntimeError(
                f"Run {run_id} has neither training nor refreshed predictions "
                "for the current dataset build. Rerun the grid search or refresh "
                "predictions before computing switch comparisons."
            )
        predictions_path = method_run_dir / "predictions/test_predictions.parquet"
        predictions = pd.read_parquet(predictions_path)

        valid_window_ids = set(test_metadata["window_id"].astype(str))
        prediction_window_ids = set(predictions["window_id"].astype(str))
        missing_prediction_ids = valid_window_ids - prediction_window_ids
        if missing_prediction_ids:
            examples = sorted(missing_prediction_ids)[:5]
            raise ValueError(
                "Saved predictions do not cover the test set. "
                f"Examples: {examples}"
            )
        prediction_is_valid = predictions["window_id"].astype(str).isin(
            valid_window_ids
        )
        num_quality_excluded_predictions = int((~prediction_is_valid).sum())
        predictions = predictions.loc[prediction_is_valid].copy()

        comparison = build_probability_switch_timeseries(
            predictions,
            test_metadata,
            perfect_timeseries,
            method_name=str(method["name"]),
            probability_threshold=probability_threshold,
            signal_threshold=signal_threshold,
            signal_condition=signal_condition,
            switch_time=switch_time,
            apply_min_island_length=apply_min_island,
            require_current_signal_above_threshold=require_signal_gate,
        )

        output_dir = switch_comparison_dir(
            selection_id=selection_id,
            comparison_id=comparison_id,
        )
        output_paths = ensure_results_subdirs(
            output_dir,
            ("metrics", "predictions", "figures"),
        )
        comparison_path = (
            output_paths["predictions"] / "model_vs_perfect_switch.parquet"
        )
        if comparison_path.exists() and not overwrite:
            raise FileExistsError(f"Comparison already exists: {comparison_path}")

        summary = compute_model_vs_perfect_metrics(comparison, by_event=False)
        event_metrics = compute_model_vs_perfect_metrics(comparison, by_event=True)
        method_family = str(method.get("model_family", method["model_id"]))
        method_architecture = str(method.get("architecture", method["model_id"]))
        method_variant = str(method.get("variant", "probability"))
        method_mode = str(method.get("mode", "trained"))
        global_behavior, event_behavior = build_switch_behavior_metrics_tables(
            comparison,
            method_id=run_id,
            method_metadata={
                "method_name": str(method["name"]),
                "model_family": method_family,
                "architecture": method_architecture,
                "variant": method_variant,
                "mode": method_mode,
            },
            selection_id=selection_id,
            test_type="external_holdout",
            threshold=signal_threshold,
            condition=signal_condition,
            switch_time=switch_time,
            perfect_timeseries=perfect_timeseries,
        )
        comparison.to_parquet(comparison_path, index=False)
        summary.to_csv(
            output_paths["metrics"] / "switch_metrics_summary.csv",
            index=False,
        )
        event_metrics.to_csv(
            output_paths["metrics"] / "switch_metrics_by_event.csv",
            index=False,
        )
        global_behavior.to_csv(
            output_paths["metrics"] / "global_switch_metrics.csv",
            index=False,
        )
        event_behavior.to_csv(
            output_paths["metrics"] / "event_switch_metrics.csv",
            index=False,
        )

        figures_dir = output_paths["figures"] / "event_switch_plots"
        if overwrite and figures_dir.exists():
            shutil.rmtree(figures_dir)
        saved_figures = plot_event_switch_comparisons(
            perfect_timeseries=perfect_timeseries,
            comparison=comparison,
            figures_dir=figures_dir,
            method_name=str(method["name"]),
            threshold=signal_threshold,
            max_events=int(config["plots"]["max_events"]),
            title_suffix=f"probability threshold={probability_threshold:g}",
            display_signal_source=display_signal_source,
            display_config=display_config,
        )

        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "comparison_id": comparison_id,
            "comparison_type": "switch_eval",
            "task_name": TASK_NAME,
            "method_id": run_id,
            "method": method,
            "reference_id": reference_id,
            "selection_id": selection_id,
            "split": "test",
            "probability_threshold": probability_threshold,
            "signal_threshold": signal_threshold,
            "signal_condition": signal_condition,
            "require_current_signal_above_threshold": require_signal_gate,
            "switch_time_samples": switch_time,
            "apply_min_island_length": apply_min_island,
            "model_switch_rule": (
                "model_switch_raw(t)=1 when predicted P(long_fade) is at least "
                "probability_threshold and the current true Signal is above the "
                "signal threshold when configured; then apply the shared "
                "min-island and stateful signal-above-threshold hold per event."
            ),
            "input_files": {
                "perfect_switch_timeseries": relative_project_path(
                    perfect_timeseries_path
                ),
                "test_metadata": relative_project_path(test_metadata_path),
                "predictions": relative_project_path(predictions_path),
            },
            "output_files": {
                "comparison_timeseries": relative_project_path(comparison_path),
                "figures": saved_figures,
            },
            "num_events": int(comparison["event_id"].nunique()),
            "num_decisions": int(len(comparison)),
            "num_quality_excluded_predictions": num_quality_excluded_predictions,
            "config_path": relative_project_path(config_path),
            "config_fingerprint": config_fingerprint(config),
            "created_at": created_at,
        }
        save_yaml(output_dir / "metadata.yaml", metadata)
        upsert_index_row(
            get_results_index_dir() / "comparisons.csv",
            {
                "comparison_id": comparison_id,
                "method_id": run_id,
                "reference_id": reference_id,
                "comparison_type": "switch_eval",
                "selection_id": selection_id,
                "results_path": relative_project_path(output_dir),
                "metrics_path": relative_project_path(output_paths["metrics"]),
                "figures_path": relative_project_path(output_paths["figures"]),
                "status": "complete",
                "created_at": created_at,
            },
            id_column="comparison_id",
            columns=COMPARISON_INDEX_COLUMNS,
        )

        summary.insert(0, "run_id", run_id)
        summary.insert(1, "model_id", str(method["model_id"]))
        summary_rows.append(summary)
        behavior_rows.append(
            global_behavior.loc[global_behavior["method_id"].eq(run_id)]
        )
        if perfect_behavior_row is None:
            perfect_behavior_row = global_behavior.loc[
                global_behavior["method_id"].eq("perfect_switch")
            ]
        print(
            f"{method['name']}: F1={summary['f1'].iloc[0]:.4f}, "
            f"precision={summary['precision'].iloc[0]:.4f}, "
            f"recall={summary['recall'].iloc[0]:.4f}, "
            f"quality-excluded predictions={num_quality_excluded_predictions:,}"
        )

    summary_id = str(config["output"]["summary_id"])
    summary_dir = switch_summary_dir(
        selection_id=selection_id,
        summary_id=summary_id,
    )
    summary_paths = ensure_results_subdirs(summary_dir, ("tables",))
    comparison_summary = pd.concat(summary_rows, ignore_index=True)
    behavior_frames = [*behavior_rows]
    if perfect_behavior_row is not None:
        behavior_frames.append(perfect_behavior_row)
    behavior_summary = pd.concat(behavior_frames, ignore_index=True)
    comparison_summary.to_csv(
        summary_paths["tables"] / "switch_metrics_comparison.csv",
        index=False,
    )
    behavior_summary.to_csv(
        summary_paths["tables"] / "global_switch_behavior_comparison.csv",
        index=False,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    save_yaml(
        summary_dir / "metadata.yaml",
        {
            "comparison_id": summary_id,
            "comparison_type": "model_summary",
            "task_name": TASK_NAME,
            "selection_id": selection_id,
            "reference_id": reference_id,
            "probability_threshold": probability_threshold,
            "require_current_signal_above_threshold": require_signal_gate,
            "method_ids": [str(method["run_id"]) for method in config["methods"]],
            "created_at": created_at,
        },
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": summary_id,
            "method_id": "multiple",
            "reference_id": reference_id,
            "comparison_type": "model_summary",
            "selection_id": selection_id,
            "results_path": relative_project_path(summary_dir),
            "metrics_path": relative_project_path(summary_paths["tables"]),
            "figures_path": "",
            "status": "complete",
            "created_at": created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )
    print(f"Summary: {relative_project_path(summary_dir)}")


if __name__ == "__main__":
    main()
