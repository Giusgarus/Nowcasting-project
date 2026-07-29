"""Compare saved model-derived switches against the Perfect Switch oracle."""

# %%
# Path setup and imports
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.switching.comparison import (
    build_model_switch_timeseries,
    compute_model_vs_perfect_metrics,
)
from src.switching.metrics import build_switch_behavior_metrics_tables
from src.switching.plots import plot_switch_methods_for_event
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.results_paths import (
    COMPARISON_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_comparison_dir,
    get_results_index_dir,
    make_comparison_id,
    make_run_id,
    relative_project_path,
    resolve_perfect_switch_dir,
    resolve_run_dir,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/switch_comparison.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for alternate switch-comparison configs."""

    parser = argparse.ArgumentParser(
        description="Compare saved model-derived switches against Perfect Switch.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a switch-comparison YAML config.",
    )
    return parser.parse_args()


CONFIG_PATH = parse_args().config
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = PROJECT_ROOT / CONFIG_PATH

# %%
# Load configuration and Perfect Switch reference
config = load_yaml_config(CONFIG_PATH)
selection_id = config["data"]["selection_id"]
perfect_root = resolve_perfect_switch_dir(selection_id)
perfect_timeseries_path = perfect_root / "tables/perfect_switch_timeseries.parquet"
perfect_targets_path = perfect_root / "tables/perfect_switch_window_targets.parquet"
perfect_timeseries = pd.read_parquet(perfect_timeseries_path)
perfect_targets = pd.read_parquet(perfect_targets_path)
reference_id = f"perfect_switch_{selection_id}"
datasets_index_path = get_results_index_dir() / "datasets.csv"
datasets_index = pd.read_csv(datasets_index_path)
dataset_rows = datasets_index.loc[datasets_index["selection_id"].eq(selection_id)]
if "task_name" in dataset_rows.columns:
    task_rows = dataset_rows.loc[dataset_rows["task_name"].astype(str).eq("autoregressive")]
    if not task_rows.empty:
        dataset_rows = task_rows
if dataset_rows.empty:
    raise ValueError(f"Dataset index has no row for selection_id={selection_id}.")
dataset_path = PROJECT_ROOT / str(dataset_rows.iloc[0]["dataset_path"])
test_type = str(dataset_rows.iloc[0].get("dataset_selection_mode", "test"))
test_metadata_path = dataset_path / "test_metadata.parquet"
test_metadata = pd.read_parquet(test_metadata_path)
prediction_aggregation = config["conversion"].get(
    "prediction_aggregation",
    "latest_available",
)

print(
    "=== Analysis overview ===\n"
    "Prints: forecast-to-switch rule and model-vs-Perfect metrics.\n"
    "Displays: no figures.\n"
    "Saves: one independent grouped comparison folder per method against Perfect "
    "Switch, with switch-summary metrics, switch-behavior metrics, predictions, "
    "metadata, and event plots.\n"
    "Note: uses saved test predictions only; no model is trained or reloaded.\n"
)
if prediction_aggregation not in {"latest_available", "horizon_threshold_count"}:
    raise ValueError(
        "Only prediction_aggregation=latest_available or "
        "horizon_threshold_count is supported."
    )
required_points_above_threshold = config["conversion"].get(
    "required_points_above_threshold"
)
if (
    prediction_aggregation == "horizon_threshold_count"
    and required_points_above_threshold is None
):
    raise ValueError(
        "horizon_threshold_count requires conversion.required_points_above_threshold."
    )

# %%
# Build one independent model-vs-Perfect comparison per method
reports = []
for method in config["methods"]:
    run_id = make_run_id(
        method["model_family"],
        method["architecture"],
        method["variant"],
        selection_id,
    )
    comparison_id = make_comparison_id(run_id)
    run_dir = resolve_run_dir(run_id)
    predictions_path = run_dir / "predictions/test_predictions.parquet"
    output_dir = get_comparison_dir(comparison_id)
    output_paths = ensure_results_subdirs(
        output_dir,
        ("metrics", "predictions", "figures", "tables"),
    )
    comparison_path = output_paths["predictions"] / "model_vs_perfect_switch.parquet"
    summary_path = output_paths["metrics"] / "switch_metrics_summary.csv"
    event_metrics_path = output_paths["metrics"] / "switch_metrics_by_event.csv"
    global_behavior_path = output_paths["metrics"] / "global_switch_metrics.csv"
    event_behavior_path = output_paths["metrics"] / "event_switch_metrics.csv"
    metadata_path = output_dir / "metadata.yaml"
    figures_dir = output_paths["figures"] / "event_switch_plots"

    required_outputs = (
        comparison_path,
        summary_path,
        event_metrics_path,
        global_behavior_path,
        event_behavior_path,
        metadata_path,
    )
    if not config["output"]["overwrite"]:
        existing = [path for path in required_outputs if path.exists()]
        if len(existing) == len(required_outputs):
            reports.append(pd.read_csv(summary_path))
            print(f"Skipped existing complete comparison: {comparison_id}")
            continue
        if existing:
            raise FileExistsError(
                f"Comparison outputs are incomplete under {output_dir}. "
                "Set output.overwrite=true to rebuild this method."
            )

    predictions = pd.read_parquet(predictions_path)
    comparison = build_model_switch_timeseries(
        predictions,
        perfect_targets,
        method_name=method["name"],
        threshold=float(config["conversion"]["threshold"]),
        condition=config["conversion"]["condition"],
        switch_time=int(config["conversion"]["switch_time"]),
        apply_min_island_length=bool(
            config["conversion"]["apply_min_island_length"]
        ),
        prediction_aggregation=prediction_aggregation,
        required_points_above_threshold=(
            int(required_points_above_threshold)
            if required_points_above_threshold is not None
            else None
        ),
        window_metadata=test_metadata,
        perfect_timeseries=perfect_timeseries,
    )
    summary = compute_model_vs_perfect_metrics(comparison, by_event=False)
    event_metrics = compute_model_vs_perfect_metrics(comparison, by_event=True)
    method_mode = method.get(
        "mode",
        "zero_shot" if method["model_family"] == "chronos" else "trained",
    )
    global_behavior, event_behavior = build_switch_behavior_metrics_tables(
        comparison,
        method_id=run_id,
        method_metadata={
            "method_name": method["name"],
            "model_family": method["model_family"],
            "architecture": method["architecture"],
            "variant": method["variant"],
            "mode": method_mode,
        },
        selection_id=selection_id,
        test_type=test_type,
        threshold=float(config["conversion"]["threshold"]),
        condition=config["conversion"]["condition"],
        switch_time=int(config["conversion"]["switch_time"]),
        perfect_timeseries=perfect_timeseries,
    )
    comparison.to_parquet(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    event_metrics.to_csv(event_metrics_path, index=False)
    global_behavior.to_csv(global_behavior_path, index=False)
    event_behavior.to_csv(event_behavior_path, index=False)

    saved_figures = []
    event_order = (
        perfect_timeseries.groupby("event_id", sort=False)["event_timestamp"]
        .first()
        .sort_values()
        .index[: int(config["plots"]["max_events"])]
    )
    for event_id in event_order:
        perfect_event = perfect_timeseries.loc[
            perfect_timeseries["event_id"].eq(event_id)
        ].sort_values("Time", kind="stable")
        model_event = comparison.loc[
            comparison["event_id"].eq(event_id),
            ["Time", "model_switch_min_time"],
        ]
        lookup = model_event.set_index("Time")["model_switch_min_time"]
        methods: dict[str, np.ndarray] = {
            "Perfect Switch": perfect_event["perfect_switch"].to_numpy(dtype=float),
            method["name"]: perfect_event["Time"].map(lookup).to_numpy(dtype=float),
        }
        figure_path = figures_dir / f"{event_id}_switch_comparison.png"
        plot_switch_methods_for_event(
            perfect_event,
            methods,
            figure_path,
            threshold=float(config["conversion"]["threshold"]),
            event_title=(
                f"{event_id} | {perfect_event['dataset_name'].iloc[0]} | "
                f"Perfect vs {method['name']}"
            ),
        )
        saved_figures.append(relative_project_path(figure_path))

    created_at = datetime.now(timezone.utc).isoformat()
    fingerprint_config = {
        "data": config["data"],
        "method": method,
        "conversion": config["conversion"],
        "plots": config["plots"],
        "output": {**config["output"], "overwrite": False},
    }
    metadata = {
        "comparison_id": comparison_id,
        "comparison_type": "switch_eval",
        "method_id": run_id,
        "method": method,
        "reference_id": reference_id,
        "selection_id": selection_id,
        "split": "test",
        "prediction_aggregation": prediction_aggregation,
        "model_switch_rule": (
            "At every input_end_time t, count how many of the saved horizon "
            "forecasts y_hat(t+1)...y_hat(t+h) satisfy the configured threshold "
            "condition. model_switch_raw(t)=1 if that count is at least "
            "required_points_above_threshold. The configured minimum-island "
            "post-processing is then applied to the decision-time switch series."
            if prediction_aggregation == "horizon_threshold_count"
            else (
                "For every event target timestamp, select the saved forecast with "
                "the smallest horizon_step, then apply the pointwise threshold and "
                "minimum-island post-processing."
            )
        ),
        "threshold": float(config["conversion"]["threshold"]),
        "condition": config["conversion"]["condition"],
        "required_points_above_threshold": (
            int(required_points_above_threshold)
            if required_points_above_threshold is not None
            else None
        ),
        "switch_time": int(config["conversion"]["switch_time"]),
        "input_files": {
            "perfect_switch_timeseries": relative_project_path(perfect_timeseries_path),
            "perfect_switch_window_targets": relative_project_path(perfect_targets_path),
            "test_metadata": relative_project_path(test_metadata_path),
            "predictions": relative_project_path(predictions_path),
        },
        "output_files": {
            "comparison_timeseries": relative_project_path(comparison_path),
            "switch_metrics_summary": relative_project_path(summary_path),
            "switch_metrics_by_event": relative_project_path(event_metrics_path),
            "global_switch_metrics": relative_project_path(global_behavior_path),
            "event_switch_metrics": relative_project_path(event_behavior_path),
            "figures": saved_figures,
        },
        "num_events": int(comparison["event_id"].nunique()),
        "num_figures": len(saved_figures),
        "config_path": relative_project_path(CONFIG_PATH),
        "config_fingerprint": config_fingerprint(fingerprint_config),
        "created_at": created_at,
    }
    save_yaml(metadata_path, metadata)
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
    reports.append(summary)
    print(f"Saved comparison: {comparison_id}")

print("=== Compact final report ===")
print(f"Selection ID: {selection_id}")
if prediction_aggregation == "horizon_threshold_count":
    print(
        "Prediction aggregation: decision-time horizon threshold count "
        f"(required points={required_points_above_threshold})"
    )
    print(
        f"Model switch: at input_end_time, at least "
        f"{required_points_above_threshold} horizon predictions satisfy "
        f"{config['conversion']['condition']} {config['conversion']['threshold']}; "
        f"then minimum island length {config['conversion']['switch_time']}"
    )
else:
    print("Prediction aggregation: latest available forecast (smallest horizon step)")
    print(
        f"Model switch: y_pred_raw {config['conversion']['condition']} "
        f"{config['conversion']['threshold']} then minimum island length "
        f"{config['conversion']['switch_time']}"
    )
print(pd.concat(reports, ignore_index=True).to_string(index=False))
