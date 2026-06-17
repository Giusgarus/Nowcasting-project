"""Build compact cross-model forecast and switch comparison tables."""

# %%
# Path setup and imports
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.results_paths import (
    COMPARISON_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_comparison_dir,
    get_results_index_dir,
    get_run_dir,
    make_comparison_id,
    make_run_id,
    relative_project_path,
    upsert_index_row,
)

CONFIG_PATH = PROJECT_ROOT / "configs/switch_comparison.yaml"
RUN_INDEX_PATH = PROJECT_ROOT / "results/index/runs.csv"
COMPARISON_INDEX_PATH = PROJECT_ROOT / "results/index/comparisons.csv"

FORECAST_PARAMETER_COLUMNS = [
    "best_epoch",
    "final_training_epochs",
    "final_retrained_on_full_development",
    "hidden_size",
    "num_layers",
    "learning_rate",
    "weight_decay",
    "teacher_forcing_ratio",
    "patch_len",
    "stride",
    "d_model",
    "n_heads",
    "dropout",
]

SWITCH_METRIC_COLUMNS = [
    "num_points",
    "num_perfect_positive",
    "num_model_positive_raw",
    "num_model_positive",
    "points_changed_by_min_island",
    "active_duration_seconds",
    "precision",
    "recall",
    "f1",
    "intersection_over_union",
    "balanced_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
]


def _first_row(path: Path) -> dict[str, Any]:
    """Read the first row of a CSV file as a dictionary."""

    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _method_name_lookup(config: dict[str, Any], selection_id: str) -> dict[str, str]:
    """Map run IDs to readable method names from the switch-comparison config."""

    lookup = {}
    for method in config.get("methods", []):
        run_id = make_run_id(
            method["model_family"],
            method["architecture"],
            method["variant"],
            selection_id,
        )
        lookup[run_id] = method["name"]
    return lookup


def _run_label(run: pd.Series, method_names: dict[str, str]) -> str:
    """Return a readable model label for a run."""

    run_id = str(run["run_id"])
    if run_id in method_names:
        return method_names[run_id]
    family = str(run["model_family"]).upper()
    architecture = str(run["architecture"])
    variant = str(run["variant"]).replace("_", " ").title()
    return f"{family} {architecture} {variant}"


def _comparison_summary_path(run_id: str) -> Path:
    comparison_id = make_comparison_id(run_id)
    return (
        get_comparison_dir(comparison_id)
        / "metrics"
        / "switch_metrics_summary.csv"
    )


def _build_forecast_rows(
    runs: pd.DataFrame,
    method_names: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build validation and test forecast rows from run-level metric files."""

    validation_rows = []
    test_rows = []
    for _, run in runs.sort_values(
        ["model_family", "architecture", "variant"],
        kind="stable",
    ).iterrows():
        run_id = str(run["run_id"])
        metrics = _first_row(get_run_dir(run_id) / "metrics" / "metrics_summary.csv")
        label = _run_label(run, method_names)
        base = {
            "method": label,
            "run_id": run_id,
            "model_family": run["model_family"],
            "architecture": run["architecture"],
            "variant": run["variant"],
            "selection_id": run["selection_id"],
            "status": run["status"],
        }

        validation_row = {
            **base,
            "validation_available": pd.notna(metrics.get("validation_mae_raw"))
            and pd.notna(metrics.get("validation_rmse_raw")),
            "validation_mae": metrics.get("validation_mae_raw"),
            "validation_rmse": metrics.get("validation_rmse_raw"),
            "num_train_windows": metrics.get("num_train_windows"),
            "num_val_windows": metrics.get("num_val_windows"),
        }
        for column in FORECAST_PARAMETER_COLUMNS:
            validation_row[column] = metrics.get(column)
        validation_rows.append(validation_row)

        test_row = {
            **base,
            "test_mae": metrics.get("test_mae"),
            "test_rmse": metrics.get("test_rmse"),
            "num_test_windows": metrics.get("num_test_windows"),
        }
        for column in FORECAST_PARAMETER_COLUMNS:
            test_row[column] = metrics.get(column)
        test_rows.append(test_row)

    return validation_rows, test_rows


def _merge_switch_rows(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach model-vs-Perfect switch metrics when available."""

    enriched_rows = []
    for row in test_rows:
        switch_metrics = _first_row(_comparison_summary_path(str(row["run_id"])))
        enriched = {
            **row,
            "switch_metrics_available": bool(switch_metrics),
        }
        for column in SWITCH_METRIC_COLUMNS:
            enriched[column] = switch_metrics.get(column)
        enriched_rows.append(enriched)
    return enriched_rows


def _ordered_validation_columns(frame: pd.DataFrame) -> list[str]:
    front = [
        "method",
        "model_family",
        "architecture",
        "variant",
        "validation_available",
        "validation_mae",
        "validation_rmse",
        "best_epoch",
        "final_training_epochs",
        "final_retrained_on_full_development",
        "num_train_windows",
        "num_val_windows",
        "run_id",
        "selection_id",
        "status",
    ]
    return front + [column for column in frame.columns if column not in front]


def _ordered_test_columns(frame: pd.DataFrame) -> list[str]:
    front = [
        "method",
        "model_family",
        "architecture",
        "variant",
        "test_mae",
        "test_rmse",
        "switch_metrics_available",
        "precision",
        "recall",
        "f1",
        "intersection_over_union",
        "balanced_accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "num_perfect_positive",
        "num_model_positive",
        "active_duration_seconds",
        "num_test_windows",
        "run_id",
        "selection_id",
        "status",
    ]
    return front + [column for column in frame.columns if column not in front]


# %%
# Load run index and build summary tables
config = load_yaml_config(CONFIG_PATH)
selection_id = config["data"]["selection_id"]
comparison_id = make_comparison_id(selection_id, comparison_type="model_result_summary")
output_dir = get_comparison_dir(comparison_id)
output_paths = ensure_results_subdirs(output_dir, ("tables", "metrics"))
tables_dir = output_paths["tables"]

print(
    "=== Analysis overview ===\n"
    "Prints: compact validation and test model comparison tables.\n"
    "Displays: no figures.\n"
    "Saves: one validation forecast table and one test forecast+switch table.\n"
    "Note: reads existing run metrics and existing model-vs-Perfect switch "
    "comparisons; it does not train or evaluate models.\n"
)

if not RUN_INDEX_PATH.exists():
    raise FileNotFoundError(f"Run index not found: {RUN_INDEX_PATH}")

runs = pd.read_csv(RUN_INDEX_PATH)
runs = runs.loc[
    runs["selection_id"].astype(str).eq(selection_id)
    & runs["status"].astype(str).eq("complete")
].copy()
if runs.empty:
    raise ValueError(f"No complete indexed runs found for selection_id={selection_id}")

method_names = _method_name_lookup(config, selection_id)
validation_rows, test_rows = _build_forecast_rows(runs, method_names)
test_rows = _merge_switch_rows(test_rows)

validation_table = pd.DataFrame(validation_rows)
test_table = pd.DataFrame(test_rows)
validation_table = validation_table[_ordered_validation_columns(validation_table)]
test_table = test_table[_ordered_test_columns(test_table)]

validation_table = validation_table.sort_values(
    ["validation_available", "validation_rmse", "method"],
    ascending=[False, True, True],
    kind="stable",
)
test_table = test_table.sort_values(
    ["test_rmse", "method"],
    ascending=[True, True],
    kind="stable",
)

validation_path = tables_dir / "validation_forecast_comparison.csv"
test_path = tables_dir / "test_forecast_switch_comparison.csv"
validation_table.to_csv(validation_path, index=False)
test_table.to_csv(test_path, index=False)

created_at = datetime.now(timezone.utc).isoformat()
metadata_path = output_dir / "metadata.yaml"
metadata = {
    "comparison_id": comparison_id,
    "comparison_type": "model_result_summary",
    "selection_id": selection_id,
    "num_runs": int(len(runs)),
    "num_runs_with_validation_metrics": int(
        validation_table["validation_available"].fillna(False).sum()
    ),
    "num_runs_with_switch_metrics": int(
        test_table["switch_metrics_available"].fillna(False).sum()
    ),
    "input_files": {
        "run_index": relative_project_path(RUN_INDEX_PATH),
        "comparison_index": relative_project_path(COMPARISON_INDEX_PATH),
        "switch_comparison_config": relative_project_path(CONFIG_PATH),
    },
    "output_files": {
        "validation_forecast_comparison": relative_project_path(validation_path),
        "test_forecast_switch_comparison": relative_project_path(test_path),
    },
    "config_fingerprint": config_fingerprint(config),
    "created_at": created_at,
}
save_yaml(metadata_path, metadata)

upsert_index_row(
    get_results_index_dir() / "comparisons.csv",
    {
        "comparison_id": comparison_id,
        "method_id": "all_indexed_runs",
        "reference_id": f"perfect_switch_{selection_id}",
        "comparison_type": "model_result_summary",
        "selection_id": selection_id,
        "results_path": relative_project_path(output_dir),
        "metrics_path": relative_project_path(tables_dir),
        "figures_path": "",
        "status": "complete",
        "created_at": created_at,
    },
    id_column="comparison_id",
    columns=COMPARISON_INDEX_COLUMNS,
)

print("=== Validation forecast comparison ===")
print(
    validation_table[
        [
            "method",
            "validation_available",
            "validation_mae",
            "validation_rmse",
            "best_epoch",
        ]
    ].to_string(index=False)
)
print("\n=== Test forecast and switch comparison ===")
print(
    test_table[
        [
            "method",
            "test_mae",
            "test_rmse",
            "switch_metrics_available",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
        ]
    ].to_string(index=False)
)
print("\nSaved:")
print(f"- {relative_project_path(validation_path)}")
print(f"- {relative_project_path(test_path)}")
