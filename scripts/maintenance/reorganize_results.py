"""Conservatively mirror known artifacts into the stable shallow result structure."""

import filecmp
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config, save_yaml
from src.utils.results_paths import (
    COMPARISON_INDEX_COLUMNS,
    DATASET_INDEX_COLUMNS,
    RUN_INDEX_COLUMNS,
    SWITCH_REFERENCE_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_comparison_dir,
    get_data_preparation_dir,
    get_model_dir,
    get_perfect_switch_dir,
    get_results_index_dir,
    get_run_dir,
    make_run_id,
    relative_project_path,
    selection_id_from_metadata,
    upsert_index_row,
)

def copy_file(source: Path, destination: Path, report: dict[str, list[str]]) -> None:
    """Copy a file only when the destination is absent or byte-identical."""

    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if filecmp.cmp(source, destination, shallow=False):
            report["already_mirrored"].append(relative_project_path(destination))
        else:
            report["conflicts"].append(
                f"{relative_project_path(source)} -> {relative_project_path(destination)}"
            )
        return
    shutil.copy2(source, destination)
    report["copied"].append(
        f"{relative_project_path(source)} -> {relative_project_path(destination)}"
    )


def copy_tree(source: Path, destination: Path, report: dict[str, list[str]]) -> None:
    """Conservatively copy all files below a known directory."""

    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy_file(path, destination / path.relative_to(source), report)


def ensure_empty_index(path: Path, columns: list[str]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def write_yaml_if_absent(
    path: Path,
    content: dict,
    report: dict[str, list[str]],
) -> None:
    """Write migration metadata without replacing an existing destination file."""

    if path.exists():
        category = (
            "already_mirrored"
            if load_yaml_config(path) == content
            else "conflicts"
        )
        report[category].append(relative_project_path(path))
        return
    save_yaml(path, content)
    report["copied"].append(f"generated -> {relative_project_path(path)}")


def migrate_dataset_selection(
    dataset_folder: Path,
    report: dict[str, list[str]],
) -> tuple[str, dict]:
    metadata = load_yaml_config(dataset_folder / "dataset_metadata.yaml")
    selection_id = selection_id_from_metadata(metadata)
    destination = get_data_preparation_dir(selection_id)
    destination.mkdir(parents=True, exist_ok=True)

    old_summary = (
        PROJECT_ROOT
        / "results/autoregressive/data_preparation"
        / metadata["threshold_folder"]
        / "final_datasets"
        / metadata["selection_folder"]
    )
    copy_tree(old_summary, destination, report)
    threshold_summary = (
        PROJECT_ROOT
        / "results/autoregressive/data_preparation"
        / metadata["threshold_folder"]
    )
    for path in sorted(threshold_summary.glob("*")):
        if path.is_file():
            copy_file(path, destination / path.name, report)
    copy_file(
        dataset_folder / "dataset_metadata.yaml",
        destination / "dataset_metadata.yaml",
        report,
    )
    upsert_index_row(
        get_results_index_dir() / "datasets.csv",
        {
            "selection_id": selection_id,
            "dataset_selection_mode": metadata["dataset_selection_mode"],
            "selected_datasets": ";".join(metadata["selected_datasets"]),
            "threshold": metadata["threshold"],
            "context_length": metadata["context_length"],
            "prediction_length": metadata["prediction_length"],
            "dataset_path": relative_project_path(dataset_folder),
            "num_train_windows": metadata["num_train_windows"],
            "num_val_windows": metadata["num_val_windows"],
            "num_test_windows": metadata["num_test_windows"],
            "created_at": metadata["created_at"],
        },
        id_column="selection_id",
        columns=DATASET_INDEX_COLUMNS,
    )
    return selection_id, metadata


def migrate_gru_runs(
    selection_id: str,
    metadata: dict,
    report: dict[str, list[str]],
) -> None:
    old_results_root = (
        PROJECT_ROOT / "results/autoregressive/gru" / metadata["selection_folder"]
    )
    old_models_root = (
        PROJECT_ROOT / "models/autoregressive/gru" / metadata["selection_folder"]
    )
    metrics_names = {
        "metrics_summary.csv",
        "metrics_by_horizon.csv",
        "metrics_by_dataset.csv",
        "metrics_by_quality_flag.csv",
        "training_history.csv",
    }
    for architecture_dir in sorted(old_results_root.glob("gru_*")):
        if not architecture_dir.is_dir():
            continue
        for variant_dir in sorted(path for path in architecture_dir.iterdir() if path.is_dir()):
            architecture = architecture_dir.name
            variant = variant_dir.name
            run_id = make_run_id("gru", architecture, variant, selection_id)
            run_dir = get_run_dir(run_id)
            run_paths = ensure_results_subdirs(
                run_dir,
                ("metrics", "predictions", "figures", "tables"),
            )
            for filename in metrics_names:
                copy_file(
                    variant_dir / filename,
                    run_paths["metrics"] / filename,
                    report,
                )
            copy_file(
                variant_dir / "test_predictions.parquet",
                run_paths["predictions"] / "test_predictions.parquet",
                report,
            )
            copy_tree(variant_dir / "figures", run_paths["figures"], report)
            copy_file(variant_dir / "run_metadata.yaml", run_dir / "metadata.yaml", report)

            old_model_dir = old_models_root / architecture / variant
            model_dir = get_model_dir("gru", run_id)
            copy_tree(old_model_dir, model_dir, report)
            run_metadata = load_yaml_config(variant_dir / "run_metadata.yaml")
            upsert_index_row(
                get_results_index_dir() / "runs.csv",
                {
                    "run_id": run_id,
                    "model_family": "gru",
                    "architecture": architecture,
                    "variant": variant,
                    "selection_id": selection_id,
                    "dataset_selection": ";".join(metadata["selected_datasets"]),
                    "threshold": metadata["threshold"],
                    "context_length": metadata["context_length"],
                    "prediction_length": metadata["prediction_length"],
                    "results_path": relative_project_path(run_dir),
                    "model_path": relative_project_path(model_dir),
                    "predictions_path": relative_project_path(
                        run_paths["predictions"] / "test_predictions.parquet"
                    ),
                    "metrics_path": relative_project_path(run_paths["metrics"]),
                    "status": "mirrored",
                    "created_at": run_metadata["created_at"],
                },
                id_column="run_id",
                columns=RUN_INDEX_COLUMNS,
            )

    old_comparison = old_results_root / "comparison/gru_architecture_variant_comparison.csv"
    if not old_comparison.is_file():
        return
    comparison_id = f"gru_architecture_variant_{selection_id}"
    comparison_dir = get_comparison_dir(comparison_id)
    comparison_paths = ensure_results_subdirs(
        comparison_dir,
        ("tables",),
    )
    copy_file(
        old_comparison,
        comparison_paths["tables"] / old_comparison.name,
        report,
    )
    comparison_created_at = datetime.fromtimestamp(
        old_comparison.stat().st_mtime,
        timezone.utc,
    ).isoformat()
    write_yaml_if_absent(
        comparison_dir / "metadata.yaml",
        {
            "comparison_id": comparison_id,
            "selection_id": selection_id,
            "comparison_type": "forecast_metrics_across_gru_runs",
            "table": relative_project_path(
                comparison_paths["tables"] / old_comparison.name
            ),
            "migration_source": relative_project_path(old_comparison),
            "created_at": comparison_created_at,
        },
        report,
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": comparison_id,
            "method_id": "multiple_gru_runs",
            "reference_id": "",
            "comparison_type": "forecast_metrics",
            "selection_id": selection_id,
            "results_path": relative_project_path(comparison_dir),
            "metrics_path": relative_project_path(comparison_paths["tables"]),
            "figures_path": "",
            "status": "mirrored",
            "created_at": comparison_created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )


def migrate_perfect_switch(
    selection_id: str,
    metadata: dict,
    report: dict[str, list[str]],
) -> None:
    old_dir = (
        PROJECT_ROOT
        / "results/switching/perfect_switch"
        / metadata["selection_folder"]
    )
    if not old_dir.is_dir():
        return
    destination = get_perfect_switch_dir(selection_id)
    paths = ensure_results_subdirs(destination, ("tables", "figures"))
    for path in sorted(old_dir.glob("*")):
        if path.is_file() and path.name != "perfect_switch_metadata.yaml":
            copy_file(path, paths["tables"] / path.name, report)
    copy_tree(old_dir / "figures", paths["figures"], report)
    copy_file(
        old_dir / "perfect_switch_metadata.yaml",
        destination / "metadata.yaml",
        report,
    )
    perfect_metadata = load_yaml_config(old_dir / "perfect_switch_metadata.yaml")
    upsert_index_row(
        get_results_index_dir() / "switch_references.csv",
        {
            "reference_id": f"perfect_switch_{selection_id}",
            "reference_type": "perfect_switch",
            "selection_id": selection_id,
            "threshold": perfect_metadata["threshold"],
            "switch_time": perfect_metadata["switch_time"],
            "results_path": relative_project_path(destination),
            "timeseries_path": relative_project_path(
                paths["tables"] / "perfect_switch_timeseries.parquet"
            ),
            "window_targets_path": relative_project_path(
                paths["tables"] / "perfect_switch_window_targets.parquet"
            ),
            "status": "mirrored",
            "created_at": perfect_metadata["created_at"],
        },
        id_column="reference_id",
        columns=SWITCH_REFERENCE_INDEX_COLUMNS,
    )


def main() -> None:
    report: dict[str, list[str]] = {
        "copied": [],
        "already_mirrored": [],
        "conflicts": [],
        "unclassified": [],
    }
    index_dir = get_results_index_dir()
    ensure_empty_index(index_dir / "runs.csv", RUN_INDEX_COLUMNS)
    ensure_empty_index(index_dir / "switch_references.csv", SWITCH_REFERENCE_INDEX_COLUMNS)
    ensure_empty_index(index_dir / "comparisons.csv", COMPARISON_INDEX_COLUMNS)
    ensure_empty_index(index_dir / "datasets.csv", DATASET_INDEX_COLUMNS)

    dataset_roots = sorted(
        (
            PROJECT_ROOT / "data/processed/autoregressive"
        ).glob("threshold_*/datasets_L*_h*/*/dataset_metadata.yaml")
    )
    classified_threshold_folders = {
        metadata_path.parent.parent.parent.name for metadata_path in dataset_roots
    }
    for metadata_path in dataset_roots:
        dataset_folder = metadata_path.parent
        selection_id, metadata = migrate_dataset_selection(dataset_folder, report)
        migrate_gru_runs(selection_id, metadata, report)
        migrate_perfect_switch(selection_id, metadata, report)

    old_switch_comparisons = PROJECT_ROOT / "results/switching/comparisons"
    if old_switch_comparisons.exists():
        report["unclassified"].append(
            f"{relative_project_path(old_switch_comparisons)}: combined multi-method "
            "comparison; regenerate with scripts/analysis/13_compare_switch_methods.py"
        )
    for path in sorted(
        (PROJECT_ROOT / "results/autoregressive/data_preparation").glob("threshold_*")
    ):
        if path.name not in classified_threshold_folders:
            report["unclassified"].append(
                f"{relative_project_path(path)}: no final dataset selection metadata"
            )

    print("=== Conservative result migration report ===")
    for category, entries in report.items():
        print(f"{category}: {len(entries)}")
        for entry in entries:
            print(f"  - {entry}")
    print("Old outputs were not deleted.")


if __name__ == "__main__":
    main()
