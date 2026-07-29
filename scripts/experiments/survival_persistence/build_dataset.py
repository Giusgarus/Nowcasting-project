"""Build the continuous-time survival-persistence dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import assign_external_holdout_splits  # noqa: E402
from src.data.imputation import (  # noqa: E402
    impute_small_time_gaps,
    small_gap_imputation_config_from_mapping,
)
from src.tasks.survival_persistence.data.dataset import (  # noqa: E402
    SCALAR_CONTEXT_FEATURE_NAMES,
    SPLIT_FILE_NAMES,
    SPLITS,
    build_survival_arrays,
    build_survival_window_index,
    load_full_signal,
    save_split_npz,
    split_metadata_for_yaml,
    summarize_censoring,
    summarize_event_index,
    summarize_split_metadata,
    validate_model_input_fields,
)
from src.tasks.survival_persistence.data.event_state_machine import (  # noqa: E402
    StableRecoveryConfig,
)
from src.tasks.survival_persistence.utils.paths import (  # noqa: E402
    TASK_NAME,
    data_preparation_dir,
    make_selection_id,
    processed_dataset_dir,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    DATASET_INDEX_COLUMNS,
    get_results_index_dir,
    relative_project_path,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml"
)


def project_path(path: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def required_output_paths(output_dir: Path) -> list[Path]:
    """Return files that define a complete dataset build."""

    return [
        output_dir / "train.npz",
        output_dir / "val.npz",
        output_dir / "test.npz",
        output_dir / "train_metadata.parquet",
        output_dir / "val_metadata.parquet",
        output_dir / "test_metadata.parquet",
        output_dir / "event_summary.csv",
        output_dir / "censoring_summary.csv",
        output_dir / "split_summary.csv",
        output_dir / "dataset_metadata.yaml",
    ]


def dataset_is_complete(output_dir: Path) -> bool:
    """Return true if all required outputs are present."""

    return all(path.exists() for path in required_output_paths(output_dir))


def recovery_config_from_yaml(config: dict[str, Any]) -> StableRecoveryConfig:
    """Build the validated recovery config dataclass from YAML."""

    event = config["event_definition"]
    return StableRecoveryConfig(
        threshold_on=float(event["threshold_on"]),
        threshold_off=float(event["threshold_off"]),
        threshold_on_operator=str(event.get("threshold_on_operator", ">=")),  # type: ignore[arg-type]
        threshold_off_operator=str(event.get("threshold_off_operator", "<")),  # type: ignore[arg-type]
        recovery_window_seconds=float(event["recovery_window_seconds"]),
        recovery_required_fraction=float(event["recovery_required_fraction"]),
        recovery_fraction_mode=str(event.get("recovery_fraction_mode", "sample_fraction")),  # type: ignore[arg-type]
        minimum_recovery_observations=int(
            event.get("minimum_recovery_observations", 2)
        ),
        abort_candidate_recovery_on_threshold_on_crossing=bool(
            event.get("abort_candidate_recovery_on_threshold_on_crossing", True)
        ),
    )


def validate_saved_dataset(selected_index: pd.DataFrame) -> None:
    """Fail loudly on survival-label or leakage invariants."""

    if selected_index.empty:
        raise ValueError("Survival dataset has no samples.")
    if selected_index["global_window_id"].duplicated().any():
        raise ValueError("global_window_id values must be unique.")
    if selected_index.groupby("global_event_id")["split"].nunique().max() > 1:
        raise ValueError("A survival event appears in more than one split.")
    if selected_index["y_time_seconds"].le(0).any():
        raise ValueError("All survival times must be strictly positive.")
    observed = selected_index.loc[selected_index["y_event_observed"].eq(1)]
    censored = selected_index.loc[selected_index["y_event_observed"].eq(0)]
    if not observed.empty:
        if not np.allclose(
            observed["y_lower_bound_seconds"].to_numpy(dtype=float),
            observed["y_upper_bound_seconds"].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-8,
        ):
            raise ValueError("Observed samples must have equal lower/upper bounds.")
        if not np.isfinite(observed["y_upper_bound_seconds"].to_numpy(dtype=float)).all():
            raise ValueError("Observed samples must have finite upper bounds.")
        if not (
            pd.to_datetime(observed["sample_time"])
            < pd.to_datetime(observed["event_end_time"])
        ).all():
            raise ValueError("Observed samples must satisfy sample_time < event_end_time.")
    if not censored.empty:
        if not np.isinf(censored["y_upper_bound_seconds"].to_numpy(dtype=float)).all():
            raise ValueError("Censored samples must have infinite upper bounds.")
        if censored["censoring_reason"].astype(str).eq("").any():
            raise ValueError("Censored samples must have explicit censoring reasons.")
    weight_sums = selected_index.groupby("global_event_id")[
        "event_balanced_weight"
    ].sum()
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("event_balanced_weight must sum to one within each event.")


def _confirmation_delay_summary(event_summary: pd.DataFrame) -> pd.DataFrame:
    observed = event_summary.loc[event_summary["event_observed"].astype(bool)].copy()
    if observed.empty:
        return pd.DataFrame(
            [{"num_observed_events": 0, "min": np.nan, "median": np.nan, "max": np.nan}]
        )
    delay = (
        pd.to_datetime(observed["event_end_confirmation_time"])
        - pd.to_datetime(observed["event_end_time"])
    ).dt.total_seconds()
    return pd.DataFrame(
        [
            {
                "num_observed_events": int(len(observed)),
                "min": float(delay.min()),
                "median": float(delay.median()),
                "max": float(delay.max()),
            }
        ]
    )


def main() -> None:
    """Build and save the configured survival-persistence dataset."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")

    validate_model_input_fields()
    recovery_config = recovery_config_from_yaml(config)
    dataset_config = config["dataset"]
    split_config = config["split"]["development_event_split"]
    source_config = config["source"]
    context_length = int(dataset_config["context_length"])
    sampling_time_seconds = int(dataset_config.get("sampling_time_seconds", 30))
    external_test_dataset = str(config["split"]["external_test_dataset"])
    selection_id = make_selection_id(external_test_dataset)
    output_dir = project_path(
        config.get(
            "output_dir",
            processed_dataset_dir(
                threshold_on=recovery_config.threshold_on,
                context_length=context_length,
                selection_id=selection_id,
            ),
        )
    )
    summary_dir = data_preparation_dir(
        selection_id=selection_id,
        threshold_on=recovery_config.threshold_on,
        context_length=context_length,
    )
    overwrite = bool(config.get("output", {}).get("overwrite", False)) or args.force

    print("=== Survival-Persistence Dataset Builder ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Activation threshold: {recovery_config.threshold_on:g}")
    print(f"Recovery threshold: {recovery_config.threshold_off:g}")
    print(f"Recovery window seconds: {recovery_config.recovery_window_seconds:g}")
    print(f"Context length: {context_length}")
    print(f"External test dataset: {external_test_dataset}")
    print(f"Output folder: {output_dir.relative_to(PROJECT_ROOT)}")
    print("Displays: no figures.")
    print("Saves: split NPZ files, split metadata parquet, event summaries, and metadata.\n")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        if dataset_is_complete(output_dir):
            print(
                "Dataset already exists and appears complete. "
                "Use --force or output.overwrite=true to rebuild."
            )
            return
        raise FileExistsError(
            f"Output directory exists but is incomplete: {output_dir}. "
            "Use --force after reviewing it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    full_signal_path = project_path(source_config["full_signal_path"])
    full_signal = load_full_signal(
        full_signal_path,
        signal_column=str(source_config.get("signal_column", "Signal")),
    )
    imputation_config = small_gap_imputation_config_from_mapping(
        config.get("small_gap_imputation"),
        enabled_default=False,
    )
    imputation_summary_path: Path | None = None
    if imputation_config.enabled:
        imputation_result = impute_small_time_gaps(
            full_signal,
            time_column="Time",
            signal_column="Signal",
            group_columns=("dataset_id", "dataset_name", "segment_id"),
            config=imputation_config,
        )
        full_signal = imputation_result.dataframe
        imputation_summary_path = output_dir / "small_gap_imputation_summary.csv"
        imputation_result.summary.to_csv(imputation_summary_path, index=False)

    window_index, event_index, diagnostics = build_survival_window_index(
        full_signal,
        recovery_config=recovery_config,
        context_length=context_length,
        sampling_time_seconds=sampling_time_seconds,
        sample_policy=str(dataset_config["sample_policy"]),
        require_contiguous_context=bool(dataset_config["require_contiguous_context"]),
    )
    if window_index.empty:
        raise ValueError("No survival-persistence windows were built.")
    available_datasets = list(dict.fromkeys(window_index["dataset_name"].tolist()))
    if external_test_dataset not in available_datasets:
        raise ValueError(
            f"External test dataset has no survival windows: {external_test_dataset}"
        )
    development_datasets = [
        dataset for dataset in available_datasets if dataset != external_test_dataset
    ]
    selected_index, split_plan = assign_external_holdout_splits(
        window_index,
        heldout_test_dataset=external_test_dataset,
        development_datasets=development_datasets,
        train_ratio=float(split_config["train_fraction"]),
        min_events_for_validation_split=int(
            split_config["min_events_for_validation_split"]
        ),
    )
    validate_saved_dataset(selected_index)

    split_arrays, split_metadata, scalar_scaler = build_survival_arrays(
        full_signal,
        selected_index,
        context_length=context_length,
        threshold_on=recovery_config.threshold_on,
        standardize_scalar_context=bool(
            config["input_representation"].get("standardize_scalar_context", True)
        ),
    )

    output_files: dict[str, str] = {}
    if imputation_summary_path is not None:
        output_files["small_gap_imputation_summary"] = relative_project_path(
            imputation_summary_path
        )
    for split in SPLITS:
        stem = SPLIT_FILE_NAMES[split]
        npz_path = output_dir / f"{stem}.npz"
        metadata_path = output_dir / f"{stem}_metadata.parquet"
        save_split_npz(npz_path, split_arrays[split])
        split_metadata[split].to_parquet(metadata_path, index=False)
        output_files[f"{stem}_npz"] = relative_project_path(npz_path)
        output_files[f"{stem}_metadata"] = relative_project_path(metadata_path)

    event_summary = summarize_event_index(selected_index, event_index)
    split_summary = summarize_split_metadata(split_metadata)
    censoring_summary = summarize_censoring(split_metadata)
    confirmation_delay = _confirmation_delay_summary(event_summary)

    split_plan_path = output_dir / "external_holdout_split_plan.csv"
    event_summary_path = output_dir / "event_summary.csv"
    split_summary_path = output_dir / "split_summary.csv"
    censoring_summary_path = output_dir / "censoring_summary.csv"
    confirmation_delay_path = output_dir / "confirmation_delay_summary.csv"
    split_plan.to_csv(split_plan_path, index=False)
    event_summary.to_csv(event_summary_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)
    censoring_summary.to_csv(censoring_summary_path, index=False)
    confirmation_delay.to_csv(confirmation_delay_path, index=False)
    for source, destination in [
        (event_summary_path, summary_dir / "event_summary.csv"),
        (split_summary_path, summary_dir / "split_summary.csv"),
        (censoring_summary_path, summary_dir / "censoring_summary.csv"),
        (confirmation_delay_path, summary_dir / "confirmation_delay_summary.csv"),
    ]:
        pd.read_csv(source).to_csv(destination, index=False)
    if imputation_summary_path is not None:
        pd.read_csv(imputation_summary_path).to_csv(
            summary_dir / "small_gap_imputation_summary.csv",
            index=False,
        )

    output_files.update(
        {
            "external_holdout_split_plan": relative_project_path(split_plan_path),
            "event_summary": relative_project_path(event_summary_path),
            "split_summary": relative_project_path(split_summary_path),
            "censoring_summary": relative_project_path(censoring_summary_path),
            "confirmation_delay_summary": relative_project_path(confirmation_delay_path),
        }
    )

    dataset_metadata = {
        "task_name": TASK_NAME,
        "schema_version": int(config.get("schema_version", 1)),
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "selection_id": selection_id,
        "context_length": context_length,
        "sampling_time_seconds": sampling_time_seconds,
        "event_definition": {
            "threshold_on": recovery_config.threshold_on,
            "threshold_off": recovery_config.threshold_off,
            "threshold_on_operator": recovery_config.threshold_on_operator,
            "threshold_off_operator": recovery_config.threshold_off_operator,
            "recovery_window_seconds": recovery_config.recovery_window_seconds,
            "recovery_required_fraction": recovery_config.recovery_required_fraction,
            "recovery_fraction_mode": recovery_config.recovery_fraction_mode,
            "minimum_recovery_observations": recovery_config.minimum_recovery_observations,
            "abort_candidate_recovery_on_threshold_on_crossing": (
                recovery_config.abort_candidate_recovery_on_threshold_on_crossing
            ),
            "state_machine": (
                "NORMAL -> ACTIVE_FADE on threshold_on; ACTIVE_FADE -> "
                "CANDIDATE_RECOVERY on threshold_off; recovery is observed only "
                "when the configured future recovery window satisfies the required "
                "sample fraction; otherwise the event remains active or is censored "
                "at segment end."
            ),
        },
        "target_definition": (
            "R_t = event_end_time - t, where event_end_time is the start of the "
            "stable recovery interval, not the later confirmation timestamp."
        ),
        "canonical_labels": [
            "y_time_seconds",
            "y_event_observed",
            "y_lower_bound_seconds",
            "y_upper_bound_seconds",
        ],
        "model_independent_dataset": True,
        "suggested_discretization": config.get("suggested_discretization", {}),
        "input_representations": [
            "X_raw",
            "X_relative_to_threshold",
            "X_relative_to_current",
            "scalar_context_features",
        ],
        "scalar_context_feature_names": list(SCALAR_CONTEXT_FEATURE_NAMES),
        "scalar_context_scaler": scalar_scaler,
        "event_balanced_weight": "1 / event_sample_count; sums to one per event",
        "external_test_dataset": external_test_dataset,
        "development_datasets": development_datasets,
        "source_datasets": available_datasets,
        "source_files": {"full_signal": relative_project_path(full_signal_path)},
        "small_gap_imputation": {
            "enabled": imputation_config.enabled,
            "expected_seconds": imputation_config.expected_seconds,
            "max_gap_seconds": imputation_config.max_gap_seconds,
            "max_missing_run": imputation_config.max_missing_run,
            "method": imputation_config.method,
            "summary": relative_project_path(imputation_summary_path)
            if imputation_summary_path is not None
            else "",
        },
        "split": config["split"],
        "splits": split_metadata_for_yaml(split_metadata),
        "diagnostics": diagnostics,
        "output_files": output_files,
        "num_train_windows": len(split_metadata["train"]),
        "num_val_windows": len(split_metadata["validation"]),
        "num_test_windows": len(split_metadata["test"]),
        "num_events_total": int(selected_index["global_event_id"].nunique()),
        "num_observed_events": int(
            event_summary["event_observed"].astype(bool).sum()
        )
        if not event_summary.empty
        else 0,
        "num_censored_events": int(
            (~event_summary["event_observed"].astype(bool)).sum()
        )
        if not event_summary.empty
        else 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / "dataset_metadata.yaml"
    dataset_metadata["output_files"]["dataset_metadata"] = relative_project_path(
        metadata_path
    )
    save_yaml(metadata_path, dataset_metadata)

    upsert_index_row(
        get_results_index_dir() / "datasets.csv",
        {
            "dataset_index_id": f"{TASK_NAME}::{selection_id}",
            "task_name": TASK_NAME,
            "selection_id": selection_id,
            "dataset_selection_mode": "external_holdout_survival_persistence",
            "selected_datasets": ";".join(available_datasets),
            "threshold": recovery_config.threshold_on,
            "context_length": context_length,
            "prediction_length": "",
            "dataset_path": relative_project_path(output_dir),
            "num_train_windows": len(split_metadata["train"]),
            "num_val_windows": len(split_metadata["validation"]),
            "num_test_windows": len(split_metadata["test"]),
            "created_at": dataset_metadata["created_at"],
        },
        id_column="dataset_index_id",
        columns=DATASET_INDEX_COLUMNS,
    )

    print("=== Compact summary ===")
    print(f"Selection ID: {selection_id}")
    print(f"Development datasets: {', '.join(development_datasets)}")
    print(f"Train samples: {len(split_metadata['train']):,}")
    print(f"Validation samples: {len(split_metadata['validation']):,}")
    print(f"Test samples: {len(split_metadata['test']):,}")
    print(
        "Observed/censored events: "
        f"{dataset_metadata['num_observed_events']:,}/"
        f"{dataset_metadata['num_censored_events']:,}"
    )
    print(f"Context exclusions: {diagnostics['num_context_exclusions']:,}")
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")
    print("Survival-persistence dataset build completed.")


if __name__ == "__main__":
    main()
