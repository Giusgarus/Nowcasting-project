"""Final supervised dataset construction from traceable autoregressive windows."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.data_quality import safe_name

SUPPORTED_SELECTION_MODES = {
    "auto_largest",
    "single_dataset",
    "multiple_datasets",
    "all_datasets",
    "external_holdout",
}
SPLITS = ("train", "validation", "test")
SPLIT_FILE_NAMES = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def filter_window_index(
    window_index: pd.DataFrame,
    *,
    allow_warning_events: bool,
) -> pd.DataFrame:
    """Return usable windows and optionally warning windows."""

    allowed = {"usable", "warning"} if allow_warning_events else {"usable"}
    return window_index.loc[window_index["quality_flag"].isin(allowed)].copy()


def rank_datasets_for_autoregressive_training(
    window_index: pd.DataFrame,
    *,
    allow_warning_events: bool,
    ranking_metric: str,
) -> pd.DataFrame:
    """Rank available datasets by valid autoregressive-window count."""

    if ranking_metric != "num_valid_autoregressive_windows":
        raise ValueError(
            "Only ranking_metric='num_valid_autoregressive_windows' is supported."
        )
    rows = []
    for (dataset_id, dataset_name), frame in window_index.groupby(
        ["dataset_id", "dataset_name"],
        sort=False,
    ):
        valid = filter_window_index(
            frame,
            allow_warning_events=allow_warning_events,
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "num_windows": len(valid),
                "num_valid_autoregressive_windows": len(valid),
                "num_usable_windows": int(frame["quality_flag"].eq("usable").sum()),
                "num_warning_windows": int(frame["quality_flag"].eq("warning").sum()),
                "num_events": int(valid["event_id"].nunique()),
                "num_usable_events": int(
                    frame.loc[frame["quality_flag"].eq("usable"), "event_id"].nunique()
                ),
                "num_warning_events": int(
                    frame.loc[frame["quality_flag"].eq("warning"), "event_id"].nunique()
                ),
                "num_train_windows": int(valid["split"].eq("train").sum()),
                "num_val_windows": int(valid["split"].eq("validation").sum()),
                "num_test_windows": int(valid["split"].eq("test").sum()),
                "ranking_metric": ranking_metric,
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise ValueError("The window index contains no datasets.")
    return ranking.sort_values(
        ["num_valid_autoregressive_windows", "dataset_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def select_datasets(
    ranking: pd.DataFrame,
    *,
    mode: str,
    selected_datasets: Sequence[str],
    heldout_test_dataset: str | None = None,
    include_all_except_heldout: bool = True,
    explicit_development_datasets: Sequence[str] | None = None,
) -> tuple[list[str], str, pd.DataFrame]:
    """Select dataset names and return a collision-resistant output folder."""

    if mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(f"Unsupported dataset-selection mode: {mode}")
    available = ranking["dataset_name"].tolist()
    requested = list(dict.fromkeys(selected_datasets))

    if mode == "auto_largest":
        chosen = [ranking.iloc[0]["dataset_name"]]
        selection_folder = f"auto_largest_{_dataset_folder_name(chosen[0])}"
    elif mode == "single_dataset":
        if len(requested) != 1:
            raise ValueError("single_dataset requires exactly one selected dataset.")
        chosen = requested
        selection_folder = f"single_{_dataset_folder_name(chosen[0])}"
    elif mode == "multiple_datasets":
        if len(requested) < 2:
            raise ValueError("multiple_datasets requires at least two selected datasets.")
        chosen = requested
        digest = hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest()[:10]
        selection_folder = f"multi_{len(chosen)}datasets_{digest}"
    elif mode == "external_holdout":
        if not heldout_test_dataset:
            raise ValueError("external_holdout requires heldout_test_dataset.")
        if heldout_test_dataset not in available:
            raise ValueError(
                f"Held-out test dataset not found in window index: "
                f"{heldout_test_dataset}"
            )
        if explicit_development_datasets:
            development = list(dict.fromkeys(explicit_development_datasets))
        elif include_all_except_heldout:
            development = [
                dataset for dataset in available if dataset != heldout_test_dataset
            ]
        else:
            raise ValueError(
                "external_holdout requires explicit development datasets or "
                "include_all_except_heldout=true."
            )
        if heldout_test_dataset in development:
            raise ValueError("Held-out dataset cannot be a development dataset.")
        chosen = development + [heldout_test_dataset]
        selection_folder = (
            f"externalHoldout_test_{_dataset_folder_name(heldout_test_dataset)}"
        )
    else:
        chosen = available
        selection_folder = "all_datasets"

    missing = sorted(set(chosen) - set(available))
    if missing:
        raise ValueError(f"Selected datasets not found in window index: {missing}")
    selected_ranking = ranking.copy()
    selected_ranking["is_selected"] = selected_ranking["dataset_name"].isin(chosen)
    return chosen, selection_folder, selected_ranking


def add_global_window_identifiers(window_index: pd.DataFrame) -> pd.DataFrame:
    """Add stable dataset-scoped event/window IDs without dropping source IDs."""

    frame = window_index.copy()
    frame["source_event_id"] = frame["event_id"].astype(str)
    frame["source_window_id"] = frame["window_id"].astype(str)
    dataset_component = frame["dataset_name"].astype(str)
    frame["global_event_id"] = dataset_component + "::" + frame["source_event_id"]
    frame["global_window_id"] = dataset_component + "::" + frame["source_window_id"]
    return frame


def assign_external_holdout_splits(
    window_index: pd.DataFrame,
    *,
    heldout_test_dataset: str,
    development_datasets: Sequence[str],
    train_ratio: float,
    min_events_for_validation_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign external test and development train/validation splits by event."""

    if not 0 < train_ratio < 1:
        raise ValueError("development_split.train_ratio must lie in (0, 1).")
    if min_events_for_validation_split < 2:
        raise ValueError("min_events_for_validation_split must be at least 2.")
    frame = add_global_window_identifiers(window_index)
    available = set(frame["dataset_name"])
    missing = sorted(set(development_datasets).difference(available))
    if missing:
        raise ValueError(f"Development datasets not found in window index: {missing}")
    if heldout_test_dataset not in available:
        raise ValueError(
            f"Held-out test dataset not found in window index: {heldout_test_dataset}"
        )
    if heldout_test_dataset in development_datasets:
        raise ValueError("Held-out dataset cannot be used for development.")

    selected = frame.loc[
        frame["dataset_name"].isin(set(development_datasets) | {heldout_test_dataset})
    ].copy()
    selected["split"] = ""
    summary_rows: list[dict[str, object]] = []

    test_mask = selected["dataset_name"].eq(heldout_test_dataset)
    selected.loc[test_mask, "split"] = "test"
    test_events = _chronological_event_table(selected.loc[test_mask])
    summary_rows.append(
        {
            "dataset_name": heldout_test_dataset,
            "role": "external_test",
            "num_events": len(test_events),
            "num_train_events": 0,
            "num_val_events": 0,
            "num_test_events": len(test_events),
            "num_train_windows": 0,
            "num_val_windows": 0,
            "num_test_windows": int(test_mask.sum()),
            "policy": "use_full_dataset",
        }
    )

    for dataset_name in development_datasets:
        dataset_mask = selected["dataset_name"].eq(dataset_name)
        event_table = _chronological_event_table(selected.loc[dataset_mask])
        event_ids = event_table["global_event_id"].tolist()
        if len(event_ids) < min_events_for_validation_split:
            train_events = set(event_ids)
            val_events: set[str] = set()
            policy = "put_all_in_train"
        else:
            train_count = int(np.floor(len(event_ids) * train_ratio))
            train_count = min(max(train_count, 1), len(event_ids) - 1)
            train_events = set(event_ids[:train_count])
            val_events = set(event_ids[train_count:])
            policy = "chronological_train_validation"

        event_values = selected.loc[dataset_mask, "global_event_id"]
        selected.loc[dataset_mask & event_values.isin(train_events), "split"] = "train"
        selected.loc[
            dataset_mask & event_values.isin(val_events), "split"
        ] = "validation"
        dataset_frame = selected.loc[dataset_mask]
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "role": "development",
                "num_events": len(event_ids),
                "num_train_events": len(train_events),
                "num_val_events": len(val_events),
                "num_test_events": 0,
                "num_train_windows": int(dataset_frame["split"].eq("train").sum()),
                "num_val_windows": int(
                    dataset_frame["split"].eq("validation").sum()
                ),
                "num_test_windows": 0,
                "policy": policy,
            }
        )

    if selected["split"].eq("").any():
        raise ValueError("External holdout split assignment left unassigned windows.")
    split_plan = pd.DataFrame(summary_rows)
    validate_external_holdout_split(
        selected,
        heldout_test_dataset=heldout_test_dataset,
        development_datasets=development_datasets,
        min_events_for_validation_split=min_events_for_validation_split,
        split_plan=split_plan,
    )
    return selected.reset_index(drop=True), split_plan


def validate_external_holdout_split(
    selected_index: pd.DataFrame,
    *,
    heldout_test_dataset: str,
    development_datasets: Sequence[str],
    min_events_for_validation_split: int,
    split_plan: pd.DataFrame,
) -> None:
    """Fail fast on leakage or malformed external-holdout split assignments."""

    required = {
        "dataset_name",
        "split",
        "global_event_id",
        "global_window_id",
        "input_start_time",
    }
    missing = sorted(required - set(selected_index.columns))
    if missing:
        raise ValueError(f"External holdout index is missing columns: {missing}")
    heldout_splits = set(
        selected_index.loc[
            selected_index["dataset_name"].eq(heldout_test_dataset), "split"
        ]
    )
    if heldout_splits != {"test"}:
        raise ValueError("Held-out dataset must appear only in the test split.")
    non_test = selected_index.loc[
        selected_index["split"].isin(["train", "validation"])
        & selected_index["dataset_name"].eq(heldout_test_dataset)
    ]
    if not non_test.empty:
        raise ValueError("Held-out dataset leaked into development splits.")
    if selected_index.groupby("global_event_id")["split"].nunique().max() > 1:
        raise ValueError("A global_event_id appears in more than one split.")
    if selected_index["global_window_id"].duplicated().any():
        raise ValueError("global_window_id values must be unique.")

    for dataset_name in development_datasets:
        frame = selected_index.loc[selected_index["dataset_name"].eq(dataset_name)]
        event_table = _chronological_event_table(frame)
        splits_by_event = (
            frame.groupby("global_event_id", sort=False)["split"].first().to_dict()
        )
        if len(event_table) < min_events_for_validation_split:
            if set(frame["split"]) != {"train"}:
                raise ValueError(
                    f"Development dataset {dataset_name} has fewer than "
                    f"{min_events_for_validation_split} events and must be all train."
                )
            continue
        ordered_splits = [
            splits_by_event[event_id]
            for event_id in event_table["global_event_id"].tolist()
        ]
        if ordered_splits != sorted(ordered_splits, key={"train": 0, "validation": 1}.get):
            raise ValueError(
                f"Development dataset {dataset_name} is not split chronologically."
            )
        if "validation" not in ordered_splits:
            raise ValueError(
                f"Development dataset {dataset_name} should have validation events."
            )

    if selected_index["split"].eq("train").sum() <= 0:
        raise ValueError("External holdout dataset build produced no train windows.")
    if selected_index["split"].eq("test").sum() <= 0:
        raise ValueError("External holdout dataset build produced no test windows.")
    val_expected_datasets = split_plan.loc[
        split_plan["num_val_events"].gt(0), "dataset_name"
    ].tolist()
    if val_expected_datasets and selected_index["split"].eq("validation").sum() <= 0:
        raise ValueError("External holdout dataset build produced no validation windows.")
    if len(development_datasets) > 1:
        train_datasets = set(
            selected_index.loc[selected_index["split"].eq("train"), "dataset_name"]
        )
        if len(train_datasets.intersection(development_datasets)) < 2:
            raise ValueError("Train split should contain multiple development datasets.")


def _chronological_event_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return event IDs ordered by their first available input timestamp."""

    if frame.empty:
        return pd.DataFrame(columns=["global_event_id", "event_start_time"])
    return (
        frame.groupby("global_event_id", sort=False)["input_start_time"]
        .min()
        .rename("event_start_time")
        .reset_index()
        .sort_values(["event_start_time", "global_event_id"], kind="stable")
        .reset_index(drop=True)
    )


def validate_event_split_integrity(window_index: pd.DataFrame) -> None:
    """Validate immutable split and identity constraints in a window index."""

    required = {
        "window_id",
        "event_id",
        "dataset_id",
        "dataset_name",
        "segment_id",
        "split",
        "quality_flag",
        "input_start_idx",
        "input_end_idx",
        "target_start_idx",
        "target_end_idx",
        "input_start_time",
        "input_end_time",
        "target_start_time",
        "target_end_time",
        "context_length",
        "prediction_length",
    }
    missing = sorted(required - set(window_index.columns))
    if missing:
        raise ValueError(f"Window index is missing required columns: {missing}")
    window_identity = (
        "global_window_id" if "global_window_id" in window_index.columns else "window_id"
    )
    event_identity = (
        "global_event_id" if "global_event_id" in window_index.columns else "event_id"
    )
    if window_index[window_identity].duplicated().any():
        raise ValueError(f"{window_identity} values must be unique.")
    if not set(window_index["split"]).issubset(SPLITS):
        raise ValueError("Window index contains unsupported split labels.")
    if window_index.groupby(event_identity)["split"].nunique().max() > 1:
        raise ValueError(f"An {event_identity} appears in more than one split.")
    if window_index.groupby(event_identity)["dataset_id"].nunique().max() > 1:
        raise ValueError(f"An {event_identity} appears in more than one dataset.")
    input_lengths = (
        window_index["input_end_idx"] - window_index["input_start_idx"] + 1
    )
    target_lengths = (
        window_index["target_end_idx"] - window_index["target_start_idx"] + 1
    )
    if not input_lengths.eq(window_index["context_length"]).all():
        raise ValueError("Window-index input lengths do not match context_length.")
    if not target_lengths.eq(window_index["prediction_length"]).all():
        raise ValueError("Window-index target lengths do not match prediction_length.")
    if not window_index["target_start_idx"].gt(window_index["input_end_idx"]).all():
        raise ValueError("Input and target indices overlap.")
    if not window_index["target_start_time"].gt(window_index["input_end_time"]).all():
        raise ValueError("Input and target timestamps overlap.")


def compute_context_standardization(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Standardize input and target using statistics from each input context."""

    if epsilon <= 0:
        raise ValueError("Normalization epsilon must be positive.")
    if X_raw.ndim != 3 or X_raw.shape[-1] != 1:
        raise ValueError("X_raw must have shape (N, context_length, 1).")
    if y_raw.ndim != 2 or len(X_raw) != len(y_raw):
        raise ValueError("y_raw must have shape (N, prediction_length).")

    means = X_raw.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    raw_stds = X_raw.std(axis=(1, 2), dtype=np.float64).astype(np.float32)
    safe_stds = np.maximum(raw_stds, np.float32(epsilon))
    X_standard = (X_raw - means[:, None, None]) / safe_stds[:, None, None]
    y_standard = (y_raw - means[:, None]) / safe_stds[:, None]
    return {
        "X_context_standard": X_standard.astype(np.float32),
        "y_context_standard": y_standard.astype(np.float32),
        "scaling_mean": means,
        "scaling_std": safe_stds,
        "scaling_std_raw": raw_stds,
    }


def build_autoregressive_arrays_from_window_index(
    event_windows: pd.DataFrame,
    window_index: pd.DataFrame,
    *,
    context_length: int,
    prediction_length: int,
    signal_column: str,
    epsilon: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    """Build split arrays and aligned metadata from the official window index."""

    validate_event_split_integrity(window_index)
    if not window_index["context_length"].eq(context_length).all():
        raise ValueError("Configured context_length does not match the window index.")
    if not window_index["prediction_length"].eq(prediction_length).all():
        raise ValueError("Configured prediction_length does not match the window index.")
    required_event_columns = {
        "event_id",
        "dataset_id",
        "dataset_name",
        "segment_id",
        "Time",
        "event_point_idx",
        signal_column,
    }
    missing = sorted(required_event_columns - set(event_windows.columns))
    if missing:
        raise ValueError(f"Event windows are missing required columns: {missing}")
    event_windows = event_windows.copy()
    if "global_event_id" in window_index.columns:
        event_windows["global_event_id"] = (
            event_windows["dataset_name"].astype(str)
            + "::"
            + event_windows["event_id"].astype(str)
        )
        event_identity = "global_event_id"
        if "split" in event_windows.columns:
            split_lookup = (
                window_index[["global_event_id", "split"]]
                .drop_duplicates()
                .set_index("global_event_id")["split"]
            )
            event_windows["split"] = event_windows["global_event_id"].map(
                split_lookup
            ).fillna(event_windows["split"])
    else:
        event_identity = "event_id"
    if event_windows.duplicated([event_identity, "event_point_idx"]).any():
        raise ValueError(f"event_point_idx must be unique within every {event_identity}.")

    event_lookup = {
        event_id: frame.set_index("event_point_idx", drop=False).sort_index()
        for event_id, frame in event_windows.groupby(event_identity, sort=False)
    }
    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for window in window_index.itertuples(index=False):
        lookup_id = getattr(window, event_identity)
        if lookup_id not in event_lookup:
            raise ValueError(f"Event window not found for {lookup_id}.")
        event_frame = event_lookup[lookup_id]
        input_rows = _select_indexed_rows(
            event_frame,
            window.input_start_idx,
            window.input_end_idx,
            expected_length=context_length,
            label="input",
            window_id=window.window_id,
        )
        target_rows = _select_indexed_rows(
            event_frame,
            window.target_start_idx,
            window.target_end_idx,
            expected_length=prediction_length,
            label="target",
            window_id=window.window_id,
        )
        combined = pd.concat([input_rows, target_rows])
        _validate_window_traceability(combined, window, signal_column=signal_column)
        _validate_window_times(input_rows, target_rows, window)
        X_rows.append(input_rows[signal_column].to_numpy(dtype=np.float32)[:, None])
        y_rows.append(target_rows[signal_column].to_numpy(dtype=np.float32))

    X_raw = (
        np.stack(X_rows).astype(np.float32)
        if X_rows
        else np.empty((0, context_length, 1), dtype=np.float32)
    )
    y_raw = (
        np.stack(y_rows).astype(np.float32)
        if y_rows
        else np.empty((0, prediction_length), dtype=np.float32)
    )
    normalized = compute_context_standardization(X_raw, y_raw, epsilon=epsilon)
    all_arrays = {
        "X_raw": X_raw,
        "y_raw": y_raw,
        **normalized,
        "window_id": window_index["window_id"].to_numpy(dtype=str),
        "event_id": window_index["event_id"].to_numpy(dtype=str),
        "dataset_id": window_index["dataset_id"].to_numpy(dtype=str),
    }
    if "global_window_id" in window_index.columns:
        all_arrays["global_window_id"] = window_index["global_window_id"].to_numpy(
            dtype=str
        )
    if "global_event_id" in window_index.columns:
        all_arrays["global_event_id"] = window_index["global_event_id"].to_numpy(
            dtype=str
        )
    metadata = window_index.copy().reset_index(drop=True)
    metadata["scaling_mean"] = normalized["scaling_mean"]
    metadata["scaling_std"] = normalized["scaling_std"]
    metadata["scaling_std_raw"] = normalized["scaling_std_raw"]
    metadata["normalization_method"] = "context_standard_mean_std"

    split_arrays = {}
    split_metadata = {}
    for split in SPLITS:
        mask = metadata["split"].eq(split).to_numpy()
        split_arrays[split] = {key: value[mask] for key, value in all_arrays.items()}
        split_metadata[split] = metadata.loc[mask].reset_index(drop=True)
    return split_arrays, split_metadata


def save_split_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Save one split's raw and standardized arrays in compressed NPZ format."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def summarize_final_dataset(
    selected_index: pd.DataFrame,
    *,
    selection_mode: str,
    selection_folder: str,
    selected_datasets: Sequence[str],
    context_length: int,
    prediction_length: int,
    signal_threshold: float,
    allow_warning_events: bool,
    normalization_versions: Sequence[str],
) -> pd.DataFrame:
    """Create the one-row final supervised-dataset summary."""

    return pd.DataFrame(
        [
            {
                "selection_mode": selection_mode,
                "selection_folder": selection_folder,
                "selected_datasets": ";".join(selected_datasets),
                "num_datasets": selected_index["dataset_id"].nunique(),
                "num_windows_total": len(selected_index),
                "num_train_windows": int(selected_index["split"].eq("train").sum()),
                "num_val_windows": int(
                    selected_index["split"].eq("validation").sum()
                ),
                "num_test_windows": int(selected_index["split"].eq("test").sum()),
                "num_events_total": selected_index["event_id"].nunique(),
                "num_train_events": selected_index.loc[
                    selected_index["split"].eq("train"), "event_id"
                ].nunique(),
                "num_val_events": selected_index.loc[
                    selected_index["split"].eq("validation"), "event_id"
                ].nunique(),
                "num_test_events": selected_index.loc[
                    selected_index["split"].eq("test"), "event_id"
                ].nunique(),
                "context_length": context_length,
                "prediction_length": prediction_length,
                "signal_threshold": signal_threshold,
                "allow_warning_events": allow_warning_events,
                "normalization_versions": ";".join(normalization_versions),
            }
        ]
    )


def summarize_splits(
    split_metadata: dict[str, pd.DataFrame],
    selected_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize windows, events, quality, and imputation per split and dataset."""

    rows = []
    selected = selected_ranking.loc[selected_ranking["is_selected"]]
    for split in SPLITS:
        metadata = split_metadata[split]
        for dataset in selected.itertuples(index=False):
            frame = metadata.loc[metadata["dataset_id"].eq(dataset.dataset_id)]
            rows.append(
                {
                    "split": split,
                    "dataset_id": dataset.dataset_id,
                    "dataset_name": dataset.dataset_name,
                    "num_windows": len(frame),
                    "num_events": frame["event_id"].nunique(),
                    "num_usable_windows": int(frame["quality_flag"].eq("usable").sum()),
                    "num_warning_windows": int(
                        frame["quality_flag"].eq("warning").sum()
                    ),
                    "num_imputed_points": int(
                        frame["num_imputed_points_in_window"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_normalization(
    split_metadata: dict[str, pd.DataFrame],
    selected_ranking: pd.DataFrame,
    *,
    epsilon: float,
) -> pd.DataFrame:
    """Summarize context-derived normalization statistics."""

    rows = []
    selected = selected_ranking.loc[selected_ranking["is_selected"]]
    for split in SPLITS:
        metadata = split_metadata[split]
        for dataset in selected.itertuples(index=False):
            frame = metadata.loc[metadata["dataset_id"].eq(dataset.dataset_id)]
            rows.append(
                {
                    "split": split,
                    "dataset_id": dataset.dataset_id,
                    "dataset_name": dataset.dataset_name,
                    "num_windows": len(frame),
                    **_normalization_statistics(frame, epsilon),
                }
            )
    return pd.DataFrame(rows)


def _select_indexed_rows(
    event_frame: pd.DataFrame,
    start: int,
    end: int,
    *,
    expected_length: int,
    label: str,
    window_id: str,
) -> pd.DataFrame:
    indices = np.arange(start, end + 1)
    missing = indices[~np.isin(indices, event_frame.index.to_numpy())]
    if len(missing):
        raise ValueError(f"{window_id} {label} rows are missing: {missing.tolist()}")
    selected = event_frame.loc[indices]
    if len(selected) != expected_length:
        raise ValueError(f"{window_id} has an invalid {label} length.")
    return selected


def _validate_window_traceability(
    frame: pd.DataFrame,
    window: Any,
    *,
    signal_column: str,
) -> None:
    checks = {
        "event_id": window.event_id,
        "dataset_id": window.dataset_id,
        "dataset_name": window.dataset_name,
        "segment_id": window.segment_id,
    }
    for column, expected in checks.items():
        if not frame[column].eq(expected).all():
            raise ValueError(
                f"{window.window_id} crosses {column} boundaries or is misaligned."
            )
    if "split" in frame and not frame["split"].eq(window.split).all():
        raise ValueError(f"{window.window_id} event-window split is misaligned.")
    if not np.isfinite(frame[signal_column].to_numpy(dtype=float)).all():
        raise ValueError(
            f"{window.window_id} contains non-finite {signal_column} values."
        )


def _validate_window_times(
    input_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    window: Any,
) -> None:
    actual = (
        input_rows["Time"].iloc[0],
        input_rows["Time"].iloc[-1],
        target_rows["Time"].iloc[0],
        target_rows["Time"].iloc[-1],
    )
    expected = (
        window.input_start_time,
        window.input_end_time,
        window.target_start_time,
        window.target_end_time,
    )
    if any(pd.Timestamp(left) != pd.Timestamp(right) for left, right in zip(actual, expected)):
        raise ValueError(f"{window.window_id} timestamps do not match the window index.")


def _dataset_folder_name(dataset_name: str) -> str:
    return safe_name(Path(dataset_name).stem).replace("-", "_").replace(".", "_")


def _normalization_statistics(
    frame: pd.DataFrame,
    epsilon: float,
) -> dict[str, float | int]:
    if frame.empty:
        return {
            "mean_scaling_mean": np.nan,
            "std_scaling_mean": np.nan,
            "min_scaling_mean": np.nan,
            "max_scaling_mean": np.nan,
            "mean_scaling_std": np.nan,
            "std_scaling_std": np.nan,
            "min_scaling_std": np.nan,
            "max_scaling_std": np.nan,
            "num_near_zero_std_windows": 0,
        }
    means = frame["scaling_mean"]
    raw_stds = frame["scaling_std_raw"]
    return {
        "mean_scaling_mean": float(means.mean()),
        "std_scaling_mean": float(means.std(ddof=0)),
        "min_scaling_mean": float(means.min()),
        "max_scaling_mean": float(means.max()),
        "mean_scaling_std": float(raw_stds.mean()),
        "std_scaling_std": float(raw_stds.std(ddof=0)),
        "min_scaling_std": float(raw_stds.min()),
        "max_scaling_std": float(raw_stds.max()),
        "num_near_zero_std_windows": int(raw_stds.le(epsilon).sum()),
    }
