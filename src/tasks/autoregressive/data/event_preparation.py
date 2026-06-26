"""Summaries for prepared candidate-event datasets."""

from collections.abc import Sequence

import pandas as pd


def summarize_threshold_balance(
    event_windows: pd.DataFrame,
    *,
    signal_threshold: float,
    dataset_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize above/below-threshold points per dataset and globally."""

    rows = []
    names = (
        list(dataset_names)
        if dataset_names is not None
        else event_windows["dataset_name"].drop_duplicates().tolist()
    )
    for dataset_name in names:
        frame = event_windows.loc[event_windows["dataset_name"].eq(dataset_name)]
        rows.append(
            _threshold_row("dataset", dataset_name, frame, signal_threshold)
        )
    rows.append(_threshold_row("global", "ALL", event_windows, signal_threshold))
    return pd.DataFrame(rows)


def summarize_imputation(
    event_windows: pd.DataFrame,
    event_quality: pd.DataFrame,
    *,
    dataset_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize event-window imputation and missing-gap outcomes."""

    rows = []
    names = (
        list(dataset_names)
        if dataset_names is not None
        else event_quality["dataset_name"].drop_duplicates().tolist()
    )
    for dataset_name in names:
        quality = event_quality.loc[event_quality["dataset_name"].eq(dataset_name)]
        windows = event_windows.loc[event_windows["dataset_name"].eq(dataset_name)]
        total_points = len(windows)
        reason = quality["quality_reason"].fillna("")
        rows.append(
            {
                "dataset_name": dataset_name,
                "num_events_with_imputation": int(
                    quality["num_imputed_points"].gt(0).sum()
                ),
                "num_imputed_points": int(windows["is_imputed"].sum()),
                "pct_imputed_points": _percentage(
                    int(windows["is_imputed"].sum()),
                    total_points,
                ),
                "max_consecutive_imputed_points": int(
                    quality["max_consecutive_imputed_points"].max()
                    if len(quality)
                    else 0
                ),
                "num_warning_events_due_to_imputation": int(
                    quality["quality_flag"].eq("warning")
                    .where(reason.str.contains("imputed"), False)
                    .sum()
                ),
                "num_unusable_events_due_to_missing_or_gaps": int(
                    quality["quality_flag"].eq("unusable")
                    .where(reason.str.contains("gap|missing"), False)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_window_index(
    events: pd.DataFrame,
    event_quality: pd.DataFrame,
    window_index: pd.DataFrame,
    *,
    signal_threshold: float,
    context_length: int,
    prediction_length: int,
    dataset_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize generated autoregressive windows per dataset."""

    rows = []
    names = (
        list(dataset_names)
        if dataset_names is not None
        else events["dataset_name"].drop_duplicates().tolist()
    )
    for dataset_name in names:
        quality = event_quality.loc[event_quality["dataset_name"].eq(dataset_name)]
        windows = window_index.loc[window_index["dataset_name"].eq(dataset_name)]
        rows.append(
            {
                "dataset_name": dataset_name,
                "signal_threshold": signal_threshold,
                "context_length": context_length,
                "prediction_length": prediction_length,
                "num_windows": len(windows),
                "train_windows": int(windows["split"].eq("train").sum()),
                "val_windows": int(windows["split"].eq("validation").sum()),
                "test_windows": int(windows["split"].eq("test").sum()),
                "usable_event_windows": int(quality["quality_flag"].eq("usable").sum()),
                "warning_event_windows_used": int(
                    windows.loc[windows["quality_flag"].eq("warning"), "event_id"].nunique()
                ),
                "unusable_event_windows_excluded": int(
                    quality["quality_flag"].eq("unusable").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_event_dataset(
    events: pd.DataFrame,
    event_windows: pd.DataFrame,
    event_quality: pd.DataFrame,
    window_index: pd.DataFrame,
    *,
    signal_threshold: float,
    dataset_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create the main per-dataset event-preparation summary."""

    rows = []
    names = (
        list(dataset_names)
        if dataset_names is not None
        else events["dataset_name"].drop_duplicates().tolist()
    )
    for dataset_name in names:
        dataset_events = events.loc[events["dataset_name"].eq(dataset_name)]
        windows = event_windows.loc[event_windows["dataset_name"].eq(dataset_name)]
        quality = event_quality.loc[event_quality["dataset_name"].eq(dataset_name)]
        indices = window_index.loc[window_index["dataset_name"].eq(dataset_name)]
        above = windows["Signal_prepared"].gt(signal_threshold)
        rows.append(
            {
                "dataset_name": dataset_name,
                "signal_threshold": signal_threshold,
                "num_events": len(dataset_events),
                "num_usable_events": int(quality["quality_flag"].eq("usable").sum()),
                "num_warning_events": int(quality["quality_flag"].eq("warning").sum()),
                "num_unusable_events": int(quality["quality_flag"].eq("unusable").sum()),
                "total_points_in_event_windows": len(windows),
                "points_above_threshold": int(above.sum()),
                "points_below_or_equal_threshold": int((~above).sum()),
                "pct_above_threshold": float(above.mean() * 100) if len(above) else 0.0,
                "pct_below_or_equal_threshold": float((~above).mean() * 100)
                if len(above)
                else 0.0,
                "num_imputed_points": int(windows["is_imputed"].sum()),
                "pct_imputed_points": _percentage(int(windows["is_imputed"].sum()), len(windows)),
                "num_large_gaps": int(quality["num_gaps_above_90s"].sum()),
                "num_valid_autoregressive_windows": len(indices),
                "train_windows": int(indices["split"].eq("train").sum()),
                "val_windows": int(indices["split"].eq("validation").sum()),
                "test_windows": int(indices["split"].eq("test").sum()),
            }
        )
    return pd.DataFrame(rows)


def _threshold_row(
    scope: str,
    dataset_name: str,
    frame: pd.DataFrame,
    signal_threshold: float,
) -> dict[str, object]:
    above = frame["Signal_prepared"].gt(signal_threshold)
    return {
        "scope": scope,
        "dataset_name": dataset_name,
        "signal_threshold": signal_threshold,
        "total_points": len(frame),
        "points_above_threshold": int(above.sum()),
        "points_below_or_equal_threshold": int((~above).sum()),
        "pct_above_threshold": float(above.mean() * 100) if len(above) else 0.0,
        "pct_below_or_equal_threshold": float((~above).mean() * 100)
        if len(above)
        else 0.0,
    }


def _percentage(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0
