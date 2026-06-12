"""Model-agnostic alignment and evaluation against the Perfect Switch."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.switch_metrics import compute_switch_metrics
from src.switching.conversion import compute_switch_from_signal_values


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def align_forecast_predictions_to_targets(
    predictions: pd.DataFrame,
    perfect_window_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Attach target timestamps and Perfect Switch values to saved forecasts."""

    _require_columns(
        predictions,
        [
            "window_id",
            "event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "horizon_step",
            "y_true_raw",
            "y_pred_raw",
        ],
        "predictions",
    )
    _require_columns(
        perfect_window_targets,
        [
            "window_id",
            "event_id",
            "target_step",
            "target_time",
            "y_true_raw",
            "perfect_switch",
        ],
        "perfect_window_targets",
    )
    target_columns = perfect_window_targets[
        [
            "window_id",
            "event_id",
            "target_step",
            "target_time",
            "y_true_raw",
            "perfect_switch",
        ]
    ].rename(
        columns={
            "target_step": "horizon_step",
            "y_true_raw": "target_y_true_raw",
        }
    )
    aligned = predictions.merge(
        target_columns,
        on=["window_id", "event_id", "horizon_step"],
        how="left",
        validate="one_to_one",
    )
    if aligned["target_time"].isna().any():
        raise ValueError("Some model predictions could not be aligned to target timestamps.")
    if not np.allclose(
        aligned["y_true_raw"],
        aligned["target_y_true_raw"],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("Saved model y_true_raw values do not match Perfect target values.")
    return aligned.drop(columns="target_y_true_raw")


def select_latest_available_forecasts(aligned_predictions: pd.DataFrame) -> pd.DataFrame:
    """Select the smallest-horizon forecast available for every event timestamp."""

    _require_columns(
        aligned_predictions,
        ["event_id", "target_time", "horizon_step", "window_id"],
        "aligned_predictions",
    )
    ordered = aligned_predictions.sort_values(
        ["event_id", "target_time", "horizon_step", "window_id"],
        kind="stable",
    )
    selected = ordered.drop_duplicates(
        subset=["event_id", "target_time"],
        keep="first",
    ).copy()
    selected["target_time"] = pd.to_datetime(selected["target_time"], errors="raise")
    return selected.sort_values(["event_id", "target_time"], kind="stable").reset_index(
        drop=True
    )


def build_model_switch_timeseries(
    predictions: pd.DataFrame,
    perfect_window_targets: pd.DataFrame,
    *,
    method_name: str,
    threshold: float,
    condition: str,
    switch_time: int,
    apply_min_island_length: bool,
) -> pd.DataFrame:
    """Build one operational pointwise switch series from overlapping forecasts."""

    aligned = align_forecast_predictions_to_targets(predictions, perfect_window_targets)
    selected = select_latest_available_forecasts(aligned)
    selected["model_switch_raw"] = np.int8(0)
    selected["model_switch"] = np.int8(0)
    for _, indices in selected.groupby("event_id", sort=False).indices.items():
        raw, processed = compute_switch_from_signal_values(
            selected.loc[indices, "y_pred_raw"].to_numpy(),
            threshold=threshold,
            condition=condition,
            switch_time=switch_time,
            apply_min_island_length=apply_min_island_length,
        )
        selected.loc[indices, "model_switch_raw"] = raw
        selected.loc[indices, "model_switch"] = processed

    selected["method"] = method_name
    selected["split"] = "test"
    selected["model_switch_raw"] = selected["model_switch_raw"].astype(np.int8)
    selected["model_switch"] = selected["model_switch"].astype(np.int8)
    selected["threshold"] = float(threshold)
    selected["switch_time"] = int(switch_time)
    return selected.rename(
        columns={
            "target_time": "Time",
            "y_true_raw": "Signal_true",
            "y_pred_raw": "Signal_predicted",
            "horizon_step": "selected_horizon_step",
        }
    )[
        [
            "method",
            "event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "split",
            "Time",
            "Signal_true",
            "Signal_predicted",
            "selected_horizon_step",
            "perfect_switch",
            "model_switch_raw",
            "model_switch",
            "threshold",
            "switch_time",
            "window_id",
        ]
    ]


def _median_sample_interval_seconds(times: pd.Series) -> float:
    differences = pd.to_datetime(times).sort_values().diff().dt.total_seconds().dropna()
    valid = differences.loc[differences.gt(0)]
    return float(valid.median()) if not valid.empty else 1.0


def compute_model_vs_perfect_metrics(
    comparison_timeseries: pd.DataFrame,
    *,
    by_event: bool,
) -> pd.DataFrame:
    """Compute sample-level model switch metrics against Perfect Switch."""

    group_columns = ["method", "event_id"] if by_event else ["method"]
    rows: list[dict[str, object]] = []
    group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, frame in comparison_timeseries.groupby(group_key, sort=False):
        key_values = (key,) if isinstance(key, str) else key
        interval_seconds = _median_sample_interval_seconds(frame["Time"])
        metrics = compute_switch_metrics(
            frame["perfect_switch"].tolist(),
            frame["model_switch"].tolist(),
            sample_interval_seconds=interval_seconds,
        )
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "split": "test",
                "num_points": len(frame),
                "num_perfect_positive": int(frame["perfect_switch"].sum()),
                "num_model_positive_raw": int(frame["model_switch_raw"].sum()),
                "num_model_positive": int(frame["model_switch"].sum()),
                "points_changed_by_min_island": int(
                    frame["model_switch_raw"].ne(frame["model_switch"]).sum()
                ),
                "sample_interval_seconds": interval_seconds,
                **metrics.__dict__,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
