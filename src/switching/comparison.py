"""Model-agnostic alignment and evaluation against the Perfect Switch."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.switch_metrics import compute_switch_metrics
from src.switching.conversion import (
    compute_switch_from_signal_values,
    enforce_switch_time,
    ensure_min_island_length,
)


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
    optional_target_columns = [
        column
        for column in (
            "outage_mask",
            "perfect_switch_raw",
            "perfect_switch_min_time",
            "perfect_switch_adjusted",
        )
        if column in perfect_window_targets
    ]
    target_columns = perfect_window_targets[
        [
            "window_id",
            "event_id",
            "target_step",
            "target_time",
            "y_true_raw",
            "perfect_switch",
            *optional_target_columns,
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


def select_horizon_threshold_count_decisions(
    aligned_predictions: pd.DataFrame,
    window_metadata: pd.DataFrame,
    perfect_timeseries: pd.DataFrame,
    *,
    threshold: float,
    condition: str,
    required_points_above_threshold: int,
) -> pd.DataFrame:
    """Create one decision-time row per window from its full predicted horizon."""

    if (
        not isinstance(required_points_above_threshold, (int, np.integer))
        or required_points_above_threshold < 1
    ):
        raise ValueError("required_points_above_threshold must be a positive integer.")
    _require_columns(
        aligned_predictions,
        [
            "window_id",
            "event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "horizon_step",
            "y_pred_raw",
        ],
        "aligned_predictions",
    )
    _require_columns(
        window_metadata,
        [
            "window_id",
            "event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "split",
            "input_end_time",
        ],
        "window_metadata",
    )
    _require_columns(
        perfect_timeseries,
        ["event_id", "Time", "Signal_true", "perfect_switch"],
        "perfect_timeseries",
    )

    predictions = aligned_predictions.copy()
    horizon_switch, _ = compute_switch_from_signal_values(
        predictions["y_pred_raw"].to_numpy(),
        threshold=threshold,
        condition=condition,
        switch_time=1,
        apply_min_island_length=False,
    )
    predictions["horizon_point_above_threshold"] = horizon_switch
    grouped = predictions.groupby(["window_id", "event_id"], sort=False)
    decisions = grouped.agg(
        prediction_horizon_points=("horizon_step", "count"),
        num_horizon_points_above_threshold=("horizon_point_above_threshold", "sum"),
        Signal_predicted=("y_pred_raw", "mean"),
        min_predicted_signal=("y_pred_raw", "min"),
        max_predicted_signal=("y_pred_raw", "max"),
        first_horizon_prediction=("y_pred_raw", "first"),
        last_horizon_prediction=("y_pred_raw", "last"),
    ).reset_index()
    too_short = decisions.loc[
        decisions["prediction_horizon_points"].lt(required_points_above_threshold)
    ]
    if not too_short.empty:
        examples = ", ".join(too_short["window_id"].head(5).astype(str))
        raise ValueError(
            "Some windows have fewer predicted horizon points than "
            f"required_points_above_threshold={required_points_above_threshold}. "
            f"Examples: {examples}"
        )
    decisions["model_switch_raw"] = decisions[
        "num_horizon_points_above_threshold"
    ].ge(required_points_above_threshold).astype(np.int8)
    decisions["required_points_above_threshold"] = int(
        required_points_above_threshold
    )

    metadata_columns = [
        "window_id",
        "event_id",
        "dataset_id",
        "dataset_name",
        "quality_flag",
        "split",
        "input_end_time",
    ]
    metadata = window_metadata[metadata_columns].drop_duplicates(
        subset=["window_id", "event_id"]
    )
    decisions = decisions.merge(
        metadata,
        on=["window_id", "event_id"],
        how="left",
        validate="one_to_one",
    )
    if decisions["input_end_time"].isna().any():
        raise ValueError("Some forecast windows are missing input_end_time metadata.")
    decisions["Time"] = pd.to_datetime(decisions["input_end_time"], errors="raise")

    optional_reference_columns = [
        column
        for column in (
            "outage_mask",
            "perfect_switch_raw",
            "perfect_switch_min_time",
            "perfect_switch_adjusted",
        )
        if column in perfect_timeseries
    ]
    reference = perfect_timeseries[
        ["event_id", "Time", "Signal_true", "perfect_switch", *optional_reference_columns]
    ].copy()
    reference["Time"] = pd.to_datetime(reference["Time"], errors="raise")
    decisions = decisions.merge(
        reference,
        on=["event_id", "Time"],
        how="left",
        validate="many_to_one",
    )
    if decisions["perfect_switch"].isna().any() or decisions["Signal_true"].isna().any():
        missing = decisions.loc[
            decisions["perfect_switch"].isna() | decisions["Signal_true"].isna(),
            ["window_id", "event_id", "Time"],
        ].head(5)
        raise ValueError(
            "Some decision times could not be aligned to Perfect Switch. "
            f"Examples: {missing.to_dict(orient='records')}"
        )
    decisions["decision_rule"] = "horizon_threshold_count"
    return decisions.sort_values(["event_id", "Time"], kind="stable").reset_index(
        drop=True
    )


def _complete_explicit_switch_versions(
    selected: pd.DataFrame,
    *,
    threshold: float,
    condition: str,
    switch_time: int,
    apply_min_island_length: bool,
) -> pd.DataFrame:
    """Add explicit old-repository switch versions while preserving aliases."""

    output = selected.copy()
    outage_mask, _ = compute_switch_from_signal_values(
        output["Signal_true"].to_numpy(dtype=float),
        threshold=threshold,
        condition=condition,
        switch_time=1,
        apply_min_island_length=False,
    )
    output["outage_mask"] = output.get("outage_mask", pd.Series(outage_mask)).fillna(
        pd.Series(outage_mask)
    ).astype(np.int8)
    if "perfect_switch_min_time" not in output:
        output["perfect_switch_min_time"] = output["perfect_switch"]
    if "perfect_switch_raw" not in output:
        output["perfect_switch_raw"] = output["perfect_switch_min_time"]

    output["model_switch_min_time"] = np.int8(0)
    output["model_switch_adjusted"] = np.int8(0)
    if "perfect_switch_adjusted" not in output:
        output["perfect_switch_adjusted"] = np.int8(0)
    else:
        output["perfect_switch_adjusted"] = output["perfect_switch_adjusted"].fillna(0)

    for _, indices in output.groupby("event_id", sort=False).indices.items():
        raw = output.loc[indices, "model_switch_raw"].to_numpy(dtype=np.int8)
        if apply_min_island_length:
            model_min_time = ensure_min_island_length(raw, switch_time)
        else:
            model_min_time = raw.copy()
        outage = output.loc[indices, "outage_mask"].to_numpy(dtype=np.int8)
        model_adjusted = enforce_switch_time(model_min_time, outage, switch_time)
        output.loc[indices, "model_switch_min_time"] = model_min_time
        output.loc[indices, "model_switch_adjusted"] = model_adjusted

        perfect_min_time = output.loc[
            indices, "perfect_switch_min_time"
        ].to_numpy(dtype=np.int8)
        if output.loc[indices, "perfect_switch_adjusted"].isna().any() or not output.loc[
            indices, "perfect_switch_adjusted"
        ].to_numpy(dtype=np.int8).any():
            perfect_adjusted = enforce_switch_time(perfect_min_time, outage, switch_time)
            output.loc[indices, "perfect_switch_adjusted"] = perfect_adjusted

    output["model_switch"] = output["model_switch_min_time"].astype(np.int8)
    output["perfect_switch"] = output["perfect_switch_min_time"].astype(np.int8)
    for column in (
        "outage_mask",
        "perfect_switch_raw",
        "perfect_switch_min_time",
        "perfect_switch_adjusted",
        "perfect_switch",
        "model_switch_raw",
        "model_switch_min_time",
        "model_switch_adjusted",
        "model_switch",
    ):
        output[column] = output[column].astype(np.int8)
    return output


def build_model_switch_timeseries(
    predictions: pd.DataFrame,
    perfect_window_targets: pd.DataFrame,
    *,
    method_name: str,
    threshold: float,
    condition: str,
    switch_time: int,
    apply_min_island_length: bool,
    prediction_aggregation: str = "latest_available",
    required_points_above_threshold: int | None = None,
    window_metadata: pd.DataFrame | None = None,
    perfect_timeseries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one operational pointwise switch series from overlapping forecasts."""

    aligned = align_forecast_predictions_to_targets(predictions, perfect_window_targets)
    if prediction_aggregation == "latest_available":
        selected = select_latest_available_forecasts(aligned).rename(
            columns={
                "target_time": "Time",
                "y_true_raw": "Signal_true",
                "y_pred_raw": "Signal_predicted",
                "horizon_step": "selected_horizon_step",
            }
        )
        selected["model_switch_raw"] = np.int8(0)
        selected["decision_rule"] = "latest_available"
        selected["prediction_horizon_points"] = 1
        selected["num_horizon_points_above_threshold"] = np.nan
        selected["required_points_above_threshold"] = np.nan
    elif prediction_aggregation == "horizon_threshold_count":
        if window_metadata is None or perfect_timeseries is None:
            raise ValueError(
                "horizon_threshold_count requires window_metadata and "
                "perfect_timeseries."
            )
        selected = select_horizon_threshold_count_decisions(
            aligned,
            window_metadata,
            perfect_timeseries,
            threshold=threshold,
            condition=condition,
            required_points_above_threshold=(
                int(required_points_above_threshold)
                if required_points_above_threshold is not None
                else len(aligned["horizon_step"].unique())
            ),
        )
    else:
        raise ValueError(
            "prediction_aggregation must be one of: latest_available, "
            "horizon_threshold_count."
        )
    selected["model_switch"] = np.int8(0)
    for _, indices in selected.groupby("event_id", sort=False).indices.items():
        if prediction_aggregation == "latest_available":
            raw, processed = compute_switch_from_signal_values(
                selected.loc[indices, "Signal_predicted"].to_numpy(),
                threshold=threshold,
                condition=condition,
                switch_time=switch_time,
                apply_min_island_length=apply_min_island_length,
            )
            selected.loc[indices, "model_switch_raw"] = raw
        else:
            raw = selected.loc[indices, "model_switch_raw"].to_numpy(dtype=np.int8)
            if apply_min_island_length:
                processed = ensure_min_island_length(raw, switch_time)
            else:
                processed = raw.copy()
        selected.loc[indices, "model_switch"] = processed

    selected["model_switch_min_time"] = selected["model_switch"]
    selected = _complete_explicit_switch_versions(
        selected,
        threshold=threshold,
        condition=condition,
        switch_time=switch_time,
        apply_min_island_length=apply_min_island_length,
    )
    selected["method"] = method_name
    selected["split"] = "test"
    selected["model_switch_raw"] = selected["model_switch_raw"].astype(np.int8)
    selected["model_switch"] = selected["model_switch"].astype(np.int8)
    selected["threshold"] = float(threshold)
    selected["switch_time"] = int(switch_time)
    optional_columns = [
        column
        for column in (
            "min_predicted_signal",
            "max_predicted_signal",
            "first_horizon_prediction",
            "last_horizon_prediction",
        )
        if column in selected
    ]
    return selected[
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
            "decision_rule",
            "prediction_horizon_points",
            "num_horizon_points_above_threshold",
            "required_points_above_threshold",
            *optional_columns,
            "outage_mask",
            "perfect_switch_raw",
            "perfect_switch_min_time",
            "perfect_switch_adjusted",
            "perfect_switch",
            "model_switch_raw",
            "model_switch_min_time",
            "model_switch_adjusted",
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
        if "decision_rule" in frame:
            row["decision_rule"] = frame["decision_rule"].iloc[0]
        if "required_points_above_threshold" in frame:
            value = frame["required_points_above_threshold"].dropna()
            row["required_points_above_threshold"] = (
                int(value.iloc[0]) if not value.empty else np.nan
            )
        if "prediction_horizon_points" in frame:
            row["prediction_horizon_points"] = int(
                frame["prediction_horizon_points"].max()
            )
        rows.append(row)
    return pd.DataFrame(rows)
