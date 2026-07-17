"""Shared small-gap imputation utilities for task dataset builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SmallGapImputationConfig:
    """Configuration for conservative time-linear small-gap imputation."""

    enabled: bool = False
    expected_seconds: float = 30.0
    max_gap_seconds: float = 90.0
    max_missing_run: int = 2
    method: str = "linear_time"
    marker_column: str = "is_imputed"


@dataclass(frozen=True)
class SmallGapImputationResult:
    """Dataframe plus per-group imputation summary."""

    dataframe: pd.DataFrame
    summary: pd.DataFrame


def small_gap_imputation_config_from_mapping(
    config: Mapping[str, Any] | None,
    *,
    enabled_default: bool = False,
) -> SmallGapImputationConfig:
    """Return a validated small-gap imputation config from YAML-like mapping."""

    values = dict(config or {})
    imputation_config = SmallGapImputationConfig(
        enabled=bool(values.get("enabled", enabled_default)),
        expected_seconds=float(values.get("expected_seconds", 30.0)),
        max_gap_seconds=float(values.get("max_gap_seconds", 90.0)),
        max_missing_run=int(values.get("max_missing_run", 2)),
        method=str(values.get("method", "linear_time")),
        marker_column=str(values.get("marker_column", "is_imputed")),
    )
    validate_small_gap_imputation_config(imputation_config)
    return imputation_config


def validate_small_gap_imputation_config(config: SmallGapImputationConfig) -> None:
    """Validate small-gap imputation parameters."""

    if config.method != "linear_time":
        raise ValueError("Only method=linear_time is supported.")
    if config.expected_seconds <= 0:
        raise ValueError("expected_seconds must be positive.")
    if config.max_gap_seconds <= config.expected_seconds:
        raise ValueError("max_gap_seconds must be larger than expected_seconds.")
    if config.max_missing_run < 1:
        raise ValueError("max_missing_run must be at least one.")


def impute_small_time_gaps(
    frame: pd.DataFrame,
    *,
    time_column: str = "Time",
    signal_column: str = "Signal",
    group_columns: Iterable[str] = ("dataset_id", "segment_id"),
    config: SmallGapImputationConfig | None = None,
) -> SmallGapImputationResult:
    """Impute short timestamp gaps using linear interpolation inside each group.

    The function is intentionally conservative: it never bridges distinct
    groups, and it only fills gaps whose inferred number of missing samples is
    not larger than ``max_missing_run``.
    """

    imputation_config = config or SmallGapImputationConfig()
    validate_small_gap_imputation_config(imputation_config)
    if frame.empty:
        return SmallGapImputationResult(
            dataframe=frame.copy(),
            summary=_empty_summary_frame(list(group_columns)),
        )
    required = {time_column, signal_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot impute small gaps; missing columns: {missing}")

    available_group_columns = [
        column for column in group_columns if column in frame.columns
    ]
    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], errors="raise")
    if imputation_config.marker_column not in ordered.columns:
        ordered[imputation_config.marker_column] = False
    else:
        ordered[imputation_config.marker_column] = ordered[
            imputation_config.marker_column
        ].fillna(False).astype(bool)

    sort_columns = [*available_group_columns, time_column]
    ordered = ordered.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if not imputation_config.enabled:
        return SmallGapImputationResult(
            dataframe=ordered,
            summary=_summarize_groups(
                ordered,
                time_column=time_column,
                group_columns=available_group_columns,
                inserted_counts={},
                imputed_gap_counts={},
                max_original_gap=_max_gap_by_group(
                    ordered,
                    time_column=time_column,
                    group_columns=available_group_columns,
                ),
                max_imputed_gap={},
                max_inserted_run={},
            ),
        )

    inserted_counts: dict[tuple[Any, ...], int] = {}
    imputed_gap_counts: dict[tuple[Any, ...], int] = {}
    max_original_gap = _max_gap_by_group(
        ordered,
        time_column=time_column,
        group_columns=available_group_columns,
    )
    max_imputed_gap: dict[tuple[Any, ...], float] = {}
    max_inserted_run: dict[tuple[Any, ...], int] = {}
    output_groups: list[pd.DataFrame] = []

    groups = (
        ordered.groupby(available_group_columns, sort=False, dropna=False)
        if available_group_columns
        else [((), ordered)]
    )
    for group_key, group in groups:
        normalized_key = _normalize_group_key(group_key)
        prepared_rows: list[dict[str, Any]] = []
        group_ordered = group.sort_values(time_column, kind="stable").reset_index(
            drop=True
        )
        for position, row in enumerate(group_ordered.to_dict(orient="records")):
            if position:
                previous = group_ordered.iloc[position - 1]
                current = group_ordered.iloc[position]
                dt_seconds = float(
                    (current[time_column] - previous[time_column]).total_seconds()
                )
                missing_count = _inferred_missing_count(
                    dt_seconds,
                    expected_seconds=imputation_config.expected_seconds,
                )
                should_impute = (
                    missing_count > 0
                    and dt_seconds <= imputation_config.max_gap_seconds
                    and missing_count <= imputation_config.max_missing_run
                )
                if should_impute:
                    inserted_counts[normalized_key] = (
                        inserted_counts.get(normalized_key, 0) + missing_count
                    )
                    imputed_gap_counts[normalized_key] = (
                        imputed_gap_counts.get(normalized_key, 0) + 1
                    )
                    max_imputed_gap[normalized_key] = max(
                        max_imputed_gap.get(normalized_key, 0.0),
                        dt_seconds,
                    )
                    max_inserted_run[normalized_key] = max(
                        max_inserted_run.get(normalized_key, 0),
                        missing_count,
                    )
                    previous_signal = float(previous[signal_column])
                    current_signal = float(current[signal_column])
                    for missing_index in range(1, missing_count + 1):
                        fraction = missing_index / float(missing_count + 1)
                        imputed_row = dict(previous)
                        imputed_row[time_column] = previous[
                            time_column
                        ] + pd.to_timedelta(
                            imputation_config.expected_seconds * missing_index,
                            unit="s",
                        )
                        imputed_row[signal_column] = (
                            previous_signal
                            + fraction * (current_signal - previous_signal)
                        )
                        imputed_row[imputation_config.marker_column] = True
                        if "Signal_original" in imputed_row:
                            imputed_row["Signal_original"] = np.nan
                        if "row_id" in imputed_row:
                            imputed_row["row_id"] = np.nan
                        prepared_rows.append(imputed_row)
            prepared_rows.append(row)
        output_groups.append(pd.DataFrame(prepared_rows))

    output = pd.concat(output_groups, ignore_index=True)
    output = output.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if "dt_seconds" in output.columns:
        output["dt_seconds"] = (
            output.groupby(available_group_columns, sort=False)[time_column]
            .diff()
            .dt.total_seconds()
            if available_group_columns
            else output[time_column].diff().dt.total_seconds()
        )
    summary = _summarize_groups(
        output,
        time_column=time_column,
        group_columns=available_group_columns,
        inserted_counts=inserted_counts,
        imputed_gap_counts=imputed_gap_counts,
        max_original_gap=max_original_gap,
        max_imputed_gap=max_imputed_gap,
        max_inserted_run=max_inserted_run,
    )
    return SmallGapImputationResult(dataframe=output, summary=summary)


def _inferred_missing_count(dt_seconds: float, *, expected_seconds: float) -> int:
    if not np.isfinite(dt_seconds) or dt_seconds <= expected_seconds * 1.5:
        return 0
    return max(0, int(round(dt_seconds / expected_seconds)) - 1)


def _normalize_group_key(group_key: Any) -> tuple[Any, ...]:
    if isinstance(group_key, tuple):
        return group_key
    return (group_key,)


def _empty_summary_frame(group_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *group_columns,
            "num_original_rows",
            "num_output_rows",
            "num_inserted_rows",
            "num_imputed_gaps",
            "max_original_gap_seconds",
            "max_imputed_gap_seconds",
            "max_inserted_run",
        ]
    )


def _max_gap_by_group(
    frame: pd.DataFrame,
    *,
    time_column: str,
    group_columns: list[str],
) -> dict[tuple[Any, ...], float]:
    max_gaps: dict[tuple[Any, ...], float] = {}
    groups = (
        frame.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [((), frame)]
    )
    for group_key, group in groups:
        normalized_key = _normalize_group_key(group_key)
        deltas = (
            pd.to_datetime(group[time_column], errors="raise")
            .diff()
            .dt.total_seconds()
            .dropna()
        )
        max_gaps[normalized_key] = float(deltas.max()) if len(deltas) else np.nan
    return max_gaps


def _summarize_groups(
    frame: pd.DataFrame,
    *,
    time_column: str,
    group_columns: list[str],
    inserted_counts: Mapping[tuple[Any, ...], int],
    imputed_gap_counts: Mapping[tuple[Any, ...], int],
    max_original_gap: Mapping[tuple[Any, ...], float],
    max_imputed_gap: Mapping[tuple[Any, ...], float],
    max_inserted_run: Mapping[tuple[Any, ...], int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = (
        frame.groupby(group_columns, sort=False, dropna=False)
        if group_columns
        else [((), frame)]
    )
    for group_key, group in groups:
        normalized_key = _normalize_group_key(group_key)
        row = {
            "num_output_rows": int(len(group)),
            "num_inserted_rows": int(inserted_counts.get(normalized_key, 0)),
            "num_imputed_gaps": int(imputed_gap_counts.get(normalized_key, 0)),
            "max_original_gap_seconds": float(
                max_original_gap.get(normalized_key, np.nan)
            ),
            "max_imputed_gap_seconds": float(
                max_imputed_gap.get(normalized_key, np.nan)
            ),
            "max_inserted_run": int(max_inserted_run.get(normalized_key, 0)),
        }
        row["num_original_rows"] = row["num_output_rows"] - row["num_inserted_rows"]
        for index, column in enumerate(group_columns):
            row[column] = normalized_key[index]
        rows.append(row)
    columns = [
        *group_columns,
        "num_original_rows",
        "num_output_rows",
        "num_inserted_rows",
        "num_imputed_gaps",
        "max_original_gap_seconds",
        "max_imputed_gap_seconds",
        "max_inserted_run",
    ]
    return pd.DataFrame(rows, columns=columns)
