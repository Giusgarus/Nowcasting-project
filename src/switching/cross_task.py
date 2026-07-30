"""Cross-task comparison helpers for saved switch-evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.switch_metrics import compute_switch_metrics


@dataclass(frozen=True)
class SwitchComparisonSource:
    """One saved model-vs-Perfect switch comparison to include."""

    task_name: str
    selection_id: str
    comparison_id: str
    method_id: str
    results_path: Path
    predictions_path: Path
    method_name: str


def parse_switch_eval_task(results_path: str | Path) -> tuple[str, str]:
    """Return ``(task_name, selection_id)`` from a switch-eval results path."""

    path = Path(results_path)
    parts = path.parts
    marker = ("results", "comparisons", "switch_eval")
    for index in range(0, len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            task_index = index + len(marker)
            if task_index + 1 >= len(parts):
                break
            return parts[task_index], parts[task_index + 1]
    raise ValueError(f"Cannot parse task and selection from switch-eval path: {path}")


def median_sample_interval_seconds(times: pd.Series) -> float:
    """Return the median positive interval in seconds for sorted timestamps."""

    ordered = pd.to_datetime(times, errors="raise").sort_values(kind="stable")
    deltas = ordered.diff().dt.total_seconds().dropna()
    positive = deltas.loc[deltas > 0]
    if positive.empty:
        return 1.0
    return float(positive.median())


def load_reference_grid(path: Path) -> pd.DataFrame:
    """Load and normalize a Perfect Switch timeseries used as common grid."""

    frame = pd.read_parquet(path)
    if "Signal" in frame.columns and "Signal_true" not in frame.columns:
        frame = frame.rename(columns={"Signal": "Signal_true"})
    required = ["Time", "Signal_true", "perfect_switch"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Reference grid is missing required columns: {missing}")
    optional = [
        column
        for column in (
            "event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "outage_mask",
            "perfect_switch_raw",
            "perfect_switch_min_time",
        )
        if column in frame.columns
    ]
    output = frame[[*optional, *required]].copy()
    output["Time"] = pd.to_datetime(output["Time"], errors="raise")
    return (
        output.sort_values("Time", kind="stable")
        .drop_duplicates("Time", keep="first")
        .reset_index(drop=True)
    )


def read_method_frame(source: SwitchComparisonSource) -> pd.DataFrame:
    """Load the comparable columns from one saved switch comparison."""

    frame = pd.read_parquet(source.predictions_path)
    required = ["Time", "model_switch", "model_switch_raw"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source.predictions_path} is missing required columns: {missing}"
        )
    output = frame[required].copy()
    output["Time"] = pd.to_datetime(output["Time"], errors="raise")
    duplicates = output["Time"].duplicated(keep=False)
    if duplicates.any():
        examples = output.loc[duplicates, "Time"].head(5).tolist()
        raise ValueError(
            f"{source.predictions_path} has duplicate decision timestamps: {examples}"
        )
    return output.sort_values("Time", kind="stable").reset_index(drop=True)


def discover_switch_sources(
    comparisons_index: pd.DataFrame,
    *,
    project_root: Path,
    include_sources: list[dict[str, str]],
) -> list[SwitchComparisonSource]:
    """Discover completed switch comparisons from the central comparison index."""

    if comparisons_index.empty:
        return []
    required = [
        "comparison_id",
        "method_id",
        "comparison_type",
        "selection_id",
        "results_path",
        "status",
    ]
    missing = sorted(set(required) - set(comparisons_index.columns))
    if missing:
        raise ValueError(f"Comparison index is missing required columns: {missing}")

    include_pairs = {
        (str(item["task_name"]), str(item["selection_id"])) for item in include_sources
    }
    sources: list[SwitchComparisonSource] = []
    rows = comparisons_index.loc[
        comparisons_index["comparison_type"].eq("switch_eval")
        & comparisons_index["status"].eq("complete")
    ]
    for row in rows.itertuples(index=False):
        task_name, selection_id = parse_switch_eval_task(row.results_path)
        if (task_name, selection_id) not in include_pairs:
            continue
        results_path = project_root / str(row.results_path)
        predictions_path = results_path / "predictions" / "model_vs_perfect_switch.parquet"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing switch comparison parquet: {predictions_path}")
        preview = pd.read_parquet(predictions_path, columns=["method"])
        method_name = (
            str(preview["method"].dropna().iloc[0])
            if "method" in preview and not preview["method"].dropna().empty
            else str(row.method_id)
        )
        sources.append(
            SwitchComparisonSource(
                task_name=task_name,
                selection_id=selection_id,
                comparison_id=str(row.comparison_id),
                method_id=str(row.method_id),
                results_path=results_path,
                predictions_path=predictions_path,
                method_name=method_name,
            )
        )
    return sorted(sources, key=lambda item: (item.task_name, item.method_name))


def metric_row(
    *,
    source: SwitchComparisonSource,
    denominator: str,
    reference: pd.Series,
    predicted: pd.Series,
    raw_predicted: pd.Series,
    sample_interval_seconds: float,
    num_reference_points: int,
    num_native_decision_points: int,
    num_aligned_native_points: int,
) -> dict[str, Any]:
    """Build one cross-task metric row."""

    metrics = compute_switch_metrics(
        reference.astype("int8").tolist(),
        predicted.astype("int8").tolist(),
        sample_interval_seconds=sample_interval_seconds,
    )
    row = {
        "denominator": denominator,
        "task_name": source.task_name,
        "selection_id": source.selection_id,
        "comparison_id": source.comparison_id,
        "method_id": source.method_id,
        "method_name": source.method_name,
        "num_points": int(len(reference)),
        "num_reference_points": int(num_reference_points),
        "num_native_decision_points": int(num_native_decision_points),
        "num_aligned_native_points": int(num_aligned_native_points),
        "coverage_pct": (
            100.0 * num_aligned_native_points / num_reference_points
            if num_reference_points
            else np.nan
        ),
        "num_perfect_positive": int(reference.sum()),
        "num_model_positive_raw": int(raw_predicted.sum()),
        "num_model_positive": int(predicted.sum()),
        "points_changed_by_min_island": int(raw_predicted.ne(predicted).sum()),
        "sample_interval_seconds": sample_interval_seconds,
    }
    row.update(metrics.__dict__)
    return row


def build_reference_grid_metrics(
    *,
    reference_grid: pd.DataFrame,
    sources: list[SwitchComparisonSource],
    missing_model_switch: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every method on the full common reference grid."""

    sample_interval = median_sample_interval_seconds(reference_grid["Time"])
    reference = reference_grid[["Time", "perfect_switch"]].copy()
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for source in sources:
        method_frame = read_method_frame(source)
        aligned = reference.merge(
            method_frame,
            on="Time",
            how="left",
            validate="one_to_one",
        )
        has_native = aligned["model_switch"].notna()
        aligned["model_switch"] = aligned["model_switch"].fillna(missing_model_switch)
        aligned["model_switch_raw"] = aligned["model_switch_raw"].fillna(
            missing_model_switch
        )
        perfect_missing_due_to_no_decision = int(
            aligned.loc[~has_native, "perfect_switch"].sum()
        )
        rows.append(
            metric_row(
                source=source,
                denominator="reference_grid_missing_as_zero",
                reference=aligned["perfect_switch"],
                predicted=aligned["model_switch"],
                raw_predicted=aligned["model_switch_raw"],
                sample_interval_seconds=sample_interval,
                num_reference_points=len(reference),
                num_native_decision_points=len(method_frame),
                num_aligned_native_points=int(has_native.sum()),
            )
        )
        coverage_rows.append(
            {
                "task_name": source.task_name,
                "selection_id": source.selection_id,
                "comparison_id": source.comparison_id,
                "method_id": source.method_id,
                "method_name": source.method_name,
                "num_reference_points": int(len(reference)),
                "num_native_decision_points": int(len(method_frame)),
                "num_aligned_native_points": int(has_native.sum()),
                "coverage_pct": float(100.0 * has_native.mean()),
                "perfect_positive_points_without_native_decision": (
                    perfect_missing_due_to_no_decision
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def build_strict_intersection_metrics(
    *,
    reference_grid: pd.DataFrame,
    sources: list[SwitchComparisonSource],
) -> pd.DataFrame:
    """Evaluate every method on timestamps where every method has a decision."""

    if not sources:
        return pd.DataFrame()
    method_frames = {source.comparison_id: read_method_frame(source) for source in sources}
    common_times = set(pd.to_datetime(reference_grid["Time"], errors="raise"))
    for frame in method_frames.values():
        common_times &= set(frame["Time"])
    if not common_times:
        return pd.DataFrame()
    common_index = pd.Index(sorted(common_times), name="Time")
    reference = (
        reference_grid[["Time", "perfect_switch"]]
        .drop_duplicates("Time", keep="first")
        .set_index("Time")
        .loc[common_index]
        .reset_index()
    )
    sample_interval = median_sample_interval_seconds(reference["Time"])
    rows: list[dict[str, Any]] = []
    for source in sources:
        method_frame = (
            method_frames[source.comparison_id]
            .set_index("Time")
            .loc[common_index]
            .reset_index()
        )
        rows.append(
            metric_row(
                source=source,
                denominator="strict_common_timestamp_intersection",
                reference=reference["perfect_switch"],
                predicted=method_frame["model_switch"],
                raw_predicted=method_frame["model_switch_raw"],
                sample_interval_seconds=sample_interval,
                num_reference_points=len(reference_grid),
                num_native_decision_points=len(method_frames[source.comparison_id]),
                num_aligned_native_points=len(reference),
            )
        )
    return pd.DataFrame(rows)
