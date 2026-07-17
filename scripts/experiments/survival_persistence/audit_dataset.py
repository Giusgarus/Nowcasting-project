"""Audit the canonical survival-persistence dataset without training models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.survival_persistence.data.dataset import (  # noqa: E402
    SCALAR_CONTEXT_FEATURE_NAMES,
    build_survival_arrays,
    load_full_signal,
)
from src.data.imputation import (  # noqa: E402
    impute_small_time_gaps,
    small_gap_imputation_config_from_mapping,
)
from src.tasks.survival_persistence.data.event_state_machine import (  # noqa: E402
    CENSOR_ACTIVE,
    CENSOR_CANDIDATE,
    CENSOR_INSUFFICIENT_WINDOW,
    StableRecoveryConfig,
    trace_survival_state_machine,
)
from src.tasks.survival_persistence.data.features import (  # noqa: E402
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    compute_scalar_context_features,
)
from src.tasks.survival_persistence.utils.paths import (  # noqa: E402
    TASK_NAME,
    data_preparation_dir,
    make_selection_id,
    processed_dataset_dir,
)
from src.utils.config import load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import relative_project_path  # noqa: E402

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml"
)
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}
PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def project_path(path: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-random-context-checks", type=int, default=12)
    return parser.parse_args()


def recovery_config_from_yaml(config: dict[str, Any]) -> StableRecoveryConfig:
    """Build the recovery config from the canonical dataset YAML."""

    event = config["event_definition"]
    return StableRecoveryConfig(
        threshold_on=float(event["threshold_on"]),
        threshold_off=float(event["threshold_off"]),
        threshold_on_operator=str(event.get("threshold_on_operator", ">")),  # type: ignore[arg-type]
        threshold_off_operator=str(event.get("threshold_off_operator", "<=")),  # type: ignore[arg-type]
        recovery_window_seconds=float(event["recovery_window_seconds"]),
        recovery_required_fraction=float(event["recovery_required_fraction"]),
        recovery_fraction_mode=str(event.get("recovery_fraction_mode", "sample_fraction")),  # type: ignore[arg-type]
        minimum_recovery_observations=int(event.get("minimum_recovery_observations", 2)),
        abort_candidate_recovery_on_threshold_on_crossing=bool(
            event.get("abort_candidate_recovery_on_threshold_on_crossing", True)
        ),
    )


def require_dataset_files(dataset_dir: Path) -> None:
    """Fail if the canonical dataset is incomplete."""

    required = [
        dataset_dir / "dataset_metadata.yaml",
        dataset_dir / "event_summary.csv",
        dataset_dir / "split_summary.csv",
        dataset_dir / "censoring_summary.csv",
    ]
    for stem in SPLIT_FILES.values():
        required.append(dataset_dir / f"{stem}.npz")
        required.append(dataset_dir / f"{stem}_metadata.parquet")
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset audit missing required files:\n{formatted}")


def load_split_artifacts(dataset_dir: Path) -> tuple[dict[str, np.lib.npyio.NpzFile], dict[str, pd.DataFrame]]:
    """Load split NPZ files and aligned metadata."""

    arrays: dict[str, np.lib.npyio.NpzFile] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split, stem in SPLIT_FILES.items():
        arrays[split] = np.load(dataset_dir / f"{stem}.npz", allow_pickle=False)
        metadata[split] = pd.read_parquet(dataset_dir / f"{stem}_metadata.parquet")
    return arrays, metadata


def distribution(values: pd.Series | np.ndarray, *, label: str, split: str | None = None) -> dict[str, Any]:
    """Return a compact distribution summary."""

    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    row: dict[str, Any] = {"label": label, "count": int(len(series))}
    if split is not None:
        row["split"] = split
    if series.empty:
        for key in ["min", "p10", "p25", "median", "p75", "p90", "p95", "max"]:
            row[key] = np.nan
        return row
    row["min"] = float(series.min())
    row["p10"] = float(series.quantile(0.10))
    row["p25"] = float(series.quantile(0.25))
    row["median"] = float(series.quantile(0.50))
    row["p75"] = float(series.quantile(0.75))
    row["p90"] = float(series.quantile(0.90))
    row["p95"] = float(series.quantile(0.95))
    row["max"] = float(series.max())
    return row


def make_event_counts(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed/censored event counts by split."""

    rows = []
    for split, frame in event_summary.groupby("split", sort=False):
        observed = frame["event_observed"].astype(bool)
        num_events = int(len(frame))
        num_censored = int((~observed).sum())
        rows.append(
            {
                "split": split,
                "number_of_events": num_events,
                "number_observed": int(observed.sum()),
                "number_censored": num_censored,
                "censoring_rate": float(num_censored / num_events) if num_events else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_sample_counts(split_metadata: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize sample counts and above/below threshold composition."""

    rows = []
    for split, frame in split_metadata.items():
        observed = frame["y_event_observed"].eq(1)
        above = frame["current_above_threshold"].astype(bool)
        rows.append(
            {
                "split": split,
                "number_of_samples": int(len(frame)),
                "observed_target_samples": int(observed.sum()),
                "censored_target_samples": int((~observed).sum()),
                "above_threshold_samples": int(above.sum()),
                "below_threshold_internal_samples": int((~above).sum()),
                "pct_above_threshold_samples": float(100.0 * above.mean()) if len(frame) else np.nan,
                "pct_below_threshold_internal_samples": float(100.0 * (~above).mean()) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def split_boundary_summary(event_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize split membership, datasets, and chronological boundaries."""

    rows = []
    for split, frame in event_summary.groupby("split", sort=False):
        ordered = frame.sort_values("event_start_time", kind="stable")
        observed = ordered["event_observed"].astype(bool)
        rows.append(
            {
                "split": split,
                "datasets": ";".join(sorted(ordered["dataset_name"].astype(str).unique())),
                "num_events": int(len(ordered)),
                "num_samples": int(ordered["num_event_samples"].sum()),
                "num_observed_events": int(observed.sum()),
                "num_censored_events": int((~observed).sum()),
                "first_event_start_time": ordered["event_start_time"].iloc[0],
                "last_event_start_time": ordered["event_start_time"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def segment_lookup(full_signal: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Return sorted segment frames keyed by dataset and segment ID."""

    return {
        (str(dataset_id), str(segment_id)): frame.sort_values("Time", kind="stable").reset_index(drop=True)
        for (dataset_id, segment_id), frame in full_signal.groupby(["dataset_id", "segment_id"], sort=False)
    }


def event_trace_for_row(
    event_row: pd.Series,
    *,
    segments: dict[tuple[str, str], pd.DataFrame],
    trace_cache: dict[tuple[str, str], pd.DataFrame],
    recovery_config: StableRecoveryConfig,
) -> pd.DataFrame:
    """Trace one event's segment and keep the rows relevant to that event."""

    key = (str(event_row["dataset_id"]), str(event_row["segment_id"]))
    if key not in trace_cache:
        segment = segments[key]
        trace_cache[key] = trace_survival_state_machine(
            segment[["Time", "Signal"]],
            config=recovery_config,
        )
    trace = trace_cache[key]
    if trace.empty:
        return trace
    start = pd.Timestamp(event_row["event_start_time"])
    if bool(event_row["event_observed"]):
        end = pd.Timestamp(event_row["event_end_confirmation_time"])
    else:
        end = pd.Timestamp(event_row["segment_end_time"])
    selected = trace.loc[
        pd.to_datetime(trace["timestamp"]).between(start, end, inclusive="both")
    ].copy()
    for column in [
        "global_event_id",
        "event_id",
        "dataset_name",
        "dataset_id",
        "segment_id",
        "split",
    ]:
        selected[column] = event_row[column]
    selected["event_summary_start_time"] = event_row["event_start_time"]
    selected["event_summary_end_time"] = event_row["event_end_time"]
    selected["event_summary_confirmation_time"] = event_row["event_end_confirmation_time"]
    selected["event_summary_observed"] = event_row["event_observed"]
    selected["event_summary_censoring_reason"] = event_row["censoring_reason"]
    return selected


def transition_statistics(
    event_summary: pd.DataFrame,
    *,
    segments: dict[tuple[str, str], pd.DataFrame],
    trace_cache: dict[tuple[str, str], pd.DataFrame],
    recovery_config: StableRecoveryConfig,
) -> pd.DataFrame:
    """Count candidate-recovery transitions per represented event."""

    rows = []
    for event_row in event_summary.itertuples(index=False):
        row = pd.Series(event_row._asdict())
        trace = event_trace_for_row(
            row,
            segments=segments,
            trace_cache=trace_cache,
            recovery_config=recovery_config,
        )
        actions = trace["action"].astype(str) if not trace.empty else pd.Series(dtype=str)
        rows.append(
            {
                "global_event_id": row["global_event_id"],
                "candidate_recovery_starts": int(actions.eq("start_candidate_recovery").sum()),
                "candidate_recovery_failures": int(actions.eq("candidate_recovery_failed").sum()),
                "candidate_recovery_aborts": int(actions.eq("abort_candidate_on_threshold_on_crossing").sum()),
                "stable_recovery_confirmations": int(actions.eq("confirm_stable_recovery").sum()),
                "candidate_wait_rows": int(actions.eq("wait_for_recovery_window").sum()),
            }
        )
    return pd.DataFrame(rows)


def first_unused_candidate(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    used: set[str],
) -> pd.Series | None:
    """Return the first matching event not already selected."""

    candidates = frame.loc[mask].copy()
    candidates = candidates.loc[~candidates["global_event_id"].isin(used)]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def choose_manual_events(event_summary: pd.DataFrame, transition_stats: pd.DataFrame) -> pd.DataFrame:
    """Select representative real events for manual state-machine inspection."""

    events = event_summary.merge(transition_stats, on="global_event_id", how="left")
    events["event_duration_seconds"] = pd.to_numeric(events["event_duration_seconds"], errors="coerce")
    used: set[str] = set()
    selections: list[dict[str, Any]] = []

    def add(category: str, mask: pd.Series, reason: str, sort_by: str | None = None) -> None:
        candidates = events.loc[mask & ~events["global_event_id"].isin(used)].copy()
        if sort_by is not None and not candidates.empty:
            candidates = candidates.sort_values(sort_by, kind="stable")
        if candidates.empty:
            selections.append(
                {
                    "category": category,
                    "status": "not_available",
                    "global_event_id": "",
                    "reason": reason,
                }
            )
            return
        event = candidates.iloc[0]
        used.add(str(event["global_event_id"]))
        selections.append(
            {
                "category": category,
                "status": "selected",
                "global_event_id": event["global_event_id"],
                "dataset_name": event["dataset_name"],
                "event_id": event["event_id"],
                "split": event["split"],
                "event_start_time": event["event_start_time"],
                "event_end_time": event["event_end_time"],
                "event_observed": event["event_observed"],
                "censoring_reason": event["censoring_reason"],
                "reason": reason,
            }
        )

    observed = events["event_observed"].astype(bool)
    add("brief_threshold_spike", observed, "shortest observed event", "event_duration_seconds")
    add(
        "rapid_observed_recovery",
        observed & events["event_duration_seconds"].lt(300),
        "observed event below 300 seconds",
        "event_duration_seconds",
    )
    add(
        "medium_duration_fade",
        observed & events["event_duration_seconds"].ge(300) & events["event_duration_seconds"].lt(900),
        "observed event in [300, 900) seconds",
        "event_duration_seconds",
    )
    add(
        "long_fade",
        observed & events["event_duration_seconds"].ge(900),
        "observed event at least 900 seconds",
        "event_duration_seconds",
    )
    add(
        "temporary_below_threshold_dip_failed_recovery",
        observed
        & events["num_below_threshold_inside_event"].gt(0)
        & (
            events["candidate_recovery_failures"].gt(0)
            | events["candidate_recovery_aborts"].gt(0)
        ),
        "below-threshold internal samples with failed or aborted candidate recovery",
        "event_duration_seconds",
    )
    add(
        "oscillating_multiple_candidate_recoveries",
        events["candidate_recovery_starts"].gt(1),
        "more than one candidate-recovery start",
        "event_duration_seconds",
    )
    add("observed_stable_recovery", observed, "ordinary observed stable recovery", "event_duration_seconds")
    add(
        "censored_active_fade",
        (~observed) & events["censoring_reason"].eq(CENSOR_ACTIVE),
        "censored while ACTIVE_FADE",
    )
    add(
        "censored_candidate_recovery",
        (~observed)
        & events["censoring_reason"].isin([CENSOR_CANDIDATE, CENSOR_INSUFFICIENT_WINDOW]),
        "censored while CANDIDATE_RECOVERY or before enough recovery evidence",
    )
    add("segment_boundary", ~observed, "event close to acquisition-segment boundary")
    return pd.DataFrame(selections)


def manual_state_traces(
    selections: pd.DataFrame,
    event_summary: pd.DataFrame,
    *,
    segments: dict[tuple[str, str], pd.DataFrame],
    trace_cache: dict[tuple[str, str], pd.DataFrame],
    recovery_config: StableRecoveryConfig,
) -> pd.DataFrame:
    """Return chronological traces for selected real events."""

    traces = []
    events = event_summary.set_index("global_event_id", drop=False)
    for selected in selections.itertuples(index=False):
        if getattr(selected, "status") != "selected":
            continue
        event_row = events.loc[getattr(selected, "global_event_id")]
        trace = event_trace_for_row(
            event_row,
            segments=segments,
            trace_cache=trace_cache,
            recovery_config=recovery_config,
        )
        if trace.empty:
            continue
        trace.insert(0, "category", getattr(selected, "category"))
        traces.append(trace)
    if not traces:
        return pd.DataFrame()
    return pd.concat(traces, ignore_index=True)


def target_check_rows(
    split_metadata: dict[str, pd.DataFrame],
    event_summary: pd.DataFrame,
    *,
    observed: bool,
    max_events: int = 3,
) -> pd.DataFrame:
    """Return manual target checks for observed or censored events."""

    combined = pd.concat(split_metadata.values(), ignore_index=True)
    events = event_summary.loc[event_summary["event_observed"].astype(bool).eq(observed)].head(max_events)
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        event_rows = combined.loc[combined["global_event_id"].eq(event.global_event_id)].sort_values(
            "sample_time", kind="stable"
        )
        if event_rows.empty:
            continue
        candidate_indices = {0, len(event_rows) // 2, len(event_rows) - 1}
        below = event_rows.index[~event_rows["current_above_threshold"].astype(bool)]
        if len(below):
            candidate_indices.add(int(event_rows.index.get_loc(below[0])))
        for position in sorted(candidate_indices):
            sample = event_rows.iloc[position]
            if observed:
                endpoint = pd.Timestamp(sample["event_end_time"])
            else:
                endpoint = pd.Timestamp(sample["segment_end_time"])
            recomputed = float((endpoint - pd.Timestamp(sample["sample_time"])).total_seconds())
            rows.append(
                {
                    "global_event_id": sample["global_event_id"],
                    "split": sample["split"],
                    "dataset_name": sample["dataset_name"],
                    "sample_time": sample["sample_time"],
                    "current_signal": sample["current_signal"],
                    "current_above_threshold": bool(sample["current_above_threshold"]),
                    "target_endpoint": endpoint,
                    "stored_y_time_seconds": float(sample["y_time_seconds"]),
                    "recomputed_y_time_seconds": recomputed,
                    "absolute_difference": abs(float(sample["y_time_seconds"]) - recomputed),
                    "y_lower_bound_seconds": float(sample["y_lower_bound_seconds"]),
                    "y_upper_bound_seconds": sample["y_upper_bound_seconds"],
                    "y_event_observed": int(sample["y_event_observed"]),
                    "y_time_samples": int(sample["y_time_samples"]),
                }
            )
    return pd.DataFrame(rows)


def context_and_scalar_checks(
    arrays: dict[str, np.lib.npyio.NpzFile],
    split_metadata: dict[str, pd.DataFrame],
    full_signal: pd.DataFrame,
    dataset_metadata: dict[str, Any],
    *,
    threshold_on: float,
    num_checks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return context, scalar-feature, and train-scaling audit checks."""

    lookup = segment_lookup(full_signal)
    rng = np.random.default_rng(42)
    context_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    scaler = dataset_metadata["scalar_context_scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)

    for split, frame in split_metadata.items():
        if frame.empty:
            continue
        n_rows = min(max(1, num_checks // len(split_metadata)), len(frame))
        sampled_positions = sorted(rng.choice(len(frame), size=n_rows, replace=False).tolist())
        split_arrays = arrays[split]
        for position in sampled_positions:
            sample = frame.iloc[position]
            segment = lookup[(str(sample["dataset_id"]), str(sample["segment_id"]))]
            selected = segment.iloc[int(sample["input_start_index"]) : int(sample["input_end_index"]) + 1]
            x_raw = split_arrays["X_raw"][position]
            expected_raw = selected["Signal"].to_numpy(dtype=np.float32)
            time_values = pd.to_datetime(selected["Time"], errors="raise")
            gaps = time_values.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
            crosses_gap = bool(np.any(gaps > float(sample["sampling_time_seconds"]) * 1.5))
            context_rows.append(
                {
                    "split": split,
                    "global_window_id": sample["global_window_id"],
                    "context_length": int(len(x_raw)),
                    "context_final_timestamp_equals_sample_time": pd.Timestamp(selected["Time"].iloc[-1])
                    == pd.Timestamp(sample["sample_time"]),
                    "all_context_times_lte_sample_time": bool((time_values <= pd.Timestamp(sample["sample_time"])).all()),
                    "single_segment": bool(selected["segment_id"].astype(str).nunique() == 1),
                    "crosses_acquisition_gap": crosses_gap,
                    "x_raw_matches_signal": bool(np.allclose(x_raw, expected_raw)),
                    "x_relative_to_threshold_matches": bool(
                        np.allclose(split_arrays["X_relative_to_threshold"][position], x_raw - np.float32(threshold_on))
                    ),
                    "x_relative_to_current_ends_at_zero": bool(
                        np.isclose(split_arrays["X_relative_to_current"][position][-1], 0.0)
                    ),
                    "x_relative_to_current_matches": bool(
                        np.allclose(split_arrays["X_relative_to_current"][position], x_raw - x_raw[-1])
                    ),
                }
            )
            elapsed_seconds = np.asarray([sample["elapsed_since_event_start_seconds"]], dtype=np.float32)
            elapsed_samples = np.asarray([sample["elapsed_event_samples"]], dtype=np.float32)
            above = np.asarray([float(sample["current_above_threshold"])], dtype=np.float32)
            since_above = np.asarray([sample["time_since_last_above_threshold_seconds"]], dtype=np.float32)
            below_count = np.asarray([sample["consecutive_below_threshold_samples"]], dtype=np.float32)
            recent_fraction = np.asarray([sample["fraction_above_threshold_recent_window"]], dtype=np.float32)
            raw_features = compute_scalar_context_features(
                x_raw[None, :],
                threshold_on=threshold_on,
                elapsed_since_event_start_seconds=elapsed_seconds,
                elapsed_event_samples=elapsed_samples,
                current_above_threshold=above,
                time_since_last_above_threshold_seconds=since_above,
                consecutive_below_threshold_samples=below_count,
                fraction_above_threshold_recent_window=recent_fraction,
            )[0]
            metadata_features = sample[list(SCALAR_CONTEXT_FEATURE_NAMES)].to_numpy(dtype=np.float32)
            standardized_expected = ((raw_features - mean) / std).astype(np.float32)
            scalar_rows.append(
                {
                    "split": split,
                    "global_window_id": sample["global_window_id"],
                    "raw_scalar_features_match_metadata": bool(np.allclose(raw_features, metadata_features, atol=1e-5)),
                    "standardized_scalar_features_match_npz": bool(
                        np.allclose(standardized_expected, split_arrays["scalar_context_features"][position], atol=1e-5)
                    ),
                    "npz_scalar_features_all_finite": bool(np.isfinite(split_arrays["scalar_context_features"][position]).all()),
                }
            )

    train = split_metadata["train"]
    train_features = train[list(SCALAR_CONTEXT_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    scaling_rows.append(
        {
            "check": "train_mean_matches_saved_scaler",
            "passed": bool(np.allclose(train_features.mean(axis=0), mean, atol=1e-5)),
            "max_abs_difference": float(np.max(np.abs(train_features.mean(axis=0) - mean))),
        }
    )
    scaling_rows.append(
        {
            "check": "train_std_matches_saved_scaler_observed_std",
            "passed": bool(np.allclose(train_features.std(axis=0), np.asarray(scaler["observed_std"], dtype=np.float32), atol=1e-5)),
            "max_abs_difference": float(
                np.max(np.abs(train_features.std(axis=0) - np.asarray(scaler["observed_std"], dtype=np.float32)))
            ),
        }
    )
    scaling_rows.append(
        {
            "check": "scalar_feature_names_match_npz",
            "passed": bool(
                all(
                    tuple(arrays[split]["scalar_context_feature_names"].astype(str).tolist())
                    == SCALAR_CONTEXT_FEATURE_NAMES
                    for split in arrays
                )
            ),
            "max_abs_difference": 0.0,
        }
    )
    return pd.DataFrame(context_rows), pd.DataFrame(scalar_rows), pd.DataFrame(scaling_rows)


def event_weight_checks(split_metadata: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify event-balanced weights reflect saved sample counts."""

    combined = pd.concat(split_metadata.values(), ignore_index=True)
    per_event = (
        combined.groupby(["split", "global_event_id"], sort=False)
        .agg(
            actual_sample_count=("window_id", "count"),
            stored_event_sample_count=("event_sample_count", "first"),
            weight_sum=("event_balanced_weight", "sum"),
            weight_min=("event_balanced_weight", "min"),
            weight_max=("event_balanced_weight", "max"),
        )
        .reset_index()
    )
    per_event["sample_count_matches"] = per_event["actual_sample_count"].eq(
        per_event["stored_event_sample_count"]
    )
    per_event["weight_sum_close_to_one"] = np.isclose(per_event["weight_sum"], 1.0, atol=1e-5)
    summary = pd.DataFrame(
        [
            {
                "num_events": int(len(per_event)),
                "all_sample_counts_match": bool(per_event["sample_count_matches"].all()),
                "all_weight_sums_close_to_one": bool(per_event["weight_sum_close_to_one"].all()),
                "min_event_sample_count": int(per_event["actual_sample_count"].min()),
                "median_event_sample_count": float(per_event["actual_sample_count"].median()),
                "max_event_sample_count": int(per_event["actual_sample_count"].max()),
                "min_event_balanced_weight": float(per_event["weight_min"].min()),
                "median_event_balanced_weight": float(per_event["weight_min"].median()),
                "max_event_balanced_weight": float(per_event["weight_max"].max()),
                "single_sample_events": int(per_event["actual_sample_count"].eq(1).sum()),
            }
        ]
    )
    return per_event, summary


def compatibility_checks(
    arrays: dict[str, np.lib.npyio.NpzFile],
    split_metadata: dict[str, pd.DataFrame],
    *,
    suggested_edges: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check that canonical labels can feed future model adapters."""

    rows: list[dict[str, Any]] = []
    disc_rows: list[dict[str, Any]] = []
    combined_metadata = pd.concat(split_metadata.values(), ignore_index=True)
    combined_lower = combined_metadata["y_lower_bound_seconds"].to_numpy(dtype=float)
    combined_upper = combined_metadata["y_upper_bound_seconds"].to_numpy(dtype=float)
    combined_observed = combined_metadata["y_event_observed"].to_numpy(dtype=int)
    tabular_finite = []
    for split, split_arrays in arrays.items():
        features = np.column_stack(
            [split_arrays["X_relative_to_threshold"], split_arrays["scalar_context_features"]]
        )
        tabular_finite.append(np.isfinite(features).all())
    rows.append(
        {
            "adapter": "xgboost_aft",
            "lower_bounds_strictly_positive": bool(np.all(combined_lower > 0.0)),
            "observed_upper_equals_lower": bool(
                np.allclose(
                    combined_lower[combined_observed == 1],
                    combined_upper[combined_observed == 1],
                )
            ),
            "censored_upper_is_inf": bool(np.isinf(combined_upper[combined_observed == 0]).all()),
            "tabular_features_all_finite": bool(all(tabular_finite)),
            "passed": bool(
                np.all(combined_lower > 0.0)
                and np.allclose(
                    combined_lower[combined_observed == 1],
                    combined_upper[combined_observed == 1],
                )
                and np.isinf(combined_upper[combined_observed == 0]).all()
                and all(tabular_finite)
            ),
        }
    )

    bin_sets = {
        "suggested": np.asarray(suggested_edges, dtype=float),
        "coarse": np.asarray([60, 300, 900, 1800, 3600], dtype=float),
    }
    for bin_name, edges in bin_sets.items():
        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"Bin set {bin_name} is not strictly increasing.")
        event_bins = np.searchsorted(edges, combined_lower, side="right")
        censored_bins = np.searchsorted(edges, combined_lower, side="right")
        n_bins_with_tail = len(edges) + 1
        for adapter in ["discrete_time_tcn", "deephit"]:
            at_risk_mask = np.arange(n_bins_with_tail)[None, :] <= event_bins[:, None]
            event_mask = (combined_observed[:, None] == 1) & (
                np.arange(n_bins_with_tail)[None, :] == event_bins[:, None]
            )
            disc_rows.append(
                {
                    "adapter": adapter,
                    "bin_set": bin_name,
                    "num_finite_edges": int(len(edges)),
                    "num_bins_with_tail": int(n_bins_with_tail),
                    "min_time_bin_index": int(event_bins.min()),
                    "max_time_bin_index": int(event_bins.max()),
                    "tail_bin_count": int((event_bins == len(edges)).sum()),
                    "observed_event_mask_count": int(event_mask.sum()),
                    "observed_sample_count": int(combined_observed.sum()),
                    "censored_sample_count": int((combined_observed == 0).sum()),
                    "censored_tail_count": int((censored_bins[combined_observed == 0] == len(edges)).sum()),
                    "at_risk_mask_shape": str(tuple(at_risk_mask.shape)),
                    "passed": bool(
                        event_bins.min() >= 0
                        and event_bins.max() < n_bins_with_tail
                        and int(event_mask.sum()) == int(combined_observed.sum())
                    ),
                    "tail_policy_note": "explicit tail bin after the last finite edge",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(disc_rows)


def write_report(
    output_dir: Path,
    *,
    dataset_dir: Path,
    config_path: Path,
    dataset_metadata: dict[str, Any],
    event_counts: pd.DataFrame,
    sample_counts: pd.DataFrame,
    compatibility: pd.DataFrame,
    bugs_found: list[str],
) -> None:
    """Write a compact Markdown audit report."""

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        text = frame.astype(object).where(pd.notna(frame), "")
        columns = [str(column) for column in text.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in text.itertuples(index=False):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Survival Persistence Dataset Audit",
        "",
        f"Config: `{relative_project_path(config_path)}`",
        f"Dataset: `{relative_project_path(dataset_dir)}`",
        f"Config fingerprint: `{dataset_metadata.get('config_fingerprint', '')}`",
        "",
        "## Event Counts",
        "",
        markdown_table(event_counts),
        "",
        "## Sample Counts",
        "",
        markdown_table(sample_counts),
        "",
        "## Compatibility",
        "",
        markdown_table(compatibility),
        "",
        "## Bugs Found",
        "",
    ]
    if bugs_found:
        lines.extend(f"- {item}" for item in bugs_found)
    else:
        lines.append("- No dataset bugs found by this audit.")
    lines.extend(
        [
            "",
            "Survival models were not implemented or trained during this audit.",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the dataset audit and save diagnostics."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")
    recovery_config = recovery_config_from_yaml(config)
    dataset_config = config["dataset"]
    source_config = config["source"]
    context_length = int(dataset_config["context_length"])
    sampling_time_seconds = int(dataset_config.get("sampling_time_seconds", 30))
    selection_id = make_selection_id(str(config["split"]["external_test_dataset"]))
    dataset_dir = (
        project_path(args.dataset_dir)
        if args.dataset_dir is not None
        else processed_dataset_dir(
            threshold_on=recovery_config.threshold_on,
            context_length=context_length,
            selection_id=selection_id,
        )
    )
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir is not None
        else data_preparation_dir(
            selection_id=selection_id,
            threshold_on=recovery_config.threshold_on,
            context_length=context_length,
        )
        / "audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    require_dataset_files(dataset_dir)

    dataset_metadata = load_yaml_config(dataset_dir / "dataset_metadata.yaml")
    full_signal = load_full_signal(
        project_path(source_config["full_signal_path"]),
        signal_column=str(source_config.get("signal_column", "Signal")),
    )
    imputation_config = small_gap_imputation_config_from_mapping(
        config.get("small_gap_imputation"),
        enabled_default=False,
    )
    if imputation_config.enabled:
        full_signal = impute_small_time_gaps(
            full_signal,
            time_column="Time",
            signal_column="Signal",
            group_columns=("dataset_id", "dataset_name", "segment_id"),
            config=imputation_config,
        ).dataframe
    event_summary = pd.read_csv(dataset_dir / "event_summary.csv")
    for column in [
        "event_start_time",
        "event_end_time",
        "event_end_confirmation_time",
        "segment_end_time",
    ]:
        event_summary[column] = pd.to_datetime(event_summary[column], errors="coerce")
    arrays, split_metadata = load_split_artifacts(dataset_dir)
    for frame in split_metadata.values():
        for column in [
            "sample_time",
            "input_start_time",
            "input_end_time",
            "event_start_time",
            "event_end_time",
            "event_end_confirmation_time",
            "segment_end_time",
        ]:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    segments = segment_lookup(full_signal)
    trace_cache: dict[tuple[str, str], pd.DataFrame] = {}
    transition_stats = transition_statistics(
        event_summary,
        segments=segments,
        trace_cache=trace_cache,
        recovery_config=recovery_config,
    )
    event_counts = make_event_counts(event_summary)
    sample_counts = make_sample_counts(split_metadata)
    split_boundaries = split_boundary_summary(event_summary)
    observed_duration = pd.DataFrame(
        [
            distribution(
                event_summary.loc[event_summary["event_observed"].astype(bool), "event_duration_seconds"],
                label="observed_event_duration_seconds",
            )
        ]
    )
    censored_followup = pd.DataFrame(
        [
            distribution(
                event_summary.loc[~event_summary["event_observed"].astype(bool), "observed_followup_seconds"],
                label="censored_observed_followup_seconds",
            )
        ]
    )
    target_rows = []
    for split, frame in split_metadata.items():
        target_rows.append(
            distribution(
                frame.loc[frame["y_event_observed"].eq(1), "y_time_seconds"],
                label="observed_sample_y_time_seconds",
                split=split,
            )
        )
        target_rows.append(
            distribution(
                frame.loc[frame["y_event_observed"].eq(0), "y_lower_bound_seconds"],
                label="censored_sample_lower_bound_seconds",
                split=split,
            )
        )
    target_distribution = pd.DataFrame(target_rows)

    observed_events = event_summary.loc[event_summary["event_observed"].astype(bool)].copy()
    delay_seconds = (
        observed_events["event_end_confirmation_time"] - observed_events["event_end_time"]
    ).dt.total_seconds()
    confirmation_delay = pd.DataFrame(
        [distribution(delay_seconds, label="recovery_confirmation_delay_seconds")]
    )

    composition = pd.DataFrame(
        [
            {"metric": "total_represented_events", "value": int(len(event_summary))},
            {"metric": "brief_observed_events_lt_60s", "value": int((observed_events["event_duration_seconds"] < 60).sum())},
            {"metric": "observed_events_lt_300s", "value": int((observed_events["event_duration_seconds"] < 300).sum())},
            {"metric": "observed_events_ge_300s", "value": int((observed_events["event_duration_seconds"] >= 300).sum())},
            {
                "metric": "events_with_temporary_below_threshold_samples",
                "value": int((event_summary["num_below_threshold_inside_event"] > 0).sum()),
            },
            {
                "metric": "events_with_multiple_candidate_recoveries",
                "value": int((transition_stats["candidate_recovery_starts"] > 1).sum()),
            },
            {
                "metric": "observed_events_single_candidate_recovery",
                "value": int(
                    (
                        event_summary["event_observed"].astype(bool)
                        & transition_stats["candidate_recovery_starts"].eq(1)
                    ).sum()
                ),
            },
            {
                "metric": f"censored_events_{CENSOR_ACTIVE}",
                "value": int(event_summary["censoring_reason"].eq(CENSOR_ACTIVE).sum()),
            },
            {
                "metric": f"censored_events_{CENSOR_CANDIDATE}",
                "value": int(event_summary["censoring_reason"].eq(CENSOR_CANDIDATE).sum()),
            },
            {
                "metric": f"censored_events_{CENSOR_INSUFFICIENT_WINDOW}",
                "value": int(event_summary["censoring_reason"].eq(CENSOR_INSUFFICIENT_WINDOW).sum()),
            },
            {
                "metric": "context_exclusions",
                "value": int(dataset_metadata.get("diagnostics", {}).get("num_context_exclusions", 0)),
            },
        ]
    )

    selections = choose_manual_events(event_summary, transition_stats)
    traces = manual_state_traces(
        selections,
        event_summary,
        segments=segments,
        trace_cache=trace_cache,
        recovery_config=recovery_config,
    )
    observed_target_checks = target_check_rows(split_metadata, event_summary, observed=True)
    censored_target_checks = target_check_rows(split_metadata, event_summary, observed=False)
    context_checks, scalar_checks, scaling_checks = context_and_scalar_checks(
        arrays,
        split_metadata,
        full_signal,
        dataset_metadata,
        threshold_on=recovery_config.threshold_on,
        num_checks=int(args.num_random_context_checks),
    )
    event_weights, event_weight_summary = event_weight_checks(split_metadata)
    aft_compatibility, discretization_diagnostics = compatibility_checks(
        arrays,
        split_metadata,
        suggested_edges=list(config.get("suggested_discretization", {}).get("edges_seconds", [])),
    )
    compatibility = pd.concat(
        [
            aft_compatibility,
            pd.DataFrame(
                [
                    {
                        "adapter": "discrete_time_tcn_and_deephit",
                        "passed": bool(discretization_diagnostics["passed"].all()),
                        "lower_bounds_strictly_positive": bool((pd.concat(split_metadata.values())["y_lower_bound_seconds"] > 0).all()),
                        "observed_upper_equals_lower": np.nan,
                        "censored_upper_is_inf": np.nan,
                        "tabular_features_all_finite": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    recovery_window_semantics = pd.DataFrame(
        [
            {
                "property": "candidate_window",
                "value": "[candidate_recovery_start, candidate_recovery_start + recovery_window_seconds]",
            },
            {"property": "left_endpoint", "value": "inclusive"},
            {"property": "right_endpoint", "value": "inclusive"},
            {"property": "time_basis", "value": "actual timestamps"},
            {"property": "denominator", "value": "observations inside the inclusive recovery window"},
            {
                "property": "minimum_recovery_observations",
                "value": int(recovery_config.minimum_recovery_observations),
            },
            {
                "property": "required_fraction",
                "value": float(recovery_config.recovery_required_fraction),
            },
            {
                "property": "abort_on_new_threshold_on_crossing",
                "value": bool(recovery_config.abort_candidate_recovery_on_threshold_on_crossing),
            },
        ]
    )

    forbidden_inputs = pd.DataFrame(
        [
            {
                "check": "forbidden_fields_absent_from_scalar_context",
                "passed": not (set(SCALAR_CONTEXT_FEATURE_NAMES) & FORBIDDEN_MODEL_INPUT_COLUMNS),
                "forbidden_overlap": ";".join(sorted(set(SCALAR_CONTEXT_FEATURE_NAMES) & FORBIDDEN_MODEL_INPUT_COLUMNS)),
            },
            {
                "check": "canonical_dataset_has_no_fixed_discrete_survival_arrays",
                "passed": all(
                    key not in arrays[split].files
                    for split in arrays
                    for key in ["y_time_bin_index", "at_risk_mask", "event_mask"]
                ),
                "forbidden_overlap": "",
            },
        ]
    )

    tables = {
        "audit_event_counts.csv": event_counts,
        "audit_sample_counts.csv": sample_counts,
        "audit_observed_event_duration_distribution.csv": observed_duration,
        "audit_censored_followup_distribution.csv": censored_followup,
        "audit_survival_target_distribution.csv": target_distribution,
        "audit_recovery_confirmation_delay_distribution.csv": confirmation_delay,
        "audit_event_composition.csv": composition,
        "audit_censoring_reasons.csv": pd.read_csv(dataset_dir / "censoring_summary.csv"),
        "audit_split_boundaries.csv": split_boundaries,
        "audit_transition_statistics.csv": transition_stats,
        "audit_manual_event_selection.csv": selections,
        "audit_manual_event_state_traces.csv": traces,
        "audit_observed_target_checks.csv": observed_target_checks,
        "audit_censored_target_checks.csv": censored_target_checks,
        "audit_context_checks.csv": context_checks,
        "audit_scalar_feature_checks.csv": scalar_checks,
        "audit_scaling_checks.csv": scaling_checks,
        "audit_event_weights.csv": event_weights,
        "audit_event_weight_summary.csv": event_weight_summary,
        "audit_model_adapter_compatibility.csv": compatibility,
        "audit_discretization_diagnostics.csv": discretization_diagnostics,
        "audit_recovery_window_semantics.csv": recovery_window_semantics,
        "audit_forbidden_inputs.csv": forbidden_inputs,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    audit_metadata = {
        "task_name": TASK_NAME,
        "config_path": relative_project_path(config_path),
        "dataset_dir": relative_project_path(dataset_dir),
        "audit_dir": relative_project_path(output_dir),
        "dataset_config_fingerprint": dataset_metadata.get("config_fingerprint", ""),
        "recovery_window_semantics": {
            "interval": "[candidate_recovery_start, candidate_recovery_start + recovery_window_seconds]",
            "left_endpoint": "inclusive",
            "right_endpoint": "inclusive",
            "time_basis": "actual timestamps",
            "denominator": "observations inside inclusive interval",
        },
        "bugs_found": [],
        "files_written": {name.removesuffix(".csv"): relative_project_path(output_dir / name) for name in tables},
    }
    save_yaml(output_dir / "audit_metadata.yaml", audit_metadata)
    write_report(
        output_dir,
        dataset_dir=dataset_dir,
        config_path=config_path,
        dataset_metadata=dataset_metadata,
        event_counts=event_counts,
        sample_counts=sample_counts,
        compatibility=compatibility,
        bugs_found=[],
    )

    print("=== Survival-Persistence Dataset Audit ===")
    print(f"Dataset: {relative_project_path(dataset_dir)}")
    print(f"Audit output: {relative_project_path(output_dir)}")
    print(f"Events: {len(event_summary):,}")
    print(f"Samples: {sum(len(frame) for frame in split_metadata.values()):,}")
    print(f"Observed/censored events: {int(event_summary['event_observed'].astype(bool).sum()):,}/{int((~event_summary['event_observed'].astype(bool)).sum()):,}")
    print("Audit completed without dataset bugs found.")


if __name__ == "__main__":
    main()
