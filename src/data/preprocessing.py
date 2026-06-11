"""Shared signal-only cleaning and temporal segmentation."""

from typing import Literal

import pandas as pd

DuplicateStrategy = Literal["mean", "first", "last", "error"]


def clean_signal_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_name: str,
    gap_threshold_seconds: float,
    duplicate_timestamp_strategy: DuplicateStrategy = "mean",
) -> pd.DataFrame:
    """Clean parsed Signal rows and assign continuous temporal segments."""

    if gap_threshold_seconds <= 0:
        raise ValueError("gap_threshold_seconds must be positive.")
    if duplicate_timestamp_strategy not in {"mean", "first", "last", "error"}:
        raise ValueError("Unsupported duplicate timestamp strategy.")

    clean = dataframe.loc[:, ["Time", "Signal"]].copy()
    clean = clean.dropna(subset=["Time", "Signal"])
    clean["Time"] = pd.to_datetime(clean["Time"], errors="coerce")
    clean["Signal"] = pd.to_numeric(clean["Signal"], errors="coerce")
    clean = clean.dropna(subset=["Time", "Signal"]).sort_values("Time", kind="stable")

    duplicates = clean["Time"].duplicated(keep=False)
    if duplicates.any():
        if duplicate_timestamp_strategy == "error":
            raise ValueError(f"Duplicate timestamps found in {dataset_name}.")
        if duplicate_timestamp_strategy == "mean":
            clean = clean.groupby("Time", as_index=False, sort=True)["Signal"].mean()
        else:
            clean = clean.drop_duplicates(
                subset="Time",
                keep=duplicate_timestamp_strategy,
            )

    clean = clean.sort_values("Time", kind="stable").reset_index(drop=True)
    clean.insert(0, "dataset_id", dataset_id)
    clean.insert(1, "dataset_name", dataset_name)
    clean["row_id"] = range(len(clean))
    clean["dt_seconds"] = clean["Time"].diff().dt.total_seconds()
    segment_number = clean["dt_seconds"].gt(gap_threshold_seconds).cumsum()
    clean["segment_id"] = segment_number.map(
        lambda number: f"{dataset_id}_segment_{int(number):04d}"
    )
    return clean[
        [
            "dataset_id",
            "dataset_name",
            "row_id",
            "Time",
            "Signal",
            "dt_seconds",
            "segment_id",
        ]
    ]
