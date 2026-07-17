"""Tests for shared small-gap imputation utilities."""

from __future__ import annotations

import pandas as pd

from src.data.imputation import (
    SmallGapImputationConfig,
    impute_small_time_gaps,
)


def test_imputes_one_missing_sample_inside_small_gap() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["a", "a", "a"],
            "segment_id": ["s1", "s1", "s1"],
            "Time": pd.to_datetime(
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:01:00",
                    "2021-01-01 00:01:30",
                ]
            ),
            "Signal": [10.0, 16.0, 18.0],
        }
    )

    result = impute_small_time_gaps(
        frame,
        config=SmallGapImputationConfig(enabled=True),
    )

    assert len(result.dataframe) == 4
    inserted = result.dataframe.loc[result.dataframe["is_imputed"]]
    assert inserted["Time"].iloc[0] == pd.Timestamp("2021-01-01 00:00:30")
    assert inserted["Signal"].iloc[0] == 13.0
    assert int(result.summary["num_inserted_rows"].sum()) == 1
    assert int(result.summary["num_imputed_gaps"].sum()) == 1


def test_does_not_impute_large_gap() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["a", "a"],
            "segment_id": ["s1", "s1"],
            "Time": pd.to_datetime(
                ["2021-01-01 00:00:00", "2021-01-01 00:05:00"]
            ),
            "Signal": [10.0, 20.0],
        }
    )

    result = impute_small_time_gaps(
        frame,
        config=SmallGapImputationConfig(enabled=True),
    )

    assert len(result.dataframe) == 2
    assert int(result.dataframe["is_imputed"].sum()) == 0
    assert int(result.summary["num_inserted_rows"].sum()) == 0


def test_does_not_impute_across_segments() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["a", "a"],
            "segment_id": ["s1", "s2"],
            "Time": pd.to_datetime(
                ["2021-01-01 00:00:00", "2021-01-01 00:01:00"]
            ),
            "Signal": [10.0, 20.0],
        }
    )

    result = impute_small_time_gaps(
        frame,
        config=SmallGapImputationConfig(enabled=True),
    )

    assert len(result.dataframe) == 2
    assert int(result.dataframe["is_imputed"].sum()) == 0
    assert int(result.summary["num_inserted_rows"].sum()) == 0
