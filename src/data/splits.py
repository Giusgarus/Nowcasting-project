"""Chronological train, validation, and test splitting utilities."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitRange:
    """A half-open interval identifying one chronological split."""

    start: int
    stop: int

    @property
    def size(self) -> int:
        """Return the number of samples in the split."""

        return self.stop - self.start


@dataclass(frozen=True)
class ChronologicalSplit:
    """Half-open train, validation, and test intervals."""

    train: SplitRange
    validation: SplitRange
    test: SplitRange


def chronological_split(
    n_samples: int,
    train_fraction: float,
    validation_fraction: float,
) -> ChronologicalSplit:
    """Create contiguous chronological splits without shuffling.

    The remainder after the train and validation fractions is assigned to the
    test split. At least one sample must be present in each split.
    """

    if n_samples < 3:
        raise ValueError("At least three samples are required.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    train_stop = int(n_samples * train_fraction)
    validation_stop = int(n_samples * (train_fraction + validation_fraction))

    split = ChronologicalSplit(
        train=SplitRange(0, train_stop),
        validation=SplitRange(train_stop, validation_stop),
        test=SplitRange(validation_stop, n_samples),
    )
    if min(split.train.size, split.validation.size, split.test.size) < 1:
        raise ValueError("The requested fractions produce an empty split.")
    return split


def assign_event_splits(
    events: pd.DataFrame,
    *,
    train_ratio: float,
    validation_ratio: float,
) -> pd.DataFrame:
    """Assign chronological train/validation/test splits per dataset."""

    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between zero and one.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than one.")

    assigned_frames = []
    for _, dataset_events in events.groupby("dataset_id", sort=False):
        ordered = dataset_events.sort_values("event_timestamp", kind="stable").copy()
        count = len(ordered)
        train_stop = int(count * train_ratio)
        validation_stop = int(count * (train_ratio + validation_ratio))
        splits = (
            ["train"] * train_stop
            + ["validation"] * (validation_stop - train_stop)
            + ["test"] * (count - validation_stop)
        )
        ordered["split"] = splits
        assigned_frames.append(ordered)
    if not assigned_frames:
        return events.assign(split=pd.Series(dtype="string"))
    return pd.concat(assigned_frames, ignore_index=True)


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
