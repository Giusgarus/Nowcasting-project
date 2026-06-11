"""Chronological train, validation, and test splitting utilities."""

from dataclasses import dataclass

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
