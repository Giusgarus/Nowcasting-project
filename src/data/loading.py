"""Load delimited files into the strict univariate Time and Signal schema."""

import csv
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.data.validation import validate_signal_columns

HeaderMode = bool | Literal["auto"]
TimeParser = Callable[[str], datetime]
SignalParser = Callable[[str], float]
SUPPORTED_TIME_FORMATS = (
    "%d/%m/%Y %H:%M:%S",
    "%d-%b-%Y %H:%M:%S",
)


@dataclass(frozen=True)
class SignalRecord:
    """One normalized timestamp and signal observation."""

    time: datetime
    signal: float


@dataclass(frozen=True)
class SignalDataset:
    """Loaded observations normalized to exactly Time and Signal."""

    records: tuple[SignalRecord, ...]
    ignored_columns: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, str]:
        """Return the normalized two-column schema."""

        return ("Time", "Signal")


def parse_time(value: str) -> datetime:
    """Parse ISO-8601 or explicitly supported source timestamp formats."""

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    for time_format in SUPPORTED_TIME_FORMATS:
        try:
            return datetime.strptime(value, time_format)
        except ValueError:
            continue

    raise ValueError(f"Unsupported Time format: {value}")


def load_signal_file(
    path: str | Path,
    *,
    has_header: HeaderMode = "auto",
    time_col: str = "Time",
    signal_col: str = "Signal",
    time_col_index: int = 0,
    signal_col_index: int = 1,
    delimiter: str | None = None,
    time_parser: TimeParser = parse_time,
    signal_parser: SignalParser = float,
) -> SignalDataset:
    """Load, normalize, parse, and chronologically sort signal observations.

    Header files use the configured Time and Signal names. No-header files use
    the configured positional indices. All additional columns are explicitly
    ignored and reported in the returned dataset.
    """

    if has_header not in (True, False, "auto"):
        raise ValueError("has_header must be true, false, or 'auto'.")
    if time_col_index < 0 or signal_col_index < 0:
        raise ValueError("Column indices cannot be negative.")
    if time_col_index == signal_col_index:
        raise ValueError("Time and Signal column indices must be different.")
    if delimiter is not None and len(delimiter) != 1:
        raise ValueError("delimiter must be one character.")

    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Input file is empty: {source}")

    sample = text[:8192]
    dialect = _detect_dialect(sample, delimiter)
    header_present = _detect_header(sample, has_header)
    rows = list(csv.reader(text.splitlines(), dialect=dialect))
    if not rows:
        raise ValueError(f"Input file has no rows: {source}")

    if header_present:
        time_index, signal_index, ignored_columns, data_rows = _header_layout(
            rows, time_col, signal_col
        )
    else:
        time_index, signal_index, ignored_columns, data_rows = _positional_layout(
            rows, time_col_index, signal_col_index
        )

    records = tuple(
        sorted(
            _parse_rows(
                data_rows,
                time_index=time_index,
                signal_index=signal_index,
                time_parser=time_parser,
                signal_parser=signal_parser,
            ),
            key=lambda record: record.time,
        )
    )
    if not records:
        raise ValueError(f"Input file has no data rows: {source}")

    return SignalDataset(records=records, ignored_columns=ignored_columns)


def load_signal_dataset(
    path: str | Path,
    *,
    has_header: HeaderMode = "auto",
    delimiter: str | None = None,
    suspected_sentinel_threshold: float = -10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw data into a valid-row DataFrame with exactly Time and Signal.

    Missing, unparseable, or non-finite Time and Signal rows are excluded from
    the returned DataFrame but are never hidden: their counts are returned in
    metadata. Signal values at or below ``suspected_sentinel_threshold`` are
    reported as suspected sentinels but remain in the DataFrame. Duplicate
    timestamps are preserved. No resampling, filling, or attenuation
    construction is performed.
    """

    if has_header not in (True, False, "auto"):
        raise ValueError("has_header must be true, false, or 'auto'.")
    if delimiter is not None and len(delimiter) != 1:
        raise ValueError("delimiter must be one character.")

    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Input file is empty: {source}")

    sample = text[:8192]
    dialect = _detect_dialect(sample, delimiter)
    header_present = _detect_header(sample, has_header)
    rows = list(csv.reader(text.splitlines(), dialect=dialect))
    if not rows:
        raise ValueError(f"Input file has no rows: {source}")

    raw_num_columns = max(len(row) for row in rows)
    if raw_num_columns < 2:
        raise ValueError("At least two columns are required: Time and Signal.")

    if header_present:
        header = _normalized_header(rows[0], raw_num_columns)
        data_rows = rows[1:]
        time_index, signal_index, used_positional_fallback = _analysis_column_indices(
            header
        )
        used_time_column = header[time_index]
        used_signal_column = header[signal_index]
    else:
        data_rows = rows
        time_index, signal_index = 0, 1
        used_positional_fallback = True
        used_time_column = "column_0"
        used_signal_column = "column_1"

    raw_rows = len(data_rows)
    raw_time = pd.Series(
        [_row_field(row, time_index) for row in data_rows],
        dtype="string",
    )
    raw_signal = pd.Series(
        [_row_field(row, signal_index) for row in data_rows],
        dtype="string",
    )
    parsed_time = pd.to_datetime(
        raw_time,
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )
    parsed_signal = pd.to_numeric(raw_signal, errors="coerce")

    missing_timestamp = _missing_like_mask(raw_time)
    unparseable_timestamp = parsed_time.isna() & ~missing_timestamp
    invalid_timestamp = missing_timestamp | unparseable_timestamp

    missing_signal = _missing_like_mask(raw_signal)
    non_numeric_signal = parsed_signal.isna() & ~missing_signal
    non_finite_signal = parsed_signal.notna() & ~np.isfinite(parsed_signal)
    invalid_signal = missing_signal | non_numeric_signal | non_finite_signal

    suspected_sentinel_signal = (
        parsed_signal.notna()
        & np.isfinite(parsed_signal)
        & parsed_signal.le(suspected_sentinel_threshold)
    )
    strict_parse_invalid = invalid_timestamp | invalid_signal
    total_flagged = strict_parse_invalid | suspected_sentinel_signal
    valid_mask = ~(invalid_timestamp | invalid_signal)
    dataframe = pd.DataFrame(
        {
            "Time": parsed_time[valid_mask],
            "Signal": parsed_signal[valid_mask].astype(float),
        }
    )
    dataframe = dataframe.sort_values("Time", kind="stable").reset_index(drop=True)

    metadata: dict[str, Any] = {
        "dataset_name": source.name,
        "file_path": str(source),
        "has_header_detected": bool(header_present),
        "raw_num_columns": raw_num_columns,
        "ignored_columns_count": max(0, raw_num_columns - 2),
        "used_time_column": used_time_column,
        "used_signal_column": used_signal_column,
        "used_positional_fallback": used_positional_fallback,
        "raw_rows": raw_rows,
        "valid_rows": int(valid_mask.sum()),
        "missing_timestamp_rows": int(missing_timestamp.sum()),
        "unparseable_timestamp_rows": int(unparseable_timestamp.sum()),
        "invalid_timestamp_rows": int(invalid_timestamp.sum()),
        "missing_signal_rows": int(missing_signal.sum()),
        "non_numeric_signal_rows": int(non_numeric_signal.sum()),
        "non_finite_signal_rows": int(non_finite_signal.sum()),
        "invalid_signal_rows": int(invalid_signal.sum()),
        "strict_parse_invalid_rows": int(strict_parse_invalid.sum()),
        "suspected_sentinel_threshold": suspected_sentinel_threshold,
        "suspected_sentinel_signal_rows": int(suspected_sentinel_signal.sum()),
        "total_flagged_rows": int(total_flagged.sum()),
        "rows_without_quality_flags": int((~total_flagged).sum()),
        "duplicate_timestamps": int(dataframe["Time"].duplicated().sum()),
    }
    return dataframe, metadata


def _normalized_header(row: list[str], width: int) -> list[str]:
    header = [column.strip() for column in row]
    return header + [f"column_{index}" for index in range(len(header), width)]


def _analysis_column_indices(header: list[str]) -> tuple[int, int, bool]:
    if "Time" in header and "Signal" in header:
        return header.index("Time"), header.index("Signal"), False
    return 0, 1, True


def _row_field(row: list[str], index: int) -> str | None:
    if index >= len(row):
        return None
    value = row[index].strip()
    return value or None


def _missing_like_mask(values: pd.Series) -> pd.Series:
    """Return rows containing blanks or common textual missing-value markers."""

    missing_markers = {"na", "n/a", "nan", "none", "null"}
    return values.isna() | values.str.strip().str.lower().isin(missing_markers)


def _detect_dialect(sample: str, delimiter: str | None) -> csv.Dialect:
    if delimiter is not None:
        return type("ConfiguredDialect", (csv.excel,), {"delimiter": delimiter})

    try:
        return csv.Sniffer().sniff(sample)
    except csv.Error:
        return csv.excel


def _detect_header(sample: str, has_header: HeaderMode) -> bool:
    if has_header != "auto":
        return has_header
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error as error:
        raise ValueError(
            "Could not infer whether the file has a header; configure has_header."
        ) from error


def _header_layout(
    rows: list[list[str]],
    time_col: str,
    signal_col: str,
) -> tuple[int, int, tuple[str, ...], list[list[str]]]:
    header = [column.strip() for column in rows[0]]
    selection = validate_signal_columns(
        header,
        time_col=time_col,
        signal_col=signal_col,
    )
    max_width = max(len(row) for row in rows)
    unnamed_extra_columns = tuple(
        f"column_{index}" for index in range(len(header), max_width)
    )
    return (
        header.index(time_col),
        header.index(signal_col),
        selection.ignored_columns + unnamed_extra_columns,
        rows[1:],
    )


def _positional_layout(
    rows: list[list[str]],
    time_col_index: int,
    signal_col_index: int,
) -> tuple[int, int, tuple[str, ...], list[list[str]]]:
    required_width = max(time_col_index, signal_col_index) + 1
    if len(rows[0]) < required_width:
        raise ValueError("At least two columns are required: Time and Signal.")

    max_width = max(len(row) for row in rows)
    ignored_columns = tuple(
        f"column_{index}"
        for index in range(max_width)
        if index not in (time_col_index, signal_col_index)
    )
    return time_col_index, signal_col_index, ignored_columns, rows


def _parse_rows(
    rows: list[list[str]],
    *,
    time_index: int,
    signal_index: int,
    time_parser: TimeParser,
    signal_parser: SignalParser,
) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    required_width = max(time_index, signal_index) + 1
    for row_number, row in enumerate(rows, start=1):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) < required_width:
            raise ValueError(
                f"Row {row_number} has fewer than the required two columns."
            )
        try:
            time = time_parser(row[time_index].strip())
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Time value on row {row_number}.") from error
        try:
            signal = signal_parser(row[signal_index].strip())
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Signal value on row {row_number}.") from error
        records.append(SignalRecord(time=time, signal=signal))
    return records
