"""Tests for strict signal-only file loading."""

from datetime import datetime

import pytest

from src.data.loading import load_signal_dataset, load_signal_file, parse_time


def test_loads_header_file_sorts_time_and_ignores_extra_columns(tmp_path) -> None:
    path = tmp_path / "header.csv"
    path.write_text(
        "Time,Signal,Unused\n"
        "2026-01-01T00:00:30,2.0,x\n"
        "2026-01-01T00:00:00,1.0,y\n",
        encoding="utf-8",
    )

    dataset = load_signal_file(path, has_header=True)

    assert dataset.columns == ("Time", "Signal")
    assert dataset.ignored_columns == ("Unused",)
    assert [record.signal for record in dataset.records] == [1.0, 2.0]


def test_standardizes_configured_source_column_names(tmp_path) -> None:
    path = tmp_path / "source_names.csv"
    path.write_text(
        "Timestamp,Value\n"
        "2026-01-01T00:00:00,1.0\n",
        encoding="utf-8",
    )

    dataset = load_signal_file(
        path,
        has_header=True,
        time_col="Timestamp",
        signal_col="Value",
    )

    assert dataset.columns == ("Time", "Signal")
    assert dataset.records[0].signal == 1.0


def test_loads_no_header_file_using_first_two_columns(tmp_path) -> None:
    path = tmp_path / "no_header.csv"
    path.write_text(
        "2026-01-01T00:00:00,1.0,ignored\n"
        "2026-01-01T00:00:30,2.0,ignored\n",
        encoding="utf-8",
    )

    dataset = load_signal_file(path, has_header=False)

    assert dataset.columns == ("Time", "Signal")
    assert dataset.ignored_columns == ("column_2",)
    assert len(dataset.records) == 2


def test_reports_extra_no_header_columns_found_after_first_row(tmp_path) -> None:
    path = tmp_path / "late_extra.csv"
    path.write_text(
        "2026-01-01T00:00:00,1.0\n"
        "2026-01-01T00:00:30,2.0,ignored\n",
        encoding="utf-8",
    )

    dataset = load_signal_file(path, has_header=False)

    assert dataset.ignored_columns == ("column_2",)


def test_does_not_fill_missing_signal_from_ignored_extra_column(tmp_path) -> None:
    path = tmp_path / "shifted_signal.csv"
    path.write_text(
        "2026-01-01T00:00:00,1.0,\n"
        "2026-01-01T00:00:30,,2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid Signal value on row 2"):
        load_signal_file(path, has_header=False)


def test_auto_detects_header(tmp_path) -> None:
    path = tmp_path / "auto.csv"
    path.write_text(
        "Time,Signal\n"
        "2026-01-01T00:00:00,1.0\n",
        encoding="utf-8",
    )

    dataset = load_signal_file(path, has_header="auto")

    assert dataset.records[0].time == datetime(2026, 1, 1)


def test_auto_detects_no_header_file(tmp_path) -> None:
    path = tmp_path / "auto_no_header.csv"
    path.write_text(
        "2026-01-01T00:00:00,1.0\n"
        "2026-01-01T00:00:30,2.0\n",
        encoding="utf-8",
    )

    dataset = load_signal_file(path, has_header="auto")

    assert [record.signal for record in dataset.records] == [1.0, 2.0]


def test_uses_explicit_delimiter(tmp_path) -> None:
    path = tmp_path / "semicolon.csv"
    path.write_text(
        "Time;Signal\n"
        "2026-01-01T00:00:00;1.0\n",
        encoding="utf-8",
    )

    dataset = load_signal_file(path, has_header=True, delimiter=";")

    assert dataset.records[0].signal == 1.0


def test_rejects_file_with_fewer_than_two_columns(tmp_path) -> None:
    path = tmp_path / "one_column.csv"
    path.write_text("Signal\n1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="At least two columns"):
        load_signal_file(path, has_header=True)


def test_non_iso_time_requires_explicit_parser(tmp_path) -> None:
    path = tmp_path / "custom_time.csv"
    path.write_text("Time,Signal\n09/06/2026 12:00,1.0\n", encoding="utf-8")

    dataset = load_signal_file(
        path,
        has_header=True,
        time_parser=lambda value: datetime.strptime(value, "%d/%m/%Y %H:%M"),
    )

    assert dataset.records[0].time == datetime(2026, 6, 9, 12, 0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2021-04-01 00:15:12", datetime(2021, 4, 1, 0, 15, 12)),
        ("01/4/2021 00:15:12", datetime(2021, 4, 1, 0, 15, 12)),
        ("21-Jul-2020 14:10:32", datetime(2020, 7, 21, 14, 10, 32)),
    ],
)
def test_parse_time_supports_observed_raw_formats(
    value: str,
    expected: datetime,
) -> None:
    assert parse_time(value) == expected


def test_dataframe_loader_reports_invalid_rows_and_keeps_duplicates(tmp_path) -> None:
    path = tmp_path / "quality.csv"
    path.write_text(
        "Time,Signal,Unused\n"
        "2026-01-01T00:00:00,1.0,x\n"
        "2026-01-01T00:00:00,2.0,x\n"
        "bad-time,3.0,x\n"
        "2026-01-01T00:00:30,,4.0\n",
        encoding="utf-8",
    )

    dataframe, metadata = load_signal_dataset(path, has_header=True)

    assert list(dataframe.columns) == ["Time", "Signal"]
    assert dataframe["Signal"].tolist() == [1.0, 2.0]
    assert metadata["raw_rows"] == 4
    assert metadata["valid_rows"] == 2
    assert metadata["invalid_timestamp_rows"] == 1
    assert metadata["invalid_signal_rows"] == 1
    assert metadata["duplicate_timestamps"] == 1
    assert metadata["ignored_columns_count"] == 1


def test_dataframe_loader_uses_positional_fallback_for_nonstandard_header(
    tmp_path,
) -> None:
    path = tmp_path / "fallback.csv"
    path.write_text(
        "beacon_time,beacon_signal,extra\n"
        "21-Jul-2020 14:10:32,1.1,x\n",
        encoding="utf-8",
    )

    dataframe, metadata = load_signal_dataset(path, has_header=True)

    assert len(dataframe) == 1
    assert metadata["used_positional_fallback"] is True
    assert metadata["used_time_column"] == "beacon_time"
    assert metadata["used_signal_column"] == "beacon_signal"


def test_dataframe_loader_reports_detailed_invalid_reasons_and_sentinels(
    tmp_path,
) -> None:
    path = tmp_path / "detailed_quality.csv"
    path.write_text(
        "Time,Signal\n"
        "2026-01-01T00:00:00,1.0\n"
        ",2.0\n"
        "bad-time,3.0\n"
        "2026-01-01T00:00:30,\n"
        "2026-01-01T00:01:00,not-a-number\n"
        "2026-01-01T00:01:30,inf\n"
        "2026-01-01T00:02:00,-10\n"
        "2026-01-01T00:02:30,-11\n"
        "bad-time,-12\n",
        encoding="utf-8",
    )

    dataframe, metadata = load_signal_dataset(path, has_header=True)

    assert dataframe["Signal"].tolist() == [1.0, -10.0, -11.0]
    assert metadata["missing_timestamp_rows"] == 1
    assert metadata["unparseable_timestamp_rows"] == 2
    assert metadata["invalid_timestamp_rows"] == 3
    assert metadata["missing_signal_rows"] == 1
    assert metadata["non_numeric_signal_rows"] == 1
    assert metadata["non_finite_signal_rows"] == 1
    assert metadata["invalid_signal_rows"] == 3
    assert metadata["strict_parse_invalid_rows"] == 6
    assert metadata["suspected_sentinel_threshold"] == -10.0
    assert metadata["suspected_sentinel_signal_rows"] == 3
    assert metadata["total_flagged_rows"] == 8
    assert metadata["rows_without_quality_flags"] == 1
