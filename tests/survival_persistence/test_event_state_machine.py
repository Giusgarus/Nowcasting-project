from __future__ import annotations

import pandas as pd

from src.tasks.survival_persistence.data.event_state_machine import (
    StableRecoveryConfig,
    detect_survival_fade_events,
    trace_survival_state_machine,
)


def _frame(signals: list[float], *, freq: str = "30s") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": pd.date_range("2026-01-01 00:00:00", periods=len(signals), freq=freq),
            "Signal": signals,
        }
    )


def _config(**overrides: object) -> StableRecoveryConfig:
    values = {
        "threshold_on": 10.0,
        "threshold_off": 10.0,
        "recovery_window_seconds": 60.0,
        "recovery_required_fraction": 1.0,
        "minimum_recovery_observations": 2,
    }
    values.update(overrides)
    return StableRecoveryConfig(**values)


def test_brief_spike_becomes_observed_short_event() -> None:
    events = detect_survival_fade_events(
        _frame([0, 11, 8, 8, 8]),
        dataset_name="toy.csv",
        dataset_id="dataset_001",
        segment_id="segment_001",
        event_id_prefix="sp_event",
        config=_config(),
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_observed
    assert event.event_start_time == pd.Timestamp("2026-01-01 00:00:30")
    assert event.event_end_time == pd.Timestamp("2026-01-01 00:01:00")
    assert event.event_end_confirmation_time == pd.Timestamp("2026-01-01 00:02:00")
    assert event.event_duration_seconds == 30.0


def test_temporary_recovery_region_dip_does_not_end_when_recovery_fails() -> None:
    events = detect_survival_fade_events(
        _frame([0, 11, 8, 12, 11, 8, 8, 8]),
        dataset_name="toy.csv",
        dataset_id="dataset_001",
        segment_id="segment_001",
        event_id_prefix="sp_event",
        config=_config(),
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_start_time == pd.Timestamp("2026-01-01 00:00:30")
    assert event.event_end_time == pd.Timestamp("2026-01-01 00:02:30")
    assert event.event_duration_seconds == 120.0


def test_hysteresis_uses_distinct_activation_and_recovery_thresholds() -> None:
    events = detect_survival_fade_events(
        _frame([0, 10.2, 9.8, 10.1, 9.4, 9.3, 9.2]),
        dataset_name="toy.csv",
        dataset_id="dataset_001",
        segment_id="segment_001",
        event_id_prefix="sp_event",
        config=_config(threshold_off=9.5),
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_start_time == pd.Timestamp("2026-01-01 00:00:30")
    assert event.event_end_time == pd.Timestamp("2026-01-01 00:02:00")
    assert event.event_end_confirmation_time >= event.event_end_time


def test_censored_event_when_segment_ends_before_recovery() -> None:
    events = detect_survival_fade_events(
        _frame([0, 11, 12, 11]),
        dataset_name="toy.csv",
        dataset_id="dataset_001",
        segment_id="segment_001",
        event_id_prefix="sp_event",
        config=_config(),
    )

    assert len(events) == 1
    event = events[0]
    assert not event.event_observed
    assert event.event_end_time is None
    assert event.event_end_confirmation_time is None
    assert event.censoring_reason == "segment_ended_before_recovery"


def test_censored_event_when_segment_ends_during_candidate_recovery() -> None:
    events = detect_survival_fade_events(
        _frame([0, 11, 8]),
        dataset_name="toy.csv",
        dataset_id="dataset_001",
        segment_id="segment_001",
        event_id_prefix="sp_event",
        config=_config(),
    )

    assert len(events) == 1
    event = events[0]
    assert not event.event_observed
    assert event.event_end_time is None
    assert event.event_end_confirmation_time is None
    assert event.censoring_reason == "insufficient_recovery_window_before_segment_end"


def test_state_machine_trace_exposes_recovery_confirmation_diagnostics() -> None:
    frame = _frame([0, 11, 12, 8, 8, 8])
    trace = trace_survival_state_machine(frame, config=_config())
    confirmed = trace.loc[trace["action"].eq("confirm_stable_recovery")]

    assert len(confirmed) == 1
    row = confirmed.iloc[0]
    assert row["state_before_update"] == "CANDIDATE_RECOVERY"
    assert row["state_after_update"] == "NORMAL"
    assert row["event_end_time"] == pd.Timestamp("2026-01-01 00:01:30")
    assert row["event_end_confirmation_time"] == pd.Timestamp("2026-01-01 00:02:30")
    assert row["recovery_window_start"] == pd.Timestamp("2026-01-01 00:01:30")
    assert row["recovery_window_end"] == pd.Timestamp("2026-01-01 00:02:30")
    assert row["num_recovery_observations"] == 3
    assert row["recovery_fraction"] == 1.0
