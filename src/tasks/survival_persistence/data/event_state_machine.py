"""State-based fade event detection for survival-persistence labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ThresholdOperator = Literal[">", ">=", "<", "<="]
RecoveryFractionMode = Literal["sample_fraction"]

CENSOR_ACTIVE = "segment_ended_before_recovery"
CENSOR_CANDIDATE = "segment_ended_during_candidate_recovery"
CENSOR_INSUFFICIENT_WINDOW = "insufficient_recovery_window_before_segment_end"


@dataclass(frozen=True)
class StableRecoveryConfig:
    """Configuration for state-based stable-recovery event detection."""

    threshold_on: float
    threshold_off: float
    threshold_on_operator: ThresholdOperator = ">"
    threshold_off_operator: ThresholdOperator = "<="
    recovery_window_seconds: float = 300.0
    recovery_required_fraction: float = 0.8
    recovery_fraction_mode: RecoveryFractionMode = "sample_fraction"
    minimum_recovery_observations: int = 2
    abort_candidate_recovery_on_threshold_on_crossing: bool = True


@dataclass(frozen=True)
class SurvivalFadeEvent:
    """One survival-persistence fade episode in one continuous segment."""

    dataset_name: str
    dataset_id: str
    segment_id: str
    event_id: str
    start_index: int
    sample_end_index: int
    event_start_time: pd.Timestamp
    event_end_time: pd.Timestamp | None
    event_end_confirmation_time: pd.Timestamp | None
    event_observed: bool
    censoring_reason: str
    segment_end_time: pd.Timestamp
    observed_followup_seconds: float
    event_duration_seconds: float | None
    event_duration_samples: int | None


def threshold_condition(
    values: np.ndarray | pd.Series | list[float],
    *,
    threshold: float,
    operator: ThresholdOperator,
) -> np.ndarray:
    """Return a boolean mask for a configured threshold condition."""

    array = np.asarray(values, dtype=float)
    if operator == ">":
        return array > float(threshold)
    if operator == ">=":
        return array >= float(threshold)
    if operator == "<":
        return array < float(threshold)
    if operator == "<=":
        return array <= float(threshold)
    raise ValueError(f"Unsupported threshold operator: {operator!r}")


def validate_recovery_config(config: StableRecoveryConfig) -> None:
    """Validate a stable-recovery event-detection configuration."""

    if config.threshold_on_operator not in {">", ">="}:
        raise ValueError("threshold_on_operator must be '>' or '>='.")
    if config.threshold_off_operator not in {"<", "<="}:
        raise ValueError("threshold_off_operator must be '<' or '<='.")
    if config.recovery_window_seconds <= 0:
        raise ValueError("recovery_window_seconds must be positive.")
    if not 0.0 < config.recovery_required_fraction <= 1.0:
        raise ValueError("recovery_required_fraction must lie in (0, 1].")
    if config.recovery_fraction_mode != "sample_fraction":
        raise ValueError("Only recovery_fraction_mode=sample_fraction is supported.")
    if config.minimum_recovery_observations < 1:
        raise ValueError("minimum_recovery_observations must be at least one.")
    if _activation_and_recovery_can_overlap(config):
        raise ValueError(
            "Activation and recovery conditions overlap. Use non-overlapping "
            "thresholds/operators, for example threshold_on > threshold_off or "
            "threshold_on_operator='>' with threshold_off_operator='<='."
        )


def _activation_and_recovery_can_overlap(config: StableRecoveryConfig) -> bool:
    """Return true when one value can satisfy both on and off conditions."""

    on = float(config.threshold_on)
    off = float(config.threshold_off)
    if off > on:
        return True
    if off < on:
        return False
    # Equal thresholds are safe only when there is no value satisfying both.
    return config.threshold_on_operator == ">=" and config.threshold_off_operator == "<="


def detect_survival_fade_events(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    dataset_id: str,
    segment_id: str,
    event_id_prefix: str,
    config: StableRecoveryConfig,
) -> list[SurvivalFadeEvent]:
    """Detect state-machine fade events in one continuous acquisition segment."""

    validate_recovery_config(config)
    required = {"Time", "Signal"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Signal frame is missing columns: {missing}")

    ordered = frame[["Time", "Signal"]].sort_values("Time", kind="stable").reset_index(
        drop=True
    )
    if ordered.empty:
        return []
    if ordered["Time"].duplicated().any():
        raise ValueError("Continuous segment contains duplicated timestamps.")
    signal = ordered["Signal"].to_numpy(dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values.")
    times = pd.to_datetime(ordered["Time"], errors="raise").reset_index(drop=True)
    is_on = threshold_condition(
        signal,
        threshold=config.threshold_on,
        operator=config.threshold_on_operator,
    )
    is_off = threshold_condition(
        signal,
        threshold=config.threshold_off,
        operator=config.threshold_off_operator,
    )

    events: list[SurvivalFadeEvent] = []
    state = "NORMAL"
    start_index: int | None = None
    candidate_index: int | None = None
    index = 0

    def next_event_id() -> str:
        return f"{event_id_prefix}_{len(events) + 1:05d}"

    def append_observed(candidate_start: int, confirmation_index: int) -> None:
        assert start_index is not None
        event_start = pd.Timestamp(times.iloc[start_index])
        event_end = pd.Timestamp(times.iloc[candidate_start])
        confirmation = pd.Timestamp(times.iloc[confirmation_index])
        sample_end = candidate_start - 1
        duration = float((event_end - event_start).total_seconds())
        events.append(
            SurvivalFadeEvent(
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                segment_id=segment_id,
                event_id=next_event_id(),
                start_index=int(start_index),
                sample_end_index=int(sample_end),
                event_start_time=event_start,
                event_end_time=event_end,
                event_end_confirmation_time=confirmation,
                event_observed=True,
                censoring_reason="",
                segment_end_time=pd.Timestamp(times.iloc[-1]),
                observed_followup_seconds=duration,
                event_duration_seconds=duration,
                event_duration_samples=max(1, int(candidate_start - start_index)),
            )
        )

    def append_censored(reason: str) -> None:
        assert start_index is not None
        event_start = pd.Timestamp(times.iloc[start_index])
        segment_end = pd.Timestamp(times.iloc[-1])
        sample_end = len(times) - 2
        followup = max(0.0, float((segment_end - event_start).total_seconds()))
        events.append(
            SurvivalFadeEvent(
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                segment_id=segment_id,
                event_id=next_event_id(),
                start_index=int(start_index),
                sample_end_index=int(sample_end),
                event_start_time=event_start,
                event_end_time=None,
                event_end_confirmation_time=None,
                event_observed=False,
                censoring_reason=reason,
                segment_end_time=segment_end,
                observed_followup_seconds=followup,
                event_duration_seconds=None,
                event_duration_samples=None,
            )
        )

    while index < len(times):
        if state == "NORMAL":
            if is_on[index]:
                start_index = index
                candidate_index = None
                state = "ACTIVE_FADE"
            index += 1
            continue

        if state == "ACTIVE_FADE":
            if is_off[index]:
                candidate_index = index
                state = "CANDIDATE_RECOVERY"
            index += 1
            continue

        if state == "CANDIDATE_RECOVERY":
            assert candidate_index is not None
            candidate_time = pd.Timestamp(times.iloc[candidate_index])
            current_time = pd.Timestamp(times.iloc[index])

            if (
                config.abort_candidate_recovery_on_threshold_on_crossing
                and index > candidate_index
                and is_on[index]
            ):
                candidate_index = None
                state = "ACTIVE_FADE"
                index += 1
                continue

            window_end = candidate_time + pd.Timedelta(
                seconds=float(config.recovery_window_seconds)
            )
            if current_time >= window_end:
                eligible = (times >= candidate_time) & (times <= window_end)
                eligible_indices = np.flatnonzero(eligible.to_numpy(dtype=bool))
                if len(eligible_indices) >= config.minimum_recovery_observations:
                    recovery_fraction = float(is_off[eligible_indices].mean())
                    on_crossed = bool(is_on[candidate_index : index + 1].any())
                    if (
                        recovery_fraction >= config.recovery_required_fraction
                        and not (
                            config.abort_candidate_recovery_on_threshold_on_crossing
                            and on_crossed
                        )
                    ):
                        append_observed(candidate_index, index)
                        state = "NORMAL"
                        start_index = None
                        candidate_index = None
                        index += 1
                        continue
                candidate_index = None
                state = "ACTIVE_FADE"
            index += 1
            continue

        raise RuntimeError(f"Unsupported state: {state}")

    if state == "ACTIVE_FADE" and start_index is not None:
        append_censored(CENSOR_ACTIVE)
    elif state == "CANDIDATE_RECOVERY" and start_index is not None:
        assert candidate_index is not None
        candidate_time = pd.Timestamp(times.iloc[candidate_index])
        segment_end = pd.Timestamp(times.iloc[-1])
        if segment_end < candidate_time + pd.Timedelta(
            seconds=float(config.recovery_window_seconds)
        ):
            append_censored(CENSOR_INSUFFICIENT_WINDOW)
        else:
            append_censored(CENSOR_CANDIDATE)

    return events


def trace_survival_state_machine(
    frame: pd.DataFrame,
    *,
    config: StableRecoveryConfig,
) -> pd.DataFrame:
    """Return row-level state transitions for auditing one continuous segment.

    The trace mirrors :func:`detect_survival_fade_events` and is intended for
    dataset validation only. It records the state before and after each timestamp
    update together with candidate-recovery diagnostics.
    """

    validate_recovery_config(config)
    required = {"Time", "Signal"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Signal frame is missing columns: {missing}")

    ordered = frame[["Time", "Signal"]].sort_values("Time", kind="stable").reset_index(
        drop=True
    )
    if ordered.empty:
        return pd.DataFrame()
    if ordered["Time"].duplicated().any():
        raise ValueError("Continuous segment contains duplicated timestamps.")
    signal = ordered["Signal"].to_numpy(dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values.")
    times = pd.to_datetime(ordered["Time"], errors="raise").reset_index(drop=True)
    is_on = threshold_condition(
        signal,
        threshold=config.threshold_on,
        operator=config.threshold_on_operator,
    )
    is_off = threshold_condition(
        signal,
        threshold=config.threshold_off,
        operator=config.threshold_off_operator,
    )

    rows: list[dict[str, object]] = []
    state = "NORMAL"
    start_index: int | None = None
    candidate_index: int | None = None
    event_start_time: pd.Timestamp | None = None
    event_end_time: pd.Timestamp | None = None
    event_confirmation_time: pd.Timestamp | None = None
    event_observed = False
    censoring_reason = ""
    index = 0

    def append_row(
        *,
        position: int,
        state_before: str,
        state_after: str,
        action: str,
        recovery_fraction: float | None = None,
        recovery_window_start: pd.Timestamp | None = None,
        recovery_window_end: pd.Timestamp | None = None,
        num_recovery_observations: int | None = None,
    ) -> None:
        rows.append(
            {
                "timestamp": pd.Timestamp(times.iloc[position]),
                "Signal": float(signal[position]),
                "state_before_update": state_before,
                "threshold_on_condition": bool(is_on[position]),
                "threshold_off_condition": bool(is_off[position]),
                "state_after_update": state_after,
                "action": action,
                "candidate_recovery_start": pd.Timestamp(times.iloc[candidate_index])
                if candidate_index is not None
                else pd.NaT,
                "recovery_window_start": recovery_window_start,
                "recovery_window_end": recovery_window_end,
                "num_recovery_observations": num_recovery_observations,
                "recovery_fraction": recovery_fraction,
                "event_start_time": event_start_time,
                "event_end_time": event_end_time,
                "event_end_confirmation_time": event_confirmation_time,
                "event_observed": event_observed,
                "censoring_reason": censoring_reason,
            }
        )

    while index < len(times):
        state_before = state

        if state == "NORMAL":
            action = "none"
            if is_on[index]:
                start_index = index
                candidate_index = None
                event_start_time = pd.Timestamp(times.iloc[index])
                event_end_time = None
                event_confirmation_time = None
                event_observed = False
                censoring_reason = ""
                state = "ACTIVE_FADE"
                action = "activate_event"
            append_row(position=index, state_before=state_before, state_after=state, action=action)
            index += 1
            continue

        if state == "ACTIVE_FADE":
            action = "remain_active"
            if is_off[index]:
                candidate_index = index
                state = "CANDIDATE_RECOVERY"
                action = "start_candidate_recovery"
            append_row(position=index, state_before=state_before, state_after=state, action=action)
            index += 1
            continue

        if state == "CANDIDATE_RECOVERY":
            assert candidate_index is not None
            candidate_time = pd.Timestamp(times.iloc[candidate_index])
            current_time = pd.Timestamp(times.iloc[index])

            if (
                config.abort_candidate_recovery_on_threshold_on_crossing
                and index > candidate_index
                and is_on[index]
            ):
                candidate_index = None
                state = "ACTIVE_FADE"
                append_row(
                    position=index,
                    state_before=state_before,
                    state_after=state,
                    action="abort_candidate_on_threshold_on_crossing",
                )
                index += 1
                continue

            window_end = candidate_time + pd.Timedelta(
                seconds=float(config.recovery_window_seconds)
            )
            if current_time >= window_end:
                eligible = (times >= candidate_time) & (times <= window_end)
                eligible_indices = np.flatnonzero(eligible.to_numpy(dtype=bool))
                recovery_fraction = (
                    float(is_off[eligible_indices].mean())
                    if len(eligible_indices)
                    else np.nan
                )
                if len(eligible_indices) >= config.minimum_recovery_observations:
                    on_crossed = bool(is_on[candidate_index : index + 1].any())
                    if (
                        recovery_fraction >= config.recovery_required_fraction
                        and not (
                            config.abort_candidate_recovery_on_threshold_on_crossing
                            and on_crossed
                        )
                    ):
                        event_end_time = candidate_time
                        event_confirmation_time = current_time
                        event_observed = True
                        state = "NORMAL"
                        append_row(
                            position=index,
                            state_before=state_before,
                            state_after=state,
                            action="confirm_stable_recovery",
                            recovery_fraction=recovery_fraction,
                            recovery_window_start=candidate_time,
                            recovery_window_end=window_end,
                            num_recovery_observations=int(len(eligible_indices)),
                        )
                        start_index = None
                        candidate_index = None
                        index += 1
                        continue
                candidate_index = None
                state = "ACTIVE_FADE"
                append_row(
                    position=index,
                    state_before=state_before,
                    state_after=state,
                    action="candidate_recovery_failed",
                    recovery_fraction=recovery_fraction,
                    recovery_window_start=candidate_time,
                    recovery_window_end=window_end,
                    num_recovery_observations=int(len(eligible_indices)),
                )
                index += 1
                continue

            append_row(
                position=index,
                state_before=state_before,
                state_after=state,
                action="wait_for_recovery_window",
                recovery_window_start=candidate_time,
                recovery_window_end=window_end,
            )
            index += 1
            continue

        raise RuntimeError(f"Unsupported state: {state}")

    if state in {"ACTIVE_FADE", "CANDIDATE_RECOVERY"} and start_index is not None:
        if state == "ACTIVE_FADE":
            censoring_reason = CENSOR_ACTIVE
        else:
            assert candidate_index is not None
            candidate_time = pd.Timestamp(times.iloc[candidate_index])
            segment_end = pd.Timestamp(times.iloc[-1])
            if segment_end < candidate_time + pd.Timedelta(
                seconds=float(config.recovery_window_seconds)
            ):
                censoring_reason = CENSOR_INSUFFICIENT_WINDOW
            else:
                censoring_reason = CENSOR_CANDIDATE
        if rows:
            rows[-1]["event_observed"] = False
            rows[-1]["censoring_reason"] = censoring_reason
            rows[-1]["action"] = f"{rows[-1]['action']}|censor_event_at_segment_end"

    return pd.DataFrame(rows)
