"""Tests for autoregressive window index construction."""

from src.tasks.autoregressive.data.windowed_forecasting import build_window_indices


def test_builds_no_leakage_windows() -> None:
    windows = build_window_indices(
        n_observations=8,
        context_length=3,
        prediction_length=2,
    )

    assert len(windows) == 4
    assert (
        windows[0].input_start,
        windows[0].input_stop,
        windows[0].target_start,
        windows[0].target_stop,
    ) == (0, 3, 3, 5)
    assert all(window.input_stop == window.target_start for window in windows)
    assert windows[-1].target_stop == 8


def test_respects_split_boundaries() -> None:
    windows = build_window_indices(
        n_observations=20,
        context_length=4,
        prediction_length=2,
        start=10,
        stop=20,
    )

    assert all(window.input_start >= 10 for window in windows)
    assert all(window.target_stop <= 20 for window in windows)


def test_returns_empty_list_for_short_interval() -> None:
    windows = build_window_indices(4, context_length=3, prediction_length=2)

    assert windows == []
