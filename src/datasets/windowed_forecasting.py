"""No-leakage index construction for univariate Signal forecasting windows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowIndices:
    """Half-open input and target intervals for one forecasting sample."""

    input_start: int
    input_stop: int
    target_start: int
    target_stop: int


def build_window_indices(
    n_observations: int,
    context_length: int,
    prediction_length: int,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> list[WindowIndices]:
    """Build windows whose inputs and targets both lie inside `[start, stop)`.

    The target begins exactly after the input, so target values can never enter
    the corresponding input window. The eventual input array contains only the
    univariate ``Signal`` series.
    """

    stop = n_observations if stop is None else stop
    if n_observations < 0:
        raise ValueError("n_observations cannot be negative.")
    if context_length < 1 or prediction_length < 1:
        raise ValueError("Window lengths must be positive.")
    if stride < 1:
        raise ValueError("stride must be positive.")
    if not 0 <= start <= stop <= n_observations:
        raise ValueError("start and stop must define a valid observation interval.")

    first_target_start = start + context_length
    last_target_start = stop - prediction_length
    if first_target_start > last_target_start:
        return []

    return [
        WindowIndices(
            input_start=target_start - context_length,
            input_stop=target_start,
            target_start=target_start,
            target_stop=target_start + prediction_length,
        )
        for target_start in range(first_target_start, last_target_start + 1, stride)
    ]
