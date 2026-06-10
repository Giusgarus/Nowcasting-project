"""Sample-level metrics for binary operational switch decisions."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SwitchMetrics:
    """Core sample-level metrics against a reference switch."""

    active_duration_seconds: float
    precision: float
    recall: float
    f1: float
    intersection_over_union: float
    balanced_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


def _safe_divide(numerator: float, denominator: float, zero_division: float) -> float:
    return numerator / denominator if denominator else zero_division


def compute_switch_metrics(
    reference: Sequence[int | bool],
    predicted: Sequence[int | bool],
    *,
    sample_interval_seconds: float = 1.0,
    zero_division: float = 0.0,
) -> SwitchMetrics:
    """Compute binary switch metrics against a Perfect Switch reference."""

    if len(reference) == 0:
        raise ValueError("Metric inputs cannot be empty.")
    if len(reference) != len(predicted):
        raise ValueError("Metric inputs must have the same length.")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive.")
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for actual, forecast in zip(reference, predicted, strict=True):
        if actual not in (0, 1, False, True) or forecast not in (0, 1, False, True):
            raise ValueError("Switch values must be binary.")
        if bool(actual) and bool(forecast):
            true_positives += 1
        elif not bool(actual) and not bool(forecast):
            true_negatives += 1
        elif bool(forecast):
            false_positives += 1
        else:
            false_negatives += 1

    precision = _safe_divide(
        true_positives,
        true_positives + false_positives,
        zero_division,
    )
    recall = _safe_divide(
        true_positives,
        true_positives + false_negatives,
        zero_division,
    )
    specificity = _safe_divide(
        true_negatives,
        true_negatives + false_positives,
        zero_division,
    )

    return SwitchMetrics(
        active_duration_seconds=sum(bool(value) for value in predicted)
        * sample_interval_seconds,
        precision=precision,
        recall=recall,
        f1=_safe_divide(2 * precision * recall, precision + recall, zero_division),
        intersection_over_union=_safe_divide(
            true_positives,
            true_positives + false_positives + false_negatives,
            zero_division,
        ),
        balanced_accuracy=(recall + specificity) / 2,
        false_positive_rate=_safe_divide(
            false_positives,
            false_positives + true_negatives,
            zero_division,
        ),
        false_negative_rate=_safe_divide(
            false_negatives,
            false_negatives + true_positives,
            zero_division,
        ),
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
