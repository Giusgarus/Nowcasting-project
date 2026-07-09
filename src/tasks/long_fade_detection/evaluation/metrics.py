"""Classification metrics for long-fade detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLDS = (0.3, 0.5, 0.7)


def threshold_token(threshold: float) -> str:
    """Return a compact metric suffix for a probability threshold."""

    return str(float(threshold)).replace(".", "p")


def _safe_auroc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def _safe_auprc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, prob))


def threshold_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> list[dict[str, float | int]]:
    """Return confusion and precision/recall/F1 at configured thresholds."""

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(prob, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and prob must have matching lengths.")
    rows = []
    for threshold in thresholds:
        pred = (p >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )
    return rows


def sample_classification_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, float | int]:
    """Compute split-level classification metrics."""

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(prob, dtype=float).reshape(-1)
    if len(y) == 0:
        raise ValueError("Metric inputs cannot be empty.")
    if len(y) != len(p):
        raise ValueError("y_true and prob must have matching lengths.")
    pred_05 = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_05, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "auroc": _safe_auroc(y, p),
        "auprc": _safe_auprc(y, p),
        "accuracy": float(accuracy_score(y, pred_05)),
        "balanced_accuracy_at_0p5": float(balanced_accuracy_score(y, pred_05)),
        "precision_at_0p5": float(precision_score(y, pred_05, zero_division=0)),
        "recall_at_0p5": float(recall_score(y, pred_05, zero_division=0)),
        "f1_at_0p5": float(f1_score(y, pred_05, zero_division=0)),
        "specificity_at_0p5": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "brier_score": float(brier_score_loss(y, np.clip(p, 0.0, 1.0))),
        "true_negatives_at_0p5": int(tn),
        "false_positives_at_0p5": int(fp),
        "false_negatives_at_0p5": int(fn),
        "true_positives_at_0p5": int(tp),
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
    }
    for row in threshold_metrics(y, p, thresholds=thresholds):
        suffix = threshold_token(float(row["threshold"]))
        for key, value in row.items():
            if key == "threshold":
                continue
            metrics[f"{key}_at_{suffix}"] = value
    return metrics


def event_probability_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate timestamp probabilities to one row per grouped event."""

    required = {"global_event_id", "y_true", "prob_long_fade"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table missing columns: {missing}")
    rows = []
    for event_id, frame in predictions.groupby("global_event_id", sort=False):
        ordered = frame.sort_values("position_in_fade_samples", kind="stable")
        probs = ordered["prob_long_fade"].to_numpy(dtype=float)
        early = probs[: min(10, len(probs))]
        rows.append(
            {
                "global_event_id": event_id,
                "dataset_name": ordered["dataset_name"].iloc[0],
                "event_id": ordered["event_id"].iloc[0],
                "y_true": int(ordered["y_true"].iloc[0]),
                "mean_prob": float(np.mean(probs)),
                "max_prob": float(np.max(probs)),
                "last_prob": float(probs[-1]),
                "early_mean_prob": float(np.mean(early)),
                "num_samples": int(len(ordered)),
                "event_duration_seconds": float(
                    ordered["event_duration_seconds"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def event_level_metrics(
    predictions: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    """Compute event-level metrics for each probability aggregation method."""

    events = event_probability_table(predictions)
    if events.empty:
        return pd.DataFrame()
    rows = []
    for prob_column in ["mean_prob", "max_prob", "last_prob", "early_mean_prob"]:
        y = events["y_true"].to_numpy(dtype=int)
        p = events[prob_column].to_numpy(dtype=float)
        base = {
            "aggregation": prob_column,
            "auroc": _safe_auroc(y, p),
            "auprc": _safe_auprc(y, p),
            "n_events": int(len(events)),
            "positive_event_rate": float(y.mean()),
        }
        for row in threshold_metrics(y, p, thresholds=thresholds):
            output = dict(base)
            output.update(row)
            rows.append(output)
    return pd.DataFrame(rows)
