"""Summarize completed current-level persistence learnable-shapelet runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.current_level_persistence.utils.paths import run_dir, run_index_path  # noqa: E402
from src.utils.config import load_yaml_config  # noqa: E402

DATASET_PATH = (
    PROJECT_ROOT
    / "data/processed/current_level_persistence/delta_0p5/L30/"
    / "externalHoldout_test_fc_uplink_fade"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/comparisons/model_summary/current_level_persistence/"
    / "externalHoldout_test_fc_uplink_fade/"
    / "initial_shapelet_model_comparison_delta0p5/tables"
)
OUTPUT_PATH = OUTPUT_DIR / "initial_shapelet_model_comparison_delta0p5.csv"
COMPARISON_COLUMNS = [
    "model_id",
    "run_id",
    "best_epoch",
    "best_val_mae_seconds",
    "best_val_rmse_seconds",
    "best_val_median_ae_seconds",
    "test_mae_seconds",
    "test_rmse_seconds",
    "test_median_ae_seconds",
    "test_mae_minutes",
    "test_rmse_minutes",
    "test_median_ae_minutes",
    "n_train_samples",
    "n_val_samples",
    "n_test_samples",
    "created_at",
]


def _load_dataset_counts() -> dict[str, int | None]:
    if not (DATASET_PATH / "dataset_metadata.yaml").exists():
        return {
            "n_train_samples": None,
            "n_val_samples": None,
            "n_test_samples": None,
        }
    metadata = load_yaml_config(DATASET_PATH / "dataset_metadata.yaml")
    return {
        "n_train_samples": metadata.get("num_train_windows"),
        "n_val_samples": metadata.get("num_val_windows"),
        "n_test_samples": metadata.get("num_test_windows"),
    }


def _read_metrics_summary(run_id: str) -> dict:
    path = run_dir(run_id) / "metrics" / "metrics_summary.csv"
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def build_summary() -> pd.DataFrame:
    """Return one comparison row per completed current-level persistence run."""

    index_path = run_index_path()
    if not index_path.exists():
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    runs = pd.read_csv(index_path)
    if runs.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    counts = _load_dataset_counts()
    rows = []
    for _, run in runs.iterrows():
        if run.get("status") != "complete":
            continue
        metrics = _read_metrics_summary(str(run["run_id"]))
        row = {
            "model_id": run.get("model_id"),
            "run_id": run.get("run_id"),
            "best_epoch": run.get("best_epoch"),
            "best_val_mae_seconds": run.get("best_val_mae_seconds"),
            "best_val_rmse_seconds": run.get("best_val_rmse_seconds"),
            "best_val_median_ae_seconds": metrics.get("val_median_ae_seconds"),
            "test_mae_seconds": run.get("test_mae_seconds"),
            "test_rmse_seconds": run.get("test_rmse_seconds"),
            "test_median_ae_seconds": run.get("test_median_ae_seconds"),
            "test_mae_minutes": metrics.get("test_mae_minutes"),
            "test_rmse_minutes": metrics.get("test_rmse_minutes"),
            "test_median_ae_minutes": metrics.get("test_median_ae_minutes"),
            **counts,
            "created_at": run.get("created_at"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def main() -> None:
    """Write the comparison table."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(summary)} rows to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
