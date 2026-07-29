"""Summarize completed long-fade detection runs."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.long_fade_detection.utils.paths import (  # noqa: E402
    model_summary_dir,
)
from src.utils.config import save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    COMPARISON_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_results_index_dir,
    relative_project_path,
    upsert_index_row,
)

RUNS_ROOT = PROJECT_ROOT / "results/runs/long_fade_detection"
SELECTION_ID = "externalHoldout_test_fc_uplink_fade"
SUMMARY_ID = "long_fade_model_summary_thr10p0_dur300"
SUMMARY_COLUMNS = [
    "model_id",
    "run_id",
    "status",
    "val_auroc",
    "val_auprc",
    "val_precision_at_0p5",
    "val_recall_at_0p5",
    "val_f1_at_0p5",
    "val_balanced_accuracy_at_0p5",
    "test_auroc",
    "test_auprc",
    "test_precision_at_0p5",
    "test_recall_at_0p5",
    "test_f1_at_0p5",
    "test_balanced_accuracy_at_0p5",
    "event_test_auroc",
    "event_test_auprc",
    "event_test_f1_at_0p5",
    "event_test_recall_at_0p5",
    "n_train_samples",
    "n_val_samples",
    "n_test_samples",
    "n_train_events",
    "n_val_events",
    "n_test_events",
    "positive_rate_train",
    "positive_rate_val",
    "positive_rate_test",
]


def build_summary() -> pd.DataFrame:
    """Return one summary row per completed run."""

    rows = []
    if not RUNS_ROOT.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for metrics_path in sorted(RUNS_ROOT.glob("*/*/metrics/metrics_summary.csv")):
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        run_dir = metrics_path.parents[1]
        metadata_path = run_dir / "metadata.yaml"
        status = "complete" if metadata_path.exists() else "unknown"
        row = {
            "model_id": metrics.get("model_id"),
            "run_id": metrics.get("run_id"),
            "status": status,
            "val_auroc": metrics.get("val_auroc"),
            "val_auprc": metrics.get("val_auprc"),
            "val_precision_at_0p5": metrics.get("val_precision_at_0p5"),
            "val_recall_at_0p5": metrics.get("val_recall_at_0p5"),
            "val_f1_at_0p5": metrics.get("val_f1_at_0p5"),
            "val_balanced_accuracy_at_0p5": metrics.get(
                "val_balanced_accuracy_at_0p5"
            ),
            "test_auroc": metrics.get("test_auroc"),
            "test_auprc": metrics.get("test_auprc"),
            "test_precision_at_0p5": metrics.get("test_precision_at_0p5"),
            "test_recall_at_0p5": metrics.get("test_recall_at_0p5"),
            "test_f1_at_0p5": metrics.get("test_f1_at_0p5"),
            "test_balanced_accuracy_at_0p5": metrics.get(
                "test_balanced_accuracy_at_0p5"
            ),
            "event_test_auroc": metrics.get("event_test_auroc"),
            "event_test_auprc": metrics.get("event_test_auprc"),
            "event_test_f1_at_0p5": metrics.get("event_test_f1_at_0p5"),
            "event_test_recall_at_0p5": metrics.get("event_test_recall_at_0p5"),
            "n_train_samples": metrics.get("train_n_samples"),
            "n_val_samples": metrics.get("val_n_samples"),
            "n_test_samples": metrics.get("test_n_samples"),
            "n_train_events": None,
            "n_val_events": None,
            "n_test_events": None,
            "positive_rate_train": metrics.get("train_positive_rate"),
            "positive_rate_val": metrics.get("val_positive_rate"),
            "positive_rate_test": metrics.get("test_positive_rate"),
        }
        for split_name, source_name in [
            ("train", "train"),
            ("val", "val"),
            ("test", "test"),
        ]:
            pred_path = run_dir / "predictions" / f"{source_name}_predictions.parquet"
            if pred_path.exists():
                predictions = pd.read_parquet(pred_path)
                row[f"n_{split_name}_events"] = int(
                    predictions["global_event_id"].nunique()
                )
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def main() -> None:
    """Write CSV and Markdown summaries."""

    summary = build_summary()
    output_dir = model_summary_dir(selection_id=SELECTION_ID, summary_id=SUMMARY_ID)
    paths = ensure_results_subdirs(output_dir, ("tables", "reports"))
    csv_path = paths["tables"] / "long_fade_model_comparison.csv"
    md_path = paths["reports"] / "long_fade_model_comparison.md"
    summary.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write("# Long-Fade Detection Initial Model Comparison\n\n")
        stream.write(f"Rows: {len(summary)}\n\n")
        if summary.empty:
            stream.write("No completed runs found.\n")
        else:
            stream.write("```text\n")
            stream.write(summary.to_string(index=False))
            stream.write("\n```\n")
            stream.write("\n")
    created_at = datetime.now(timezone.utc).isoformat()
    save_yaml(
        output_dir / "metadata.yaml",
        {
            "comparison_id": SUMMARY_ID,
            "comparison_type": "model_summary",
            "task_name": "long_fade_detection",
            "selection_id": SELECTION_ID,
            "num_runs": int(len(summary)),
            "output_files": {
                "summary_table": relative_project_path(csv_path),
                "summary_report": relative_project_path(md_path),
            },
            "created_at": created_at,
        },
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": SUMMARY_ID,
            "method_id": "multiple",
            "reference_id": "",
            "comparison_type": "model_summary",
            "selection_id": SELECTION_ID,
            "results_path": relative_project_path(output_dir),
            "metrics_path": relative_project_path(paths["tables"]),
            "figures_path": "",
            "status": "complete",
            "created_at": created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )
    print(f"Wrote {len(summary)} rows to {relative_project_path(csv_path)}")
    print(f"Report: {relative_project_path(md_path)}")


if __name__ == "__main__":
    main()
