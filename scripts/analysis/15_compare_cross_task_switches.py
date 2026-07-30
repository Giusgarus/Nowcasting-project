"""Build cross-task switch comparisons from saved switch-evaluation artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.switching.cross_task import (  # noqa: E402
    build_reference_grid_metrics,
    build_strict_intersection_metrics,
    discover_switch_sources,
    load_reference_grid,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    COMPARISON_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_results_index_dir,
    relative_project_path,
    sanitize_id,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/cross_task_switch_comparison.yaml"


def project_path(value: str | Path) -> Path:
    """Resolve a possibly relative project path."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def markdown_table_block(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring optional Markdown dependencies."""

    if frame.empty:
        return "_No rows available._"
    return "```text\n" + frame.to_string(index=False) + "\n```"


def compact_metric_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Return report-facing columns while keeping full metrics in CSV files."""

    columns = [
        "task_name",
        "method_name",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "active_duration_seconds",
        "coverage_pct",
        "num_points",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].copy()


def main() -> None:
    """Run the configured cross-task switch comparison."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("comparison_type") != "cross_task_switch":
        raise ValueError("comparison_type must be cross_task_switch.")

    comparison = config["comparison"]
    comparison_id = sanitize_id(str(comparison["comparison_id"]))
    normalized_selection_id = sanitize_id(str(comparison["normalized_selection_id"]))
    output_root = project_path(config["output"]["results_root"])
    output_dir = output_root / normalized_selection_id / comparison_id
    if output_dir.exists() and any(output_dir.iterdir()) and not config["output"].get(
        "overwrite",
        False,
    ):
        raise FileExistsError(
            f"Output already exists: {output_dir}. Set output.overwrite=true to replace it."
        )
    output_paths = ensure_results_subdirs(output_dir, ("tables", "reports"))

    reference_path = project_path(config["reference"]["perfect_switch_timeseries_path"])
    comparisons_index_path = project_path(config["sources"]["comparison_index_path"])
    comparisons_index = pd.read_csv(comparisons_index_path)
    sources = discover_switch_sources(
        comparisons_index,
        project_root=PROJECT_ROOT,
        include_sources=list(config["sources"]["include"]),
    )
    if not sources:
        raise RuntimeError("No switch-evaluation sources matched the configuration.")

    reference_grid = load_reference_grid(reference_path)
    denominators = list(config["evaluation"]["denominators"])
    supported_denominators = {
        "reference_grid_missing_as_zero",
        "strict_common_timestamp_intersection",
    }
    unsupported = sorted(set(denominators) - supported_denominators)
    if unsupported:
        raise ValueError(f"Unsupported cross-task denominators: {unsupported}")
    reference_metrics = pd.DataFrame()
    coverage = pd.DataFrame()
    intersection_metrics = pd.DataFrame()
    if "reference_grid_missing_as_zero" in denominators:
        reference_metrics, coverage = build_reference_grid_metrics(
            reference_grid=reference_grid,
            sources=sources,
            missing_model_switch=int(config["evaluation"].get("missing_model_switch", 0)),
        )
    if "strict_common_timestamp_intersection" in denominators:
        intersection_metrics = build_strict_intersection_metrics(
            reference_grid=reference_grid,
            sources=sources,
        )

    manifest = pd.DataFrame(
        [
            {
                "task_name": source.task_name,
                "selection_id": source.selection_id,
                "comparison_id": source.comparison_id,
                "method_id": source.method_id,
                "method_name": source.method_name,
                "results_path": relative_project_path(source.results_path),
                "predictions_path": relative_project_path(source.predictions_path),
            }
            for source in sources
        ]
    )

    tables_dir = output_paths["tables"]
    manifest.to_csv(tables_dir / "cross_task_method_manifest.csv", index=False)
    coverage.to_csv(tables_dir / "cross_task_method_coverage.csv", index=False)
    reference_metrics.to_csv(
        tables_dir / "cross_task_switch_metrics_reference_grid.csv",
        index=False,
    )
    intersection_metrics.to_csv(
        tables_dir / "cross_task_switch_metrics_strict_intersection.csv",
        index=False,
    )

    report_path = output_paths["reports"] / "cross_task_switch_comparison.md"
    best_reference = (
        reference_metrics.sort_values("f1", ascending=False).head(10)
        if not reference_metrics.empty
        else pd.DataFrame()
    )
    best_intersection = (
        intersection_metrics.sort_values("f1", ascending=False).head(10)
        if not intersection_metrics.empty
        else pd.DataFrame()
    )
    with report_path.open("w", encoding="utf-8") as stream:
        stream.write("# Cross-Task Switch Comparison\n\n")
        stream.write(
            "This comparison uses saved per-task switch-evaluation artifacts only. "
            "It does not train models or recompute task-specific switches.\n\n"
        )
        stream.write("## Denominators\n\n")
        stream.write(
            "- `reference_grid_missing_as_zero`: evaluates every method on the "
            "common Perfect Switch reference grid. Missing native method "
            "decisions are treated as switch `0`, so this is coverage-aware.\n"
        )
        stream.write(
            "- `strict_common_timestamp_intersection`: evaluates only timestamps "
            "where every included method has a native decision, so this compares "
            "conditional decision quality but hides coverage differences.\n\n"
        )
        stream.write("## Top Reference-Grid F1\n\n")
        stream.write(markdown_table_block(compact_metric_view(best_reference)))
        stream.write("\n\n## Top Strict-Intersection F1\n\n")
        stream.write(markdown_table_block(compact_metric_view(best_intersection)))
        stream.write("\n")

    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "comparison_id": comparison_id,
        "comparison_type": "cross_task_switch",
        "normalized_selection_id": normalized_selection_id,
        "reference_grid_path": relative_project_path(reference_path),
        "comparisons_index_path": relative_project_path(comparisons_index_path),
        "num_reference_points": int(len(reference_grid)),
        "num_sources": int(len(sources)),
        "denominators": denominators,
        "missing_model_switch": int(config["evaluation"].get("missing_model_switch", 0)),
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "created_at": created_at,
    }
    save_yaml(output_dir / "metadata.yaml", metadata)
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": comparison_id,
            "method_id": "multiple",
            "reference_id": str(config["reference"]["selection_id"]),
            "comparison_type": "cross_task_switch",
            "selection_id": normalized_selection_id,
            "results_path": relative_project_path(output_dir),
            "metrics_path": relative_project_path(tables_dir),
            "figures_path": "",
            "status": "complete",
            "created_at": created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )

    print("=== Cross-Task Switch Comparison ===")
    print(f"Comparison ID: {comparison_id}")
    print(f"Reference points: {len(reference_grid):,}")
    print(f"Methods: {len(sources):,}")
    print(f"Output: {relative_project_path(output_dir)}")
    if not best_reference.empty:
        print("Top reference-grid F1:")
        print(
            best_reference[
                ["task_name", "method_name", "f1", "precision", "recall"]
            ].to_string(index=False)
        )
    if not best_intersection.empty:
        print("Top strict-intersection F1:")
        print(
            best_intersection[
                ["task_name", "method_name", "f1", "precision", "recall"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
