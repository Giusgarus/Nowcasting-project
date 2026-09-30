"""Read existing artifacts for the defense; never rerun or overwrite experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.switching.cross_task import discover_switch_sources, load_reference_grid, read_method_frame
from src.switching.diagnostic_plots import map_switch_to_times
from src.switching.conversion import ensure_min_island_length, hold_while_signal_above_threshold
from src.utils.config import load_yaml_config


def main() -> None:
    """Export presentation data, compact Beamer tables, and source provenance."""
    config = load_yaml_config(ROOT / "configs/cross_task_switch_comparison.yaml")
    reference = load_reference_grid(ROOT / config["reference"]["perfect_switch_timeseries_path"])
    event = reference.loc[reference.event_id.eq("dataset_007_event_00007")].copy()
    if event.empty:
        raise ValueError("The approved common-grid example event is unavailable.")
    sources = discover_switch_sources(
        pd.read_csv(ROOT / config["sources"]["comparison_index_path"]),
        project_root=ROOT,
        include_sources=config["sources"]["include"],
    )
    representatives = [
        ("current_level_persistence", "XGBoost Scalar Context", "Current-level XGBoost"),
        ("autoregressive", "PatchTST Raw", "PatchTST raw"),
        ("survival_persistence", "XGBoost AFT Raw Flattened", "Survival XGBoost-AFT"),
        ("long_fade_detection", "TCN Classifier", "Long-fade TCN"),
    ]
    out = ROOT / "Master thesis/defense/build"
    out.mkdir(parents=True, exist_ok=True)
    comparison = (
        ROOT / config["output"]["results_root"]
        / config["comparison"]["normalized_selection_id"]
        / config["comparison"]["comparison_id"]
    )
    metrics = pd.read_csv(comparison / "tables/cross_task_switch_metrics_reference_grid.csv")
    switches = {"Perfect Switch": event.perfect_switch.tolist()}
    raw_switches = {}
    records = []
    source_paths = []
    for task, name, label in representatives:
        matches = [s for s in sources if s.task_name == task and s.method_name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one comparison for {task}/{name}, got {len(matches)}")
        source = matches[0]
        frame = read_method_frame(source)
        switches[label] = map_switch_to_times(event.Time, frame, "model_switch", fill_value=0).tolist()
        raw_switches[label] = map_switch_to_times(event.Time, frame, "model_switch_raw", fill_value=0).tolist()
        row = metrics.loc[metrics.comparison_id.eq(source.comparison_id)].iloc[0]
        records.append({
            "label": label,
            "task": task,
            "coverage": float(row.coverage_pct),
            "precision": float(row.precision),
            "recall": float(row.recall),
            "f1": float(row.f1),
            "minutes": float(row.active_duration_seconds) / 60,
        })
        source_paths.append(str(source.predictions_path.relative_to(ROOT)))

    # Keep only three actual TCN curves, selected across predicted S(300s).
    candidates = sorted((ROOT / "results/runs/survival_persistence").glob(
        "**/*grid_extended_best/predictions/test_predictions.parquet"
    ))
    if len(candidates) != 1:
        raise ValueError(f"Expected one canonical TCN prediction file, found {len(candidates)}")
    predictions = pd.read_parquet(candidates[0])
    probability_columns = sorted(
        [c for c in predictions if c.startswith("survival_probability_") and c.endswith("s")],
        key=lambda c: float(c.removeprefix("survival_probability_").removesuffix("s")),
    )
    horizon_seconds = [float(c.removeprefix("survival_probability_").removesuffix("s")) for c in probability_columns]
    ordered = predictions.sort_values("survival_probability_300s", kind="stable")
    curves = []
    for q in (0.05, 0.50, 0.95):
        row = ordered.iloc[round(q * (len(ordered) - 1))]
        curves.append({
            "quantile": q, "window_id": str(row.global_window_id),
            "s300": float(row.survival_probability_300s),
            "values": [float(row[c]) for c in probability_columns],
        })

    raw = np.zeros(15, dtype=int)
    raw[2:4] = 1
    signal = np.full(15, 9.0)
    signal[2:14] = 11.0
    extended = ensure_min_island_length(raw, 10)
    held = hold_while_signal_above_threshold(extended, signal >= 10)
    data = {
        "event_id": "dataset_007_event_00007",
        "date": str(event.Time.iloc[0].date()),
        "time_labels": event.Time.dt.strftime("%H:%M").tolist(),
        "minutes": ((event.Time - event.Time.iloc[0]).dt.total_seconds() / 60).tolist(),
        "signal": event.Signal_true.tolist(), "switches": switches, "raw_switches": raw_switches,
        "metrics": records,
        "perfect_minutes": float(metrics.iloc[0].num_perfect_positive) * 30 / 60,
        "survival": {"horizons": horizon_seconds, "curves": curves},
        "postprocessing": {"signal": signal.tolist(), "raw": raw.tolist(), "extended": extended.tolist(), "held": held.tolist()},
        "sources": source_paths + [str(candidates[0].relative_to(ROOT)), str((comparison / "tables/cross_task_switch_metrics_reference_grid.csv").relative_to(ROOT))],
    }
    destination = out / "slide_data.json"
    destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Small, self-contained plotting tables for the Beamer source.
    tables = out.parent / "data"
    tables.mkdir(exist_ok=True)
    event_table = pd.DataFrame({"minutes": data["minutes"], "signal": data["signal"]})
    for column, label in zip(("perfect", "current", "forecast", "survival", "longfade"), switches):
        event_table[column] = switches[label]
    event_table.to_csv(tables / "event.csv", index=False, float_format="%.6f")
    survival_table = pd.DataFrame({"minutes": np.array(horizon_seconds) / 60})
    for number, curve in enumerate(curves, 1):
        survival_table[f"curve{number}"] = curve["values"]
    survival_table.to_csv(tables / "survival.csv", index=False, float_format="%.6f")
    pd.DataFrame({"sample": np.arange(1, 16), **data["postprocessing"]}).to_csv(
        tables / "postprocessing.csv", index=False
    )
    (tables / "provenance.json").write_text(json.dumps({
        "event_id": data["event_id"], "sources": data["sources"],
        "metrics": records, "survival_curves": curves,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved slide-only data: {destination.relative_to(ROOT)}")
    print(f"Event samples: {len(event)}; representatives: {len(records)}; survival curves: {len(curves)}")


if __name__ == "__main__":
    main()
