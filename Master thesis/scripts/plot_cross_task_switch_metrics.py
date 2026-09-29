"""Split the four original switch-metric panels into two vertical pairs."""

import os
from pathlib import Path
import sys

THESIS = Path(__file__).resolve().parents[1]
ROOT = THESIS.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.switching.diagnostic_plots import TASK_COLORS, compact_method_label


def main() -> None:
    """Reuse saved metrics, original names, colors, and F1 ranking."""
    selection = "externalHoldout_test_fc_uplink_fade"
    source = ROOT / "results/comparisons/cross_task_switch" / selection / (
        f"crossTask_switch_{selection}_referenceGrid/tables/"
        "cross_task_switch_metrics_reference_grid.csv"
    )
    frame = pd.read_csv(source).sort_values("f1", ascending=True)
    assert len(frame) == 18 and frame.method_id.is_unique
    labels = [compact_method_label(row) for _, row in frame.iterrows()]
    colors = [TASK_COLORS[task] for task in frame.task_name]
    y = np.arange(len(frame))
    pairs = [
        (("f1", "precision"), "cross_task_reference_grid_f1_precision.png"),
        (("recall", "balanced_accuracy"), "cross_task_reference_grid_recall_balanced_accuracy.png"),
    ]
    for metrics, filename in pairs:
        figure, axes = plt.subplots(2, 1, figsize=(7.5, 7.8), sharey=True)
        for axis, metric in zip(axes, metrics, strict=True):
            axis.barh(y, frame[metric].to_numpy(float), color=colors, alpha=0.85)
            axis.set_title(metric)
            axis.set_xlim(0, 1.05)
            axis.set_yticks(y, labels, fontsize=8)
            axis.grid(axis="x", linestyle="--", alpha=0.4)
        figure.suptitle(
            "Cross-task core switch metrics | reference_grid",
            fontsize=13, fontweight="bold",
        )
        figure.tight_layout(h_pad=1.5)
        destination = THESIS / "figures/results" / filename
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(destination)


if __name__ == "__main__":
    main()
