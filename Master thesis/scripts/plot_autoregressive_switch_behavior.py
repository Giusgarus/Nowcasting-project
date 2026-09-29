"""Render Figure 6.2 with the original styling and vertically stacked panels."""

import os
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1]
ROOT = THESIS.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    """Plot saved duration and island counts without recomputing any metrics."""
    selection = "externalHoldout_test_fc_uplink_fade_L30_h10_thr10"
    source = ROOT / "results/comparisons/model_summary/autoregressive" / selection / (
        f"model_result_summary_{selection}/tables/global_switch_behavior_comparison.csv"
    )
    order = [
        "Perfect Switch", "PatchTST Raw", "XGBoost Raw", "GRU Seq2Seq Raw",
        "Chronos Zero-Shot Raw", "XGBoost Context Standard", "GRU S2V Raw",
        "PatchTST Context Standard", "GRU S2V Context Standard", "GRU Seq2Seq Context Standard",
    ]
    frame = pd.read_csv(source).set_index("method").loc[order]
    colors = [
        "#d62728" if name == "Perfect Switch" else
        "#ff7f0e" if "Context" in name else
        "#9467bd" if "Chronos" in name else "#1f77b4"
        for name in frame.index
    ]
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 8.2), sharey=True)
    y = np.arange(len(frame))
    for axis, field, factor, xlabel, legend in zip(
        axes, ("duration_samples", "events"), (0.5, 1),
        ("Active switch duration (min)", "Number of switch islands"),
        ("Perfect duration", "Perfect events"), strict=True,
    ):
        axis.barh(y, frame[field].to_numpy(float) * factor, color=colors, alpha=0.88)
        axis.axvline(
            float(frame.loc["Perfect Switch", field]) * factor,
            color="#d62728", linestyle="--", linewidth=1.8, label=legend,
        )
        axis.set_xlabel(xlabel)
        axis.set_yticks(y, frame.index, fontsize=9)
        axis.grid(axis="x", linestyle="--", alpha=0.35)
        axis.legend(loc="lower right", fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle(
        "Autoregressive switch duration and event-count behaviour",
        fontsize=13, fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    destination = THESIS / "figures/results/autoregressive_switch_duration_events.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
