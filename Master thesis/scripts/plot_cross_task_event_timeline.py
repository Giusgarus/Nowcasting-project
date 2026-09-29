"""Render full-width switch tracks with model names above their bands."""

import os
from pathlib import Path
import sys

THESIS = Path(__file__).resolve().parents[1]
ROOT = THESIS.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.switching import diagnostic_plots as plots
from src.switching.cross_task import load_reference_grid


def draw_timeline(event: pd.DataFrame, switches: dict, raw: dict, path: Path) -> None:
    """Keep the original track order and colors without a wide label column."""
    figure, (signal_axis, switch_axis) = plt.subplots(
        2, 1, figsize=(7.5, 9.7), sharex=True,
        gridspec_kw={"height_ratios": [0.9, 8.1]},
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.055, top=0.96, hspace=0.10)
    times = pd.to_datetime(event.Time)
    signal_axis.plot(times, event.Signal_true, color="gray", linewidth=1.0, label="Signal")
    signal_axis.axhline(10, color="orangered", linewidth=1.2, label="10 threshold")
    signal_axis.set_ylabel("Signal", fontsize=10)
    signal_axis.legend(loc="upper right", fontsize=8)
    signal_axis.grid(True, linestyle="--", alpha=0.35)
    signal_axis.set_title("Cross-task switch timeline | dataset_007_event_00007", fontsize=11, fontweight="bold")
    row_spacing = 2.5
    for row, (name, values) in enumerate(switches.items()):
        position = row * row_spacing
        switch_axis.fill_between(
            times, position - 0.35, position + 0.35,
            where=np.asarray(values) > 0, step="post", alpha=0.55,
            color="#444444" if name == "Perfect Switch" else "#1f77b4",
        )
        if name in raw:
            active = np.asarray(raw[name]) > 0
            switch_axis.scatter(
                times[active], np.full(int(active.sum()), position + 0.42),
                marker="|", color="black", s=18, linewidths=0.8,
            )
        switch_axis.text(
            0.005, position + 1.10, name,
            transform=switch_axis.get_yaxis_transform(), fontsize=9.5,
            va="bottom", ha="left", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
        )
    switch_axis.set_ylim(-0.65, (len(switches) - 1) * row_spacing + 2.20)
    switch_axis.set_yticks([])
    switch_axis.set_xlim(times.iloc[0], times.iloc[-1])
    switch_axis.grid(axis="x", linestyle="--", alpha=0.35)
    switch_axis.set_title("Filled bands = post-processed switch; black ticks = raw switch", fontsize=9)
    switch_axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    switch_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    switch_axis.set_xlabel(f"Time ({times.iloc[0]:%Y-%m-%d})", fontsize=10)
    for axis in (signal_axis, switch_axis):
        axis.tick_params(labelsize=9)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    """Map existing switch outputs onto the same event without new decisions."""
    selection = "externalHoldout_test_fc_uplink_fade"
    reference = load_reference_grid(ROOT / (
        "results/switching/perfect_switch/autoregressive/"
        f"{selection}_L30_h10_thr10/tables/perfect_switch_timeseries.parquet"
    ))
    event = reference.loc[reference.event_id.eq("dataset_007_event_00007")].copy()
    assert not event.empty and event.Time.is_monotonic_increasing
    manifest = pd.read_csv(ROOT / "results/comparisons/cross_task_switch" / selection / (
        f"crossTask_switch_{selection}_referenceGrid/tables/cross_task_method_manifest.csv"
    ))
    switches = {"Perfect Switch": event.perfect_switch.to_numpy()}
    raw = {}
    for _, source in manifest.iterrows():
        predictions = pd.read_parquet(ROOT / source.predictions_path)
        name = plots.compact_method_label(source)
        switches[name] = plots.map_switch_to_times(event.Time, predictions, "model_switch")
        raw[name] = plots.map_switch_to_times(event.Time, predictions, "model_switch_raw")
    assert len(switches) == 19, "Expected all 18 models and Perfect Switch."
    destination = THESIS / "figures/results/cross_task_event00007_all_methods_switches.png"
    draw_timeline(event, switches, raw, destination)
    print(destination)


if __name__ == "__main__":
    main()
