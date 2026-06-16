"""Dry-run first cleanup for generated dataset artifacts.

This script deliberately targets only reproducible generated artifacts under
``data/interim`` and ``data/processed``. Raw datasets, source code, configs,
tests, reports, and current external-holdout datasets are never removed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


CURRENT_SELECTION_FOLDER = "externalHoldout_test_fc_uplink_fade"


@dataclass(frozen=True)
class CleanupCandidate:
    """One generated path plus the reason it is safe to remove."""

    path: Path
    reason: str


def discover_cleanup_candidates(root: Path = PROJECT_ROOT) -> tuple[list[CleanupCandidate], list[Path]]:
    """Return generated cleanup candidates and uncertain skipped paths."""

    candidates: list[CleanupCandidate] = []
    uncertain: list[Path] = []
    dataset_root = (
        root
        / "data"
        / "processed"
        / "autoregressive"
        / "threshold_10p0"
        / "datasets_L30_h10"
    )
    if dataset_root.exists():
        for path in sorted(dataset_root.iterdir()):
            if not path.is_dir():
                uncertain.append(path)
                continue
            if path.name == CURRENT_SELECTION_FOLDER:
                continue
            if path.name.startswith(("auto_largest_", "single_")):
                candidates.append(
                    CleanupCandidate(
                        path=path,
                        reason="obsolete generated pilot processed dataset folder",
                    )
                )
            else:
                uncertain.append(path)

    processed_root = root / "data" / "processed" / "autoregressive" / "threshold_10p0"
    for filename in (
        "window_index_L30_h10.parquet.as.json",
        "window_index_L30_h10.parquet.tmp",
    ):
        path = processed_root / filename
        if path.exists():
            candidates.append(
                CleanupCandidate(
                    path=path,
                    reason="obsolete generated compatibility/temp window-index artifact",
                )
            )

    interim_root = root / "data" / "interim"
    for subfolder_name in ("clean_signal", "candidate_events"):
        subfolder = interim_root / subfolder_name
        if subfolder.exists():
            for path in sorted(subfolder.glob("threshold_*")):
                candidates.append(
                    CleanupCandidate(
                        path=path,
                        reason="generated interim artifact recreatable from raw data",
                    )
                )

    return candidates, uncertain


def print_cleanup_plan(candidates: list[CleanupCandidate], uncertain: list[Path]) -> None:
    """Print the exact dry-run or execute cleanup plan."""

    print("=== Generated dataset cleanup plan ===")
    if not candidates:
        print("No generated cleanup candidates were found.")
    for candidate in candidates:
        print(f"REMOVE: {candidate.path.relative_to(PROJECT_ROOT)}")
        print(f"  reason: {candidate.reason}")
    if uncertain:
        print("\n=== Uncertain paths skipped ===")
        for path in uncertain:
            print(f"SKIP: {path.relative_to(PROJECT_ROOT)}")
    print("\nProtected: data/raw, source code, configs, tests, AGENTS.md, reports.")


def execute_cleanup(candidates: list[CleanupCandidate]) -> None:
    """Remove the discovered generated candidates."""

    for candidate in candidates:
        if candidate.path.is_dir():
            shutil.rmtree(candidate.path)
        elif candidate.path.exists():
            candidate.path.unlink()


def parse_args() -> argparse.Namespace:
    """Parse CLI flags. Dry-run is the default and safest mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the cleanup plan without deleting anything. This is the default.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Delete the listed generated artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the conservative cleanup discovery and optional execution."""

    args = parse_args()
    candidates, uncertain = discover_cleanup_candidates()
    print_cleanup_plan(candidates, uncertain)
    if args.execute:
        execute_cleanup(candidates)
        print("\nCleanup executed.")
    else:
        print("\nDry-run only. Re-run with --execute to remove listed paths.")


if __name__ == "__main__":
    main()
