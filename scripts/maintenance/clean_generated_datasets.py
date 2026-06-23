"""Dry-run first cleanup for generated dataset artifacts.

The cleanup is index-aware: dataset folders listed in ``results/index`` are
protected, while stale generated thresholds, pilot selections, and temporary
compatibility files are reported as removable candidates. Raw data, source
code, configs, tests, reports, model checkpoints, and result tables are never
targeted by this script.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


PROTECTED_TOP_LEVELS = {
    "configs",
    "models",
    "reports",
    "results",
    "scripts",
    "src",
    "tests",
}
OBSOLETE_SELECTION_PREFIXES = ("auto_largest_", "single_")
TEMP_GENERATED_SUFFIXES = (".as.json", ".tmp")


@dataclass(frozen=True)
class CleanupCandidate:
    """One generated path plus the reason it is safe to remove."""

    path: Path
    reason: str


def _resolve_project_path(value: str | Path, *, root: Path) -> Path:
    """Return an absolute path from an absolute or repository-relative value."""

    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is inside ``parent`` on all supported Python versions."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _read_indexed_dataset_paths(root: Path) -> set[Path]:
    """Read protected processed-dataset folders from the dataset index."""

    index_path = root / "results" / "index" / "datasets.csv"
    if not index_path.is_file():
        return set()

    indexed_paths: set[Path] = set()
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset_path = (row.get("dataset_path") or "").strip()
            if dataset_path:
                indexed_paths.add(_resolve_project_path(dataset_path, root=root))
    return indexed_paths


def _indexed_threshold_dirs(root: Path, indexed_dataset_paths: set[Path]) -> set[Path]:
    """Return threshold folders that contain an indexed processed dataset."""

    processed_root = (root / "data" / "processed" / "autoregressive").resolve()
    thresholds: set[Path] = set()
    for dataset_path in indexed_dataset_paths:
        if not _is_relative_to(dataset_path, processed_root):
            continue
        for parent in dataset_path.parents:
            if parent.parent == processed_root and parent.name.startswith("threshold_"):
                thresholds.add(parent.resolve())
                break
    return thresholds


def _add_candidate(
    candidates: dict[Path, CleanupCandidate],
    path: Path,
    reason: str,
) -> None:
    """Record one cleanup candidate without duplicating paths."""

    candidates[path.resolve()] = CleanupCandidate(path=path, reason=reason)


def _is_protected_path(path: Path, root: Path, indexed_dataset_paths: set[Path]) -> bool:
    """Guard against accidental removal outside generated data folders."""

    resolved = path.resolve()
    if resolved == root.resolve():
        return True
    try:
        top_level = resolved.relative_to(root.resolve()).parts[0]
    except (IndexError, ValueError):
        return True
    if top_level in PROTECTED_TOP_LEVELS:
        return True
    if top_level == "data" and len(resolved.relative_to(root.resolve()).parts) > 1:
        second_level = resolved.relative_to(root.resolve()).parts[1]
        if second_level == "raw":
            return True
    return any(
        resolved == indexed_path or _is_relative_to(indexed_path, resolved)
        for indexed_path in indexed_dataset_paths
    )


def discover_cleanup_candidates(
    root: Path = PROJECT_ROOT,
) -> tuple[list[CleanupCandidate], list[Path]]:
    """Return generated cleanup candidates and uncertain skipped paths."""

    root = root.resolve()
    indexed_dataset_paths = _read_indexed_dataset_paths(root)
    protected_thresholds = _indexed_threshold_dirs(root, indexed_dataset_paths)
    protected_threshold_names = {path.name for path in protected_thresholds}
    candidates: dict[Path, CleanupCandidate] = {}
    uncertain: list[Path] = []

    processed_root = root / "data" / "processed" / "autoregressive"
    if processed_root.is_dir():
        for threshold_dir in sorted(processed_root.glob("threshold_*")):
            if not threshold_dir.is_dir():
                continue

            if protected_thresholds and threshold_dir.resolve() not in protected_thresholds:
                _add_candidate(
                    candidates,
                    threshold_dir,
                    "generated processed threshold folder not referenced by datasets index",
                )
                continue

            for temp_file in sorted(threshold_dir.glob("*")):
                if temp_file.is_file() and temp_file.name.endswith(TEMP_GENERATED_SUFFIXES):
                    _add_candidate(
                        candidates,
                        temp_file,
                        "generated compatibility or temporary processed artifact",
                    )

            for dataset_root in sorted(threshold_dir.glob("datasets_*")):
                if not dataset_root.is_dir():
                    continue
                for selection_dir in sorted(dataset_root.iterdir()):
                    if selection_dir.resolve() in indexed_dataset_paths:
                        continue
                    if not selection_dir.is_dir():
                        uncertain.append(selection_dir)
                        continue
                    if selection_dir.name.startswith(OBSOLETE_SELECTION_PREFIXES):
                        _add_candidate(
                            candidates,
                            selection_dir,
                            "obsolete generated pilot processed dataset folder",
                        )
                    else:
                        uncertain.append(selection_dir)

    interim_root = root / "data" / "interim"
    for subfolder_name in ("candidate_events", "event_windows"):
        subfolder = interim_root / subfolder_name
        if not subfolder.is_dir() or not protected_threshold_names:
            continue
        for threshold_dir in sorted(subfolder.glob("threshold_*")):
            if threshold_dir.name in protected_threshold_names:
                continue
            _add_candidate(
                candidates,
                threshold_dir,
                "generated interim threshold artifact not referenced by datasets index",
            )

    safe_candidates = []
    for candidate in candidates.values():
        if _is_protected_path(candidate.path, root, indexed_dataset_paths):
            uncertain.append(candidate.path)
        else:
            safe_candidates.append(candidate)
    safe_candidates.sort(key=lambda candidate: candidate.path.as_posix())
    uncertain = sorted(set(uncertain), key=lambda path: path.as_posix())
    return safe_candidates, uncertain


def print_cleanup_plan(
    candidates: list[CleanupCandidate],
    uncertain: list[Path],
) -> None:
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
    print(
        "\nProtected: data/raw, source code, configs, tests, AGENTS.md, reports, "
        "models, results, and indexed processed datasets."
    )


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
