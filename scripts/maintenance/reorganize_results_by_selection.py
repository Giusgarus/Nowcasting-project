"""Migrate legacy autoregressive artifacts into task-scoped folders."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.results_paths import (
    TASK_AUTOREGRESSIVE,
    get_comparison_dir,
    get_data_preparation_dir,
    get_grid_search_dir,
    get_legacy_comparison_dir,
    get_legacy_data_preparation_dir,
    get_legacy_grid_search_dir,
    get_legacy_model_dir,
    get_legacy_perfect_switch_dir,
    get_legacy_run_dir,
    get_model_dir,
    get_perfect_switch_dir,
    get_run_dir,
    relative_project_path,
)

TASK_ROOT_NAMES = {TASK_AUTOREGRESSIVE, "current_level_persistence"}
LEGACY_MODEL_FAMILIES = {"chronos", "gru", "patchtst"}


def parse_args() -> argparse.Namespace:
    """Parse maintenance-script options."""

    parser = argparse.ArgumentParser(
        description=(
            "Move legacy autoregressive artifacts into task-scoped result and "
            "model folders."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files or indexes.",
    )
    return parser.parse_args()


def move_dir(source: Path, destination: Path, *, dry_run: bool) -> tuple[Path, Path] | None:
    """Move one directory unless it is already at its destination."""

    if not source.is_dir() or source.resolve() == destination.resolve():
        return None
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if dry_run:
        return source, destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return source, destination


def prune_empty(path: Path, *, dry_run: bool) -> None:
    """Remove an empty legacy container after its children have moved."""

    if dry_run or not path.is_dir():
        return
    try:
        path.rmdir()
    except OSError:
        pass


def replace_path_prefix(value: object, old_root: Path, new_root: Path) -> str:
    """Replace a repository-relative path prefix while preserving subpaths."""

    if pd.isna(value) or str(value).strip() == "":
        return ""
    text = str(value)
    old_text = relative_project_path(old_root)
    new_text = relative_project_path(new_root)
    if text == old_text:
        return new_text
    prefix = old_text + "/"
    if text.startswith(prefix):
        return new_text + "/" + text[len(prefix) :]
    return text


def migrate_runs(*, dry_run: bool) -> list[tuple[Path, Path]]:
    """Move legacy run folders from results/runs/<selection>/<run_id>."""

    base = PROJECT_ROOT / "results" / "runs"
    moves: list[tuple[Path, Path]] = []
    if not base.is_dir():
        return moves
    for selection_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if selection_dir.name in TASK_ROOT_NAMES:
            continue
        for run_dir in sorted(path for path in selection_dir.iterdir() if path.is_dir()):
            moved = move_dir(run_dir, get_run_dir(run_dir.name), dry_run=dry_run)
            if moved:
                moves.append(moved)
        prune_empty(selection_dir, dry_run=dry_run)
    return moves


def migrate_comparisons(*, dry_run: bool) -> list[tuple[Path, Path]]:
    """Move legacy comparisons from results/comparisons/<type>/<selection>/<id>."""

    base = PROJECT_ROOT / "results" / "comparisons"
    moves: list[tuple[Path, Path]] = []
    if not base.is_dir():
        return moves
    for group_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        for selection_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            if selection_dir.name in TASK_ROOT_NAMES:
                continue
            for comparison_dir in sorted(
                path for path in selection_dir.iterdir() if path.is_dir()
            ):
                moved = move_dir(
                    comparison_dir,
                    get_comparison_dir(comparison_dir.name),
                    dry_run=dry_run,
                )
                if moved:
                    moves.append(moved)
            prune_empty(selection_dir, dry_run=dry_run)
    return moves


def migrate_grid_searches(*, dry_run: bool) -> list[tuple[Path, Path]]:
    """Move legacy grid-search folders from results/grid_searches/<selection>/<id>."""

    base = PROJECT_ROOT / "results" / "grid_searches"
    moves: list[tuple[Path, Path]] = []
    if not base.is_dir():
        return moves
    for selection_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if selection_dir.name in TASK_ROOT_NAMES:
            continue
        for search_dir in sorted(path for path in selection_dir.iterdir() if path.is_dir()):
            moved = move_dir(search_dir, get_grid_search_dir(search_dir.name), dry_run=dry_run)
            if moved:
                moves.append(moved)
        prune_empty(selection_dir, dry_run=dry_run)
    return moves


def migrate_simple_selection_tree(
    base: Path,
    destination_factory,
    *,
    dry_run: bool,
) -> list[tuple[Path, Path]]:
    """Move legacy selection folders into task-scoped selection folders."""

    moves: list[tuple[Path, Path]] = []
    if not base.is_dir():
        return moves
    for selection_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if selection_dir.name in TASK_ROOT_NAMES:
            continue
        moved = move_dir(
            selection_dir,
            destination_factory(selection_dir.name),
            dry_run=dry_run,
        )
        if moved:
            moves.append(moved)
    return moves


def migrate_models(*, dry_run: bool) -> list[tuple[Path, Path]]:
    """Move legacy model folders from models/<family>/<run_id>."""

    base = PROJECT_ROOT / "models"
    moves: list[tuple[Path, Path]] = []
    if not base.is_dir():
        return moves
    for family_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if family_dir.name not in LEGACY_MODEL_FAMILIES:
            continue
        for run_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            moved = move_dir(
                run_dir,
                get_model_dir(family_dir.name, run_dir.name),
                dry_run=dry_run,
            )
            if moved:
                moves.append(moved)
        prune_empty(family_dir, dry_run=dry_run)
    return moves


def update_run_index(*, dry_run: bool) -> None:
    """Update run index paths after task-scoping run and model folders."""

    path = PROJECT_ROOT / "results" / "index" / "runs.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        run_id = str(row["run_id"])
        model_family = str(row.get("model_family", ""))
        old_run_root = get_legacy_run_dir(run_id)
        new_run_root = get_run_dir(run_id)
        old_model_root = get_legacy_model_dir(model_family, run_id)
        new_model_root = get_model_dir(model_family, run_id)
        for column in ("results_path", "predictions_path", "metrics_path"):
            frame.loc[index, column] = replace_path_prefix(
                row.get(column),
                old_run_root,
                new_run_root,
            )
        frame.loc[index, "model_path"] = replace_path_prefix(
            row.get("model_path"),
            old_model_root,
            new_model_root,
        )
    if not dry_run:
        frame.to_csv(path, index=False)


def update_comparison_index(*, dry_run: bool) -> None:
    """Update comparison index paths after task-scoping comparison folders."""

    path = PROJECT_ROOT / "results" / "index" / "comparisons.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        comparison_id = str(row["comparison_id"])
        old_root = get_legacy_comparison_dir(comparison_id)
        new_root = get_comparison_dir(comparison_id)
        for column in ("results_path", "metrics_path", "figures_path"):
            frame.loc[index, column] = replace_path_prefix(
                row.get(column),
                old_root,
                new_root,
            )
    if not dry_run:
        frame.to_csv(path, index=False)


def update_grid_search_index(*, dry_run: bool) -> None:
    """Update grid-search index paths after task-scoping search folders."""

    path = PROJECT_ROOT / "results" / "index" / "grid_searches.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        search_id = str(row["search_id"])
        target_run_id = str(row.get("target_run_id", ""))
        old_search_root = get_legacy_grid_search_dir(search_id)
        new_search_root = get_grid_search_dir(search_id)
        for column in ("results_path",):
            frame.loc[index, column] = replace_path_prefix(
                row.get(column),
                old_search_root,
                new_search_root,
            )
        if target_run_id:
            model_family = str(row.get("model_family", ""))
            frame.loc[index, "best_model_path"] = replace_path_prefix(
                row.get("best_model_path"),
                get_legacy_model_dir(model_family, target_run_id),
                get_model_dir(model_family, target_run_id),
            )
    if not dry_run:
        frame.to_csv(path, index=False)


def update_switch_reference_index(*, dry_run: bool) -> None:
    """Update Perfect Switch index paths after task-scoping references."""

    path = PROJECT_ROOT / "results" / "index" / "switch_references.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        selection_id = str(row["selection_id"])
        old_root = get_legacy_perfect_switch_dir(selection_id)
        new_root = get_perfect_switch_dir(selection_id)
        for column in ("results_path", "timeseries_path", "window_targets_path"):
            frame.loc[index, column] = replace_path_prefix(
                row.get(column),
                old_root,
                new_root,
            )
    if not dry_run:
        frame.to_csv(path, index=False)


def update_dataset_index(*, dry_run: bool) -> None:
    """Update human-readable data-preparation paths when present."""

    path = PROJECT_ROOT / "results" / "index" / "datasets.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if "results_path" not in frame.columns:
        return
    for index, row in frame.iterrows():
        selection_id = str(row["selection_id"])
        frame.loc[index, "results_path"] = replace_path_prefix(
            row.get("results_path"),
            get_legacy_data_preparation_dir(selection_id),
            get_data_preparation_dir(selection_id),
        )
    if not dry_run:
        frame.to_csv(path, index=False)


def main() -> None:
    """Run the artifact migration."""

    args = parse_args()
    moves: list[tuple[Path, Path]] = []
    moves.extend(migrate_runs(dry_run=args.dry_run))
    moves.extend(migrate_comparisons(dry_run=args.dry_run))
    moves.extend(migrate_grid_searches(dry_run=args.dry_run))
    moves.extend(
        migrate_simple_selection_tree(
            PROJECT_ROOT / "results" / "switching" / "perfect_switch",
            get_perfect_switch_dir,
            dry_run=args.dry_run,
        )
    )
    moves.extend(
        migrate_simple_selection_tree(
            PROJECT_ROOT / "results" / "data_preparation",
            get_data_preparation_dir,
            dry_run=args.dry_run,
        )
    )
    moves.extend(migrate_models(dry_run=args.dry_run))

    update_run_index(dry_run=args.dry_run)
    update_comparison_index(dry_run=args.dry_run)
    update_grid_search_index(dry_run=args.dry_run)
    update_switch_reference_index(dry_run=args.dry_run)
    update_dataset_index(dry_run=args.dry_run)

    action = "Would move" if args.dry_run else "Moved"
    print(f"{action} {len(moves)} artifact folders.")
    for source, destination in moves:
        print(
            f"- {relative_project_path(source)} -> "
            f"{relative_project_path(destination)}"
        )
    if args.dry_run:
        print("Dry run only: indexes were not modified.")


if __name__ == "__main__":
    main()
