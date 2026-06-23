"""Group result artifacts by dataset selection and comparison type."""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.results_paths import (
    get_comparison_dir,
    get_grid_search_dir,
    get_run_dir,
    relative_project_path,
)

SKIP_DIR_NAMES = {
    ".ipynb_checkpoints",
    "_unscoped",
    "model_selection",
    "model_summary",
    "other",
    "switch_eval",
}


def parse_args() -> argparse.Namespace:
    """Parse maintenance-script options."""

    parser = argparse.ArgumentParser(
        description="Move flat result artifacts into grouped result folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files or indexes.",
    )
    return parser.parse_args()


def move_dir(source: Path, destination: Path, *, dry_run: bool) -> tuple[Path, Path] | None:
    """Move one directory when it is still in the old flat location."""

    if not source.is_dir() or source.resolve() == destination.resolve():
        return None
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if dry_run:
        return source, destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return source, destination


def move_flat_children(base: Path, destination_factory, *, dry_run: bool) -> list[tuple[Path, Path]]:
    """Move direct child directories under a base folder using a destination factory."""

    if not base.is_dir():
        return []
    planned: list[tuple[Path, Path]] = []
    children = sorted(path for path in base.iterdir() if path.is_dir())
    for child in children:
        if child.name in SKIP_DIR_NAMES:
            continue
        destination = destination_factory(child.name)
        moved = move_dir(child, destination, dry_run=dry_run)
        if moved is not None:
            planned.append(moved)
    return planned


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


def update_run_index(*, dry_run: bool) -> None:
    """Update run index paths after grouping run folders."""

    path = PROJECT_ROOT / "results/index/runs.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        run_id = str(row["run_id"])
        old_root = PROJECT_ROOT / "results/runs" / run_id
        new_root = get_run_dir(run_id)
        frame.loc[index, "results_path"] = replace_path_prefix(
            row.get("results_path"),
            old_root,
            new_root,
        )
        frame.loc[index, "predictions_path"] = replace_path_prefix(
            row.get("predictions_path"),
            old_root,
            new_root,
        )
        frame.loc[index, "metrics_path"] = replace_path_prefix(
            row.get("metrics_path"),
            old_root,
            new_root,
        )
    if not dry_run:
        frame.to_csv(path, index=False)


def update_comparison_index(*, dry_run: bool) -> None:
    """Update comparison index paths after grouping comparison folders."""

    path = PROJECT_ROOT / "results/index/comparisons.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        comparison_id = str(row["comparison_id"])
        old_root = PROJECT_ROOT / "results/comparisons" / comparison_id
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
    """Update grid-search index paths after grouping search folders."""

    path = PROJECT_ROOT / "results/index/grid_searches.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        search_id = str(row["search_id"])
        old_root = PROJECT_ROOT / "results/grid_searches" / search_id
        new_root = get_grid_search_dir(search_id)
        frame.loc[index, "results_path"] = replace_path_prefix(
            row.get("results_path"),
            old_root,
            new_root,
        )
    if not dry_run:
        frame.to_csv(path, index=False)


def main() -> None:
    """Run the result-folder reorganization."""

    args = parse_args()
    moves: list[tuple[Path, Path]] = []
    moves.extend(
        move_flat_children(
            PROJECT_ROOT / "results/runs",
            get_run_dir,
            dry_run=args.dry_run,
        )
    )
    moves.extend(
        move_flat_children(
            PROJECT_ROOT / "results/comparisons",
            get_comparison_dir,
            dry_run=args.dry_run,
        )
    )
    moves.extend(
        move_flat_children(
            PROJECT_ROOT / "results/grid_searches",
            get_grid_search_dir,
            dry_run=args.dry_run,
        )
    )
    update_run_index(dry_run=args.dry_run)
    update_comparison_index(dry_run=args.dry_run)
    update_grid_search_index(dry_run=args.dry_run)

    action = "Would move" if args.dry_run else "Moved"
    print(f"{action} {len(moves)} result folders.")
    for source, destination in moves:
        print(
            f"- {relative_project_path(source)} -> "
            f"{relative_project_path(destination)}"
        )
    if args.dry_run:
        print("Dry run only: indexes were not modified.")


if __name__ == "__main__":
    main()
