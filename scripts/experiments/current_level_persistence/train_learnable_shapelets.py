"""Train learnable-shapelet models for current-level persistence.

The entry point is intentionally prepared but conservative: it validates the
configuration, dataset location, model ID, run/checkpoint paths, and model
construction. The full training loop will be added in the next implementation
step.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.current_level_persistence.models.learnable_shapelets import (
    SUPPORTED_MODEL_IDS,
    ShapeletConfig,
    build_learnable_shapelet_model,
)
from src.tasks.current_level_persistence.utils.paths import (
    make_run_id,
    model_dir,
    run_dir,
)
from src.utils.config import load_yaml_config
from src.utils.device import select_device

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml"
)


def project_path(path: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def parse_args() -> argparse.Namespace:
    """Parse training options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Learnable-shapelet training config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/model/path setup without entering training.",
    )
    return parser.parse_args()


def infer_dataset_identifiers(dataset_path: Path) -> tuple[float, int, str]:
    """Infer delta, context length, and selection ID from the dataset path."""

    parts = dataset_path.parts
    delta = None
    context_length = None
    for part in parts:
        if part.startswith("delta_"):
            delta = float(part.removeprefix("delta_").replace("p", "."))
        elif re.fullmatch(r"L\d+", part):
            context_length = int(part.removeprefix("L"))
    if delta is None or context_length is None:
        raise ValueError(
            "Dataset path must include delta_<value> and L<context_length> folders."
        )
    return delta, context_length, dataset_path.name


def main() -> None:
    """Validate the configured training setup."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != "current_level_persistence":
        raise ValueError("Config task_name must be current_level_persistence.")
    if config.get("model_family") != "learnable_shapelets":
        raise ValueError("Only model_family=learnable_shapelets is supported here.")

    model_id = str(config["model_id"])
    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unsupported model_id: {model_id}")
    dataset_path = project_path(config["dataset"]["path"])
    delta, context_length, selection_id = infer_dataset_identifiers(dataset_path)
    shapelet_lengths = tuple(int(value) for value in config["shapelets"]["shapelet_lengths"])
    n_shapelets_per_length = int(config["shapelets"]["n_shapelets_per_length"])
    model = build_learnable_shapelet_model(
        model_id,
        ShapeletConfig(
            context_length=context_length,
            shapelet_lengths=shapelet_lengths,
            n_shapelets_per_length=n_shapelets_per_length,
        ),
    )
    requested_device = str(config["training"].get("device", "auto"))
    device = select_device() if requested_device == "auto" else requested_device
    run_id = make_run_id(
        delta=delta,
        context_length=context_length,
        model_id=model_id,
        selection_id=selection_id,
    )
    output_run_dir = run_dir(run_id)
    output_model_dir = model_dir(run_id)

    print("=== Current-Level Persistence Learnable-Shapelet Training Setup ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Model ID: {model_id}")
    print(f"Dataset: {dataset_path.relative_to(PROJECT_ROOT)}")
    print(f"Input key: {config['dataset']['input_key']}")
    print(f"Target key: {config['dataset']['target_key']}")
    print(f"Delta: {delta}")
    print(f"Context length: {context_length}")
    print(f"Selection ID: {selection_id}")
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")
    print(f"Results directory: {output_run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoint directory: {output_model_dir.relative_to(PROJECT_ROOT)}")
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. Build it with "
            "scripts/experiments/current_level_persistence/build_dataset.py first."
        )
    if args.dry_run:
        print("Dry-run completed. No training was started.")
        return
    raise NotImplementedError(
        "The training loop for current-level persistence learnable shapelets "
        "is prepared but not implemented in this step."
    )


if __name__ == "__main__":
    main()
