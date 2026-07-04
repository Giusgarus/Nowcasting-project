"""Train learnable-shapelet regressors for current-level persistence."""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from contextlib import nullcontext
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.current_level_persistence.evaluation.metrics import (
    compute_duration_metrics,
    compute_event_duration_metrics,
    seconds_from_log1p,
)
from src.tasks.current_level_persistence.data.dataset import (
    SCALAR_CONTEXT_FEATURE_NAMES,
    prepare_scalar_context_splits,
)
from src.tasks.current_level_persistence.models.learnable_shapelets import (
    SUPPORTED_MODEL_IDS,
    ShapeletConfig,
    build_learnable_shapelet_model,
)
from src.tasks.current_level_persistence.plots.diagnostics import (
    plot_duration_distribution,
    plot_error_by_event,
    plot_learned_shapelets,
    plot_residuals,
    plot_true_vs_predicted_duration,
)
from src.tasks.current_level_persistence.utils.paths import (
    RUN_INDEX_COLUMNS,
    make_run_id,
    model_dir,
    run_dir,
    run_index_path,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.device import select_device
from src.utils.reproducibility import set_seed
from src.utils.results_paths import (
    ensure_results_subdirs,
    relative_project_path,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml"
)


class CurrentLevelPersistenceDataset(Dataset):
    """Expose one current-level persistence split as tensors."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        input_key: str,
        target_key: str,
        scalar_context_key: str | None = None,
    ):
        self.inputs = torch.as_tensor(arrays[input_key], dtype=torch.float32)
        self.targets = torch.as_tensor(arrays[target_key], dtype=torch.float32)
        self.scalar_inputs = (
            torch.as_tensor(arrays[scalar_context_key], dtype=torch.float32)
            if scalar_context_key is not None and scalar_context_key in arrays
            else None
        )
        if self.inputs.ndim != 2:
            raise ValueError(f"{input_key} must have shape (N, context_length).")
        if self.targets.ndim != 1:
            raise ValueError(f"{target_key} must have shape (N,).")
        if len(self.inputs) != len(self.targets):
            raise ValueError("Input and target arrays have different lengths.")
        if scalar_context_key is not None and self.scalar_inputs is None:
            raise ValueError(
                f"Scalar context was requested, but {scalar_context_key!r} is missing."
            )
        if self.scalar_inputs is not None:
            if self.scalar_inputs.ndim != 2:
                raise ValueError(
                    f"{scalar_context_key} must have shape (N, scalar_features)."
                )
            if len(self.scalar_inputs) != len(self.targets):
                raise ValueError("Scalar context and target arrays have different lengths.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.scalar_inputs is None:
            return self.inputs[index], self.targets[index]
        return self.inputs[index], self.scalar_inputs[index], self.targets[index]


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
        help="Validate config/model/path setup without training.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional temporary override for training.max_epochs.",
    )
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Optional suffix appended to the run ID with a double-underscore separator.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing run/checkpoint directory for the resolved run ID.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional override for training.device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for DataLoader worker processes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for training.batch_size.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one NPZ split into memory."""

    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def build_shapelet_config(config: dict, context_length: int) -> ShapeletConfig:
    """Construct the model config from shared shapelet and head sections."""

    shapelets = config["shapelets"]
    model = config.get("model", {})
    features = config.get("features", {})
    scalar_encoder = config.get("scalar_encoder", {})
    use_scalar_context = bool(features.get("use_scalar_context", False))
    return ShapeletConfig(
        context_length=context_length,
        shapelet_lengths=tuple(int(value) for value in shapelets["shapelet_lengths"]),
        n_shapelets_per_length=int(shapelets["n_shapelets_per_length"]),
        hidden_dim=model.get("hidden_dim"),
        hidden_size=int(model.get("hidden_size", 128)),
        num_hidden_layers=int(model.get("num_hidden_layers", 2)),
        d_model=int(model.get("d_model", 64)),
        n_heads=int(model.get("n_heads", 4)),
        transformer_layers=int(model.get("num_layers", 1)),
        pooling=str(model.get("pooling", "mean")),
        conv_channels=int(model.get("conv_channels", 64)),
        num_conv_layers=int(model.get("num_conv_layers", 2)),
        kernel_size=int(model.get("kernel_size", 3)),
        dropout=float(model.get("dropout", 0.1)),
        use_scalar_context=use_scalar_context,
        scalar_context_dim=(
            len(SCALAR_CONTEXT_FEATURE_NAMES) if use_scalar_context else 0
        ),
        scalar_encoder_hidden_dim=int(scalar_encoder.get("hidden_dim", 32)),
        scalar_encoder_dropout=float(scalar_encoder.get("dropout", 0.1)),
    )


def prepare_split_arrays(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    config: Mapping,
    *,
    delta: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict | None]:
    """Resolve optional scalar context and train-only standardization."""

    features = config.get("features", {})
    if not bool(features.get("use_scalar_context", False)):
        return {split: dict(arrays) for split, arrays in split_arrays.items()}, None
    dataset = config["dataset"]
    scalar_key = str(
        features.get(
            "scalar_context_key",
            dataset.get("scalar_context_key", "scalar_context_features"),
        )
    )
    return prepare_scalar_context_splits(
        split_arrays,
        scalar_context_key=scalar_key,
        raw_input_key=str(dataset.get("raw_input_key", "X_raw")),
        delta=delta,
        standardize=bool(features.get("scalar_context_standardize", True)),
    )


def scalar_context_metadata(config: Mapping, scaler: Mapping | None) -> dict:
    """Return reproducibility metadata for the optional scalar-context branch."""

    features = config.get("features", {})
    enabled = bool(features.get("use_scalar_context", False))
    dataset = config.get("dataset", {})
    output = {
        "shapelet_input": dataset.get("input_key", "X_relative_to_current"),
        "scalar_context": enabled,
        "scalar_context_key": features.get(
            "scalar_context_key",
            dataset.get("scalar_context_key", "scalar_context_features"),
        ),
        "scalar_context_feature_names": list(SCALAR_CONTEXT_FEATURE_NAMES)
        if enabled
        else [],
        "scalar_context_standardize": bool(
            features.get("scalar_context_standardize", True)
        )
        if enabled
        else False,
        "scalar_scaler_fitted_on": "train" if enabled else None,
    }
    if scaler is None:
        return {"features": output, "scalar_context_scaler": None}
    return {
        "features": output,
        "scalar_context_scaler": {
            "mean": list(scaler["mean"]),
            "std": list(scaler["std"]),
        },
    }


def unpack_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Move a shapelet batch to device and expose optional scalar context."""

    if len(batch) == 2:
        inputs, targets = batch
        scalar_inputs = None
    elif len(batch) == 3:
        inputs, scalar_inputs, targets = batch
        scalar_inputs = scalar_inputs.to(device)
    else:
        raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
    return inputs.to(device), scalar_inputs, targets.to(device)


def make_loss(name: str) -> nn.Module:
    """Return the configured log-space regression loss."""

    if name == "huber":
        return nn.SmoothL1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {name}")


def make_loader(
    arrays: Mapping[str, np.ndarray],
    *,
    input_key: str,
    target_key: str,
    scalar_context_key: str | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic PyTorch dataloader."""

    dataset = CurrentLevelPersistenceDataset(
        arrays,
        input_key,
        target_key,
        scalar_context_key,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def predict(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return true and predicted log1p seconds for a split."""

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            inputs, scalar_inputs, targets = unpack_batch(batch, device)
            predictions = model(inputs, scalar_inputs).detach().cpu()
            y_pred.append(predictions.numpy())
            y_true.append(targets.detach().cpu().numpy())
    y_true_array = np.concatenate(y_true)
    y_pred_array = np.concatenate(y_pred)
    if not np.isfinite(y_pred_array).all():
        raise RuntimeError("Model produced non-finite predictions during evaluation.")
    return y_true_array, y_pred_array


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    metadata: pd.DataFrame,
    *,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate one split in log space and raw duration space."""

    y_true, y_pred = predict(model, loader, device=device)
    y_true_tensor = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.as_tensor(y_pred, dtype=torch.float32)
    metrics = compute_duration_metrics(y_true, y_pred)
    metrics.update(
        compute_event_duration_metrics(
            metadata["global_event_id"].to_numpy(),
            y_true,
            y_pred,
        )
    )
    metrics["log_loss"] = float(loss_fn(y_pred_tensor, y_true_tensor).item())
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError(f"Evaluation produced non-finite metrics: {metrics}")
    return metrics, y_true, y_pred


def prediction_frame(
    metadata: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    model_id: str,
    run_id: str,
    split: str,
) -> pd.DataFrame:
    """Build the standard prediction table for one split."""

    frame = metadata.reset_index(drop=True).copy()
    y_true_seconds = seconds_from_log1p(y_true_log, clip_negative=False)
    y_pred_seconds = seconds_from_log1p(y_pred_log, clip_negative=True)
    output = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "dataset_name": frame["dataset_name"],
            "event_id": frame["event_id"],
            "window_id": frame["window_id"],
            "global_event_id": frame["global_event_id"],
            "global_window_id": frame["global_window_id"],
            "split": split,
            "y_true_log1p_seconds": y_true_log,
            "y_pred_log1p_seconds": y_pred_log,
            "y_true_seconds": y_true_seconds,
            "y_pred_seconds": y_pred_seconds,
            "absolute_error_seconds": np.abs(y_pred_seconds - y_true_seconds),
            "absolute_error_minutes": np.abs(y_pred_seconds - y_true_seconds) / 60.0,
            "model_id": model_id,
            "run_id": run_id,
        }
    )
    return output


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    config: Mapping,
    run_id: str,
    scalar_context_scaler: Mapping | None = None,
) -> None:
    """Save a reproducible PyTorch checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "config": dict(config),
            "run_id": run_id,
            "scalar_context_scaler": dict(scalar_context_scaler)
            if scalar_context_scaler is not None
            else None,
        },
        path,
    )


def learned_shapelets_numpy(model: nn.Module) -> dict[int, np.ndarray]:
    """Extract learned shapelets from a trained model."""

    return {
        length: values.numpy()
        for length, values in model.shapelets.shapelets_by_length().items()
    }


def main() -> None:
    """Train and evaluate a configured learnable-shapelet model."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != "current_level_persistence":
        raise ValueError("Config task_name must be current_level_persistence.")
    if config.get("model_family") != "learnable_shapelets":
        raise ValueError("Only model_family=learnable_shapelets is supported.")

    model_id = str(config["model_id"])
    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unsupported model_id: {model_id}")
    training = dict(config["training"])
    if args.max_epochs is not None:
        training["max_epochs"] = int(args.max_epochs)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.device is not None:
        training["device"] = str(args.device)
    if args.num_workers is not None:
        training["num_workers"] = int(args.num_workers)
    config["training"] = training
    set_seed(int(training.get("seed", 42)))

    dataset_path = project_path(config["dataset"]["path"])
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. Build it first."
        )
    dataset_metadata = load_yaml_config(dataset_path / "dataset_metadata.yaml")
    dataset_build_fingerprint = config_fingerprint(dataset_metadata)
    delta = float(dataset_metadata["delta"])
    context_length = int(dataset_metadata["context_length"])
    selection_id = str(config.get("selection_id", dataset_metadata["selection_id"]))
    use_scalar_context = bool(
        config.get("features", {}).get("use_scalar_context", False)
    )
    run_id = make_run_id(
        delta=delta,
        context_length=context_length,
        model_id=model_id,
        selection_id=selection_id,
        scalar_context=use_scalar_context,
        run_suffix=args.run_suffix,
    )
    result_dir = run_dir(run_id)
    checkpoint_dir = model_dir(run_id)

    input_key = str(config["dataset"]["input_key"])
    target_key = str(config["dataset"]["target_key"])
    split_arrays = {
        "train": load_npz(dataset_path / "train.npz"),
        "val": load_npz(dataset_path / "val.npz"),
        "test": load_npz(dataset_path / "test.npz"),
    }
    split_arrays, scalar_scaler = prepare_split_arrays(
        split_arrays,
        config,
        delta=delta,
    )
    scalar_context_key = (
        str(
            config.get("features", {}).get(
                "scalar_context_key",
                config["dataset"].get(
                    "scalar_context_key",
                    "scalar_context_features",
                ),
            )
        )
        if use_scalar_context
        else None
    )
    split_metadata = {
        "train": pd.read_parquet(dataset_path / "train_metadata.parquet"),
        "val": pd.read_parquet(dataset_path / "val_metadata.parquet"),
        "test": pd.read_parquet(dataset_path / "test_metadata.parquet"),
    }
    requested_device = str(training.get("device", "auto"))
    device_name = select_device() if requested_device == "auto" else requested_device
    device = torch.device(device_name)
    num_workers = int(training.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    loaders = {
        "train": make_loader(
            split_arrays["train"],
            input_key=input_key,
            target_key=target_key,
            scalar_context_key=scalar_context_key,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": make_loader(
            split_arrays["val"],
            input_key=input_key,
            target_key=target_key,
            scalar_context_key=scalar_context_key,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": make_loader(
            split_arrays["test"],
            input_key=input_key,
            target_key=target_key,
            scalar_context_key=scalar_context_key,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    model = build_learnable_shapelet_model(
        model_id,
        build_shapelet_config(config, context_length),
    ).to(device)

    print("=== Current-Level Persistence Learnable-Shapelet Training ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Model ID: {model_id}")
    print(f"Dataset: {dataset_path.relative_to(PROJECT_ROOT)}")
    print(f"Input key: {input_key}")
    print(f"Scalar context: {use_scalar_context}")
    if scalar_context_key is not None:
        print(f"Scalar context key: {scalar_context_key} (standardized on train only)")
    print(f"Target key: {target_key}")
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")
    print(f"Results: {result_dir.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoints: {checkpoint_dir.relative_to(PROJECT_ROOT)}")
    print(f"Batch size: {int(training['batch_size'])}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Gradient clip norm: {training.get('gradient_clip_norm', 'disabled')}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    if args.dry_run:
        print("Dry-run completed. No training was started.")
        return

    existing_paths = [path for path in (result_dir, checkpoint_dir) if path.exists()]
    if existing_paths and not args.force:
        relative_paths = ", ".join(relative_project_path(path) for path in existing_paths)
        raise FileExistsError(
            f"Run artifacts already exist: {relative_paths}. "
            "Use --force to overwrite them, or pass a different --run-suffix."
        )
    if args.force:
        for path in existing_paths:
            shutil.rmtree(path)

    loss_fn = make_loss(str(training.get("loss", "huber")))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp_enabled = bool(training.get("mixed_precision", False)) and device.type == "cuda"
    gradient_clip_norm = training.get("gradient_clip_norm")
    gradient_clip_norm = (
        float(gradient_clip_norm) if gradient_clip_norm is not None else None
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    result_paths = ensure_results_subdirs(
        result_dir,
        ("metrics", "predictions", "figures", "tables"),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    patience = int(training["early_stopping_patience"])
    epochs_without_improvement = 0

    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for batch in loaders["train"]:
            inputs, scalar_inputs, targets = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda")
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                predictions = model(inputs, scalar_inputs)
                loss = loss_fn(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}: {loss.item()}"
                )
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                except TypeError:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        raise RuntimeError(
                            f"Non-finite gradient norm at epoch {epoch}: {grad_norm}"
                        )
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(targets)
            train_loss_sum += float(loss.detach().cpu().item()) * batch_size
            n_train += batch_size

        val_metrics, _, _ = evaluate_split(
            model,
            loaders["val"],
            split_metadata["val"],
            loss_fn=loss_fn,
            device=device,
        )
        train_loss = train_loss_sum / max(n_train, 1)
        row = {
            "epoch": epoch,
            "train_log_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        current_metric = float(val_metrics["mae_seconds"])
        improved = current_metric < best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                run_id=run_id,
                scalar_context_scaler=scalar_scaler,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6g} | "
            f"val_mae_seconds={val_metrics['mae_seconds']:.6g} | "
            f"val_rmse_seconds={val_metrics['rmse_seconds']:.6g}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    save_checkpoint(
        checkpoint_dir / "last_model.pt",
        model=model,
        optimizer=optimizer,
        epoch=history[-1]["epoch"],
        metrics={key: value for key, value in history[-1].items() if key != "epoch"},
        config=config,
        run_id=run_id,
        scalar_context_scaler=scalar_scaler,
    )
    model.load_state_dict(best_state)

    val_metrics, val_true, val_pred = evaluate_split(
        model,
        loaders["val"],
        split_metadata["val"],
        loss_fn=loss_fn,
        device=device,
    )
    test_metrics, test_true, test_pred = evaluate_split(
        model,
        loaders["test"],
        split_metadata["test"],
        loss_fn=loss_fn,
        device=device,
    )
    val_predictions = prediction_frame(
        split_metadata["val"],
        val_true,
        val_pred,
        model_id=model_id,
        run_id=run_id,
        split="val",
    )
    test_predictions = prediction_frame(
        split_metadata["test"],
        test_true,
        test_pred,
        model_id=model_id,
        run_id=run_id,
        split="test",
    )

    pd.DataFrame(history).to_csv(
        result_paths["tables"] / "training_history.csv",
        index=False,
    )
    val_predictions.to_parquet(
        result_paths["predictions"] / "val_predictions.parquet",
        index=False,
    )
    test_predictions.to_parquet(
        result_paths["predictions"] / "test_predictions.parquet",
        index=False,
    )
    pd.DataFrame([{f"val_{k}": v for k, v in val_metrics.items()}]).to_csv(
        result_paths["metrics"] / "val_metrics.csv",
        index=False,
    )
    pd.DataFrame([{f"test_{k}": v for k, v in test_metrics.items()}]).to_csv(
        result_paths["metrics"] / "test_metrics.csv",
        index=False,
    )
    metrics_summary = {
        "run_id": run_id,
        "model_id": model_id,
        "best_epoch": best_epoch,
        "final_training_epochs": int(history[-1]["epoch"]),
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    pd.DataFrame([metrics_summary]).to_csv(
        result_paths["metrics"] / "metrics_summary.csv",
        index=False,
    )

    plot_true_vs_predicted_duration(
        test_predictions,
        result_paths["figures"] / "y_true_vs_y_pred_test.png",
    )
    plot_residuals(test_predictions, result_paths["figures"] / "residuals_test.png")
    plot_error_by_event(
        test_predictions,
        result_paths["figures"] / "error_by_event_test.png",
    )
    plot_duration_distribution(
        test_predictions,
        result_paths["figures"] / "predicted_vs_true_distribution_test.png",
    )
    shapelets = learned_shapelets_numpy(model)
    plot_learned_shapelets(shapelets, result_paths["figures"])
    for length, values in shapelets.items():
        np.save(result_paths["tables"] / f"learned_shapelets_L{length}.npy", values)

    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_id": run_id,
        "task_name": "current_level_persistence",
        "model_family": "learnable_shapelets",
        "model_id": model_id,
        "selection_id": selection_id,
        "delta": delta,
        "context_length": context_length,
        "run_suffix": args.run_suffix,
        "dataset_path": relative_project_path(dataset_path),
        "dataset_build_fingerprint": dataset_build_fingerprint,
        "target_search_scope": dataset_metadata.get("target_search_scope"),
        "censoring_policy": dataset_metadata.get("censoring_policy"),
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "best_epoch": best_epoch,
        "device": str(device),
        "results_path": relative_project_path(result_dir),
        "checkpoint_path": relative_project_path(checkpoint_dir),
        "created_at": created_at,
        **scalar_context_metadata(config, scalar_scaler),
    }
    save_yaml(result_dir / "metadata.yaml", metadata)
    save_yaml(result_dir / "config_resolved.yaml", config)
    save_yaml(checkpoint_dir / "metadata.yaml", metadata)
    save_yaml(checkpoint_dir / "config_resolved.yaml", config)
    upsert_index_row(
        run_index_path(),
        {
            "run_id": run_id,
            "task_name": "current_level_persistence",
            "model_family": "learnable_shapelets",
            "model_id": model_id,
            "delta": delta,
            "context_length": context_length,
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_path),
            "best_epoch": best_epoch,
            "best_val_mae_seconds": val_metrics["mae_seconds"],
            "best_val_rmse_seconds": val_metrics["rmse_seconds"],
            "test_mae_seconds": test_metrics["mae_seconds"],
            "test_rmse_seconds": test_metrics["rmse_seconds"],
            "test_median_ae_seconds": test_metrics["median_ae_seconds"],
            "created_at": created_at,
            "status": "complete",
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )

    print("=== Final metrics ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation MAE seconds: {val_metrics['mae_seconds']:.6g}")
    print(f"Validation RMSE seconds: {val_metrics['rmse_seconds']:.6g}")
    print(f"Test MAE seconds: {test_metrics['mae_seconds']:.6g}")
    print(f"Test RMSE seconds: {test_metrics['rmse_seconds']:.6g}")
    print(f"Results: {result_dir}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
