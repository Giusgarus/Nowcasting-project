"""Train or evaluate one long-fade detection model."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.long_fade_detection.evaluation.metrics import (  # noqa: E402
    THRESHOLDS,
    event_level_metrics,
    sample_classification_metrics,
)
from src.tasks.long_fade_detection.models.classifiers import (  # noqa: E402
    MODEL_ID_SHAPELET_CONV,
    MODEL_ID_TCN,
    MODEL_ID_XGBOOST,
    build_neural_classifier,
)
from src.tasks.long_fade_detection.utils.paths import (  # noqa: E402
    make_run_id,
    make_selection_id,
    model_dir,
    run_dir,
    selection_id_from_run_id,
)
from src.utils.config import load_yaml_config, save_yaml  # noqa: E402
from src.utils.device import select_device  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402
from src.utils.results_paths import relative_project_path  # noqa: E402

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/long_fade_detection/xgboost_lag_scalar_threshold10_duration300.yaml"
)


class LongFadeDataset(Dataset):
    """Torch dataset for neural long-fade classifiers."""

    def __init__(self, arrays: dict[str, np.ndarray], *, sequence_key: str) -> None:
        self.x_sequence = torch.as_tensor(arrays[sequence_key], dtype=torch.float32)
        self.x_scalar = torch.as_tensor(
            arrays["scalar_context_features"],
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(arrays["y_long_fade"], dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_sequence[index], self.x_scalar[index], self.y[index]


def project_path(path: str | Path) -> Path:
    """Resolve a repository-relative or absolute path."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one NPZ file."""

    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in arrays.files}


def load_split_data(dataset_path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    """Load train/validation/test arrays and metadata."""

    arrays = {
        "train": load_npz(dataset_path / "train.npz"),
        "validation": load_npz(dataset_path / "val.npz"),
        "test": load_npz(dataset_path / "test.npz"),
    }
    metadata = {
        "train": pd.read_parquet(dataset_path / "train_metadata.parquet"),
        "validation": pd.read_parquet(dataset_path / "val_metadata.parquet"),
        "test": pd.read_parquet(dataset_path / "test_metadata.parquet"),
    }
    return arrays, metadata


def positive_weight(y: np.ndarray) -> float:
    """Return n_negative / n_positive with safe fallback."""

    labels = np.asarray(y, dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives <= 0:
        return 1.0
    return float(max(1.0, negatives / positives))


def xgboost_classifier(config: dict[str, Any], *, scale_pos_weight: float):
    """Build XGBoost if available, otherwise a sklearn fallback."""

    try:
        from xgboost import XGBClassifier

        params = dict(config.get("model", {}))
        params["scale_pos_weight"] = scale_pos_weight
        params.setdefault("random_state", int(config["training"].get("seed", 42)))
        return XGBClassifier(**params), "xgboost"
    except Exception as error:  # pragma: no cover - exercised when xgboost unavailable
        print(
            "XGBoost is unavailable; falling back to "
            f"HistGradientBoostingClassifier ({type(error).__name__}: {error}).",
            flush=True,
        )
        params = config.get("model", {})
        return (
            HistGradientBoostingClassifier(
                learning_rate=float(params.get("learning_rate", 0.05)),
                max_iter=int(params.get("n_estimators", 300)),
                max_leaf_nodes=2 ** int(params.get("max_depth", 3)),
                random_state=int(config["training"].get("seed", 42)),
            ),
            "sklearn_hist_gradient_boosting",
        )


def prediction_frame(
    metadata: pd.DataFrame,
    *,
    arrays: dict[str, np.ndarray],
    split: str,
    logits: np.ndarray,
    probabilities: np.ndarray,
    model_id: str,
    run_id: str,
) -> pd.DataFrame:
    """Return the standard prediction table."""

    frame = metadata.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["split"] = "val" if split == "validation" else split
    frame["y_true"] = arrays["y_long_fade"].astype(int)
    frame["logit"] = logits.astype(float)
    frame["prob_long_fade"] = probabilities.astype(float)
    frame["model_id"] = model_id
    frame["run_id"] = run_id
    columns = [
        "timestamp",
        "dataset_name",
        "event_id",
        "global_event_id",
        "window_id",
        "split",
        "y_true",
        "logit",
        "prob_long_fade",
        "event_duration_seconds",
        "event_duration_samples",
        "position_in_fade_samples",
        "elapsed_since_fade_start_seconds",
        "model_id",
        "run_id",
    ]
    return frame.loc[:, columns]


def evaluate_predictions(predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute sample-level, threshold, and event-level metrics."""

    summary: dict[str, float | int | str] = {}
    threshold_rows = []
    event_rows = []
    for split, frame in predictions.items():
        split_name = "val" if split == "validation" else split
        metrics = sample_classification_metrics(
            frame["y_true"].to_numpy(dtype=int),
            frame["prob_long_fade"].to_numpy(dtype=float),
            thresholds=THRESHOLDS,
        )
        for key, value in metrics.items():
            summary[f"{split_name}_{key}"] = value
        for threshold in THRESHOLDS:
            suffix = str(threshold).replace(".", "p")
            threshold_rows.append(
                {
                    "split": split_name,
                    "threshold": threshold,
                    "precision": metrics[f"precision_at_{suffix}"],
                    "recall": metrics[f"recall_at_{suffix}"],
                    "f1": metrics[f"f1_at_{suffix}"],
                    "false_positives": metrics[f"false_positives_at_{suffix}"],
                    "false_negatives": metrics[f"false_negatives_at_{suffix}"],
                    "true_positives": metrics[f"true_positives_at_{suffix}"],
                    "true_negatives": metrics[f"true_negatives_at_{suffix}"],
                }
            )
        events = event_level_metrics(frame, thresholds=THRESHOLDS)
        if not events.empty:
            events.insert(0, "split", split_name)
            event_rows.append(events)
            event_05 = events.loc[
                events["threshold"].eq(0.5)
                & events["aggregation"].eq("early_mean_prob")
            ]
            if not event_05.empty:
                row = event_05.iloc[0]
                summary[f"event_{split_name}_auroc"] = row["auroc"]
                summary[f"event_{split_name}_auprc"] = row["auprc"]
                summary[f"event_{split_name}_f1_at_0p5"] = row["f1"]
                summary[f"event_{split_name}_recall_at_0p5"] = row["recall"]
    return (
        pd.DataFrame([summary]),
        pd.DataFrame(threshold_rows),
        pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame(),
    )


def fit_xgboost(
    config: dict[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
) -> tuple[object, str]:
    """Fit the XGBoost/fallback classifier."""

    y_train = arrays["train"]["y_long_fade"].astype(int)
    scale_pos = positive_weight(y_train)
    model, backend = xgboost_classifier(config, scale_pos_weight=scale_pos)
    x_train = arrays["train"]["X_lag_scalar"]
    if backend == "xgboost":
        model.fit(x_train, y_train)
    else:
        weights = np.where(y_train == 1, scale_pos, 1.0)
        model.fit(x_train, y_train, sample_weight=weights)
    return model, backend


def predict_sklearn(model: object, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return logits and probabilities from a sklearn-style classifier."""

    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(x))[:, 1]
    else:
        prob = np.asarray(model.predict(x), dtype=float)
    prob = np.clip(prob.astype(float), 1e-7, 1.0 - 1e-7)
    logits = np.log(prob / (1.0 - prob))
    return logits.astype(np.float32), prob.astype(np.float32)


def train_neural(
    config: dict[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
    *,
    model_id: str,
    device_name: str,
    max_epochs_override: int | None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a neural classifier with validation AUPRC selection."""

    training = config["training"]
    set_seed(int(training.get("seed", 42)))
    device = torch.device(device_name)
    model_config = dict(config.get("model", {}))
    if model_id == MODEL_ID_SHAPELET_CONV:
        model_config["shapelets"] = config.get("shapelets", {})
    model = build_neural_classifier(
        model_id,
        context_length=int(config["dataset"]["context_length"]),
        scalar_context_dim=int(arrays["train"]["scalar_context_features"].shape[1]),
        config=model_config,
    ).to(device)
    train_dataset = LongFadeDataset(
        arrays["train"],
        sequence_key=str(config["dataset"]["sequence_input_key"]),
    )
    val_dataset = LongFadeDataset(
        arrays["validation"],
        sequence_key=str(config["dataset"]["sequence_input_key"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 256)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training.get("batch_size", 256)),
        shuffle=False,
    )
    pos_weight = torch.tensor(
        [positive_weight(arrays["train"]["y_long_fade"])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
    )
    max_epochs = int(max_epochs_override or training.get("max_epochs", 80))
    patience = int(training.get("early_stopping_patience", 12))
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    best_state = None
    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for x_sequence, x_scalar, y in train_loader:
            x_sequence = x_sequence.to(device)
            x_scalar = x_scalar.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_sequence, x_scalar)
            loss = criterion(logits, y)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_logits, val_prob = predict_neural(model, val_loader, device=device)
        val_metrics = sample_classification_metrics(
            arrays["validation"]["y_long_fade"],
            val_prob,
        )
        score = float(val_metrics["auprc"])
        if not np.isfinite(score):
            score = -float(np.mean(train_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_auprc": val_metrics["auprc"],
                "val_auroc": val_metrics["auroc"],
                "val_f1_at_0p5": val_metrics["f1_at_0p5"],
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "backend": "pytorch",
        "best_epoch": best_epoch,
        "history": history,
        "pos_weight": float(pos_weight.item()),
    }


def predict_neural(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return logits and probabilities from a neural classifier."""

    model.eval()
    logits = []
    with torch.no_grad():
        for x_sequence, x_scalar, _ in loader:
            output = model(x_sequence.to(device), x_scalar.to(device))
            logits.append(output.detach().cpu().numpy())
    logit = np.concatenate(logits).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return logit, prob.astype(np.float32)


def neural_predictions(
    model: nn.Module,
    arrays: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, pd.DataFrame],
    *,
    config: dict[str, Any],
    model_id: str,
    run_id: str,
    device_name: str,
) -> dict[str, pd.DataFrame]:
    """Build prediction frames for all splits."""

    device = torch.device(device_name)
    output = {}
    for split in ["train", "validation", "test"]:
        dataset = LongFadeDataset(
            arrays[split],
            sequence_key=str(config["dataset"]["sequence_input_key"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"].get("batch_size", 256)),
            shuffle=False,
        )
        logits, probs = predict_neural(model, loader, device=device)
        output[split] = prediction_frame(
            metadata[split],
            arrays=arrays[split],
            split=split,
            logits=logits,
            probabilities=probs,
            model_id=model_id,
            run_id=run_id,
        )
    return output


def save_outputs(
    *,
    config: dict[str, Any],
    run_id: str,
    model_id: str,
    backend: str,
    run_path: Path,
    checkpoint_path: Path,
    predictions: dict[str, pd.DataFrame],
    metrics_summary: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    extra_metadata: dict[str, Any],
) -> None:
    """Write run artifacts."""

    for child in ["metrics", "predictions", "figures", "tables"]:
        (run_path / child).mkdir(parents=True, exist_ok=True)
    metrics_summary.to_csv(run_path / "metrics" / "metrics_summary.csv", index=False)
    threshold_metrics.to_csv(run_path / "metrics" / "threshold_metrics.csv", index=False)
    event_metrics.to_csv(run_path / "metrics" / "event_metrics.csv", index=False)
    all_predictions = pd.concat(predictions.values(), ignore_index=True)
    all_predictions.to_parquet(run_path / "predictions" / "predictions.parquet", index=False)
    for split, frame in predictions.items():
        stem = "val" if split == "validation" else split
        frame.to_parquet(run_path / "predictions" / f"{stem}_predictions.parquet", index=False)
    save_yaml(run_path / "config_resolved.yaml", config)
    save_yaml(
        run_path / "metadata.yaml",
        {
            "task_name": "long_fade_detection",
            "model_id": model_id,
            "run_id": run_id,
            "selection_id": selection_id_from_run_id(run_id),
            "backend": backend,
            "results_path": relative_project_path(run_path),
            "model_path": relative_project_path(checkpoint_path),
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **extra_metadata,
        },
    )


def main() -> None:
    """Run one configured long-fade detection experiment."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != "long_fade_detection":
        raise ValueError("Config task_name must be long_fade_detection.")
    model_id = str(config["model_id"])
    event_config = config["event_definition"]
    source_config = config["source"]
    selection_id = make_selection_id(str(source_config["external_test_dataset"]))
    run_id = make_run_id(
        threshold_db=float(event_config["threshold_db"]),
        min_fade_duration_seconds=int(event_config["min_fade_duration_seconds"]),
        model_id=model_id,
        selection_id=selection_id,
        run_suffix=args.run_suffix,
    )
    run_path = run_dir(run_id)
    checkpoint_path = model_dir(run_id)
    overwrite = bool(config.get("output", {}).get("overwrite", False)) or args.force
    if run_path.exists() and any(run_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Run already exists: {run_path}. Use --force.")
    if args.dry_run:
        print(f"Would run {model_id} as {run_id}")
        print(f"Dataset: {config['dataset']['path']}")
        return
    if run_path.exists() and overwrite:
        shutil.rmtree(run_path)
    if checkpoint_path.exists() and overwrite:
        shutil.rmtree(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    dataset_path = project_path(config["dataset"]["path"])
    arrays, metadata = load_split_data(dataset_path)
    set_seed(int(config["training"].get("seed", 42)))

    print("=== Long-Fade Detection Run ===")
    print(f"Model: {model_id}")
    print(f"Run ID: {run_id}")
    print(f"Dataset: {dataset_path.relative_to(PROJECT_ROOT)}")

    if model_id == MODEL_ID_XGBOOST:
        model, backend = fit_xgboost(config, arrays)
        joblib.dump(model, checkpoint_path / "model.joblib")
        predictions = {}
        for split in ["train", "validation", "test"]:
            logits, probs = predict_sklearn(model, arrays[split]["X_lag_scalar"])
            predictions[split] = prediction_frame(
                metadata[split],
                arrays=arrays[split],
                split=split,
                logits=logits,
                probabilities=probs,
                model_id=model_id,
                run_id=run_id,
            )
        extra_metadata = {"best_epoch": None, "pos_weight": positive_weight(arrays["train"]["y_long_fade"])}
    else:
        requested_device = args.device or "auto"
        device_name = select_device() if requested_device == "auto" else requested_device
        model, train_metadata = train_neural(
            config,
            arrays,
            model_id=model_id,
            device_name=device_name,
            max_epochs_override=args.max_epochs,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "model_id": model_id,
                "run_id": run_id,
            },
            checkpoint_path / "best_model.pt",
        )
        (run_path / "tables").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_metadata["history"]).to_csv(
            run_path / "tables" / "training_history.csv",
            index=False,
        )
        predictions = neural_predictions(
            model,
            arrays,
            metadata,
            config=config,
            model_id=model_id,
            run_id=run_id,
            device_name=device_name,
        )
        backend = train_metadata["backend"]
        extra_metadata = {
            "selected_device": device_name,
            "best_epoch": train_metadata["best_epoch"],
            "pos_weight": train_metadata["pos_weight"],
        }

    metrics_summary, threshold_table, event_table = evaluate_predictions(predictions)
    metrics_summary.insert(0, "run_id", run_id)
    metrics_summary.insert(0, "model_id", model_id)
    save_outputs(
        config=config,
        run_id=run_id,
        model_id=model_id,
        backend=backend,
        run_path=run_path,
        checkpoint_path=checkpoint_path,
        predictions=predictions,
        metrics_summary=metrics_summary,
        threshold_metrics=threshold_table,
        event_metrics=event_table,
        extra_metadata=extra_metadata,
    )
    print(
        f"Completed {run_id}: "
        f"val AUPRC={metrics_summary['val_auprc'].iloc[0]:.4g}, "
        f"test AUPRC={metrics_summary['test_auprc'].iloc[0]:.4g}"
    )
    print(f"Results: {run_path.relative_to(PROJECT_ROOT)}")
    print(f"Model: {checkpoint_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
