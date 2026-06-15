"""Training helpers for deterministic PatchTST forecasting experiments."""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class PatchTSTForecastDataset(Dataset):
    """Expose one final autoregressive NPZ split as PyTorch tensors."""

    def __init__(self, npz_path: str | Path, *, variant: str) -> None:
        if variant == "raw":
            X_key, y_key = "X_raw", "y_raw"
        elif variant == "context_standard":
            X_key, y_key = "X_context_standard", "y_context_standard"
        else:
            raise ValueError(f"Unsupported dataset variant: {variant}")
        with np.load(npz_path) as arrays:
            self.X = torch.from_numpy(arrays[X_key].astype(np.float32, copy=False))
            self.y = torch.from_numpy(arrays[y_key].astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.X[index], self.y[index], index


@dataclass(frozen=True)
class PatchTSTTrainingResult:
    """Best and last model states plus epoch-level optimization history."""

    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float | int]]
    best_state_dict: dict[str, torch.Tensor]
    last_state_dict: dict[str, torch.Tensor]


def create_patchtst_data_loader(
    npz_path: str | Path,
    *,
    variant: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic data loader for one final dataset split."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PatchTSTForecastDataset(npz_path, variant=variant),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_patchtst_forecaster(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    early_stopping_patience: int,
    gradient_clip_norm: float,
) -> PatchTSTTrainingResult:
    """Train with MSE and retain validation-best and final model states."""

    if learning_rate <= 0 or max_epochs < 1 or early_stopping_patience < 1:
        raise ValueError("Training rate, epochs, and patience must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")
    if len(train_loader.dataset) == 0 or len(validation_loader.dataset) == 0:
        raise ValueError("Training and validation splits must both be non-empty.")

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model.to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = run_patchtst_epoch(
            model,
            train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_loss = run_patchtst_epoch(
            model,
            validation_loader,
            loss_function=loss_function,
            device=device,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return PatchTSTTrainingResult(
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        history=history,
        best_state_dict=best_state,
        last_state_dict=last_state,
    )


def run_patchtst_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> float:
    """Run one train or evaluation epoch and return mean sample loss."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            loss = loss_function(model(X), y)
        if training:
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        total_loss += float(loss.detach()) * len(X)
        total_samples += len(X)
    if total_samples == 0:
        raise ValueError("Cannot run an epoch on an empty data loader.")
    return total_loss / total_samples


def predict_patchtst_forecaster(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict deterministically and return predictions plus aligned row indices."""

    if len(loader.dataset) == 0:
        raise ValueError("Cannot predict an empty data split.")
    model.eval()
    predictions = []
    indices = []
    with torch.no_grad():
        for X, _, batch_indices in loader:
            predictions.append(model(X.to(device)).cpu().numpy())
            indices.append(batch_indices.numpy())
    all_predictions = np.concatenate(predictions)
    all_indices = np.concatenate(indices)
    order = np.argsort(all_indices)
    return all_predictions[order], all_indices[order]
