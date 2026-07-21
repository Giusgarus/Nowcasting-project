"""YAML configuration helpers."""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is a project dependency.
    np = None


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {source}")
    return config


def to_yaml_safe(value: Any) -> Any:
    """Convert common scientific-Python values into YAML-safe objects."""

    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_yaml_safe(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_yaml_safe(item) for item in value]
    return value


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Save a mapping as readable YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(to_yaml_safe(data), stream, sort_keys=False)


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a configuration mapping."""

    serialized = yaml.safe_dump(to_yaml_safe(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
