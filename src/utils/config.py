"""YAML configuration helpers."""

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {source}")
    return config


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Save a mapping as readable YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a configuration mapping."""

    serialized = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
