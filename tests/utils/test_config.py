from pathlib import Path

import numpy as np

from src.utils.config import config_fingerprint, load_yaml_config, save_yaml


def test_save_yaml_accepts_numpy_scalars(tmp_path: Path) -> None:
    path = tmp_path / "metadata.yaml"
    data = {
        "np_int": np.int64(293991),
        "np_float": np.float32(1.25),
        "nested": {"values": (np.int32(1), np.float64(2.5))},
    }

    save_yaml(path, data)
    loaded = load_yaml_config(path)

    assert loaded == {
        "np_int": 293991,
        "np_float": 1.25,
        "nested": {"values": [1, 2.5]},
    }


def test_config_fingerprint_accepts_numpy_scalars() -> None:
    fingerprint = config_fingerprint({"value": np.int64(293991)})

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64
