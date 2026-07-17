from __future__ import annotations

from src.tasks.survival_persistence.data.features import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    SCALAR_CONTEXT_FEATURE_NAMES,
)


def test_forbidden_target_fields_are_not_scalar_inputs() -> None:
    assert not (set(SCALAR_CONTEXT_FEATURE_NAMES) & FORBIDDEN_MODEL_INPUT_COLUMNS)

