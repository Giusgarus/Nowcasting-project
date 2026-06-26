"""Readable progress formatting for model-selection trials."""

from collections.abc import Mapping
from typing import Any

SECTION_ORDERS = {
    "model": [
        "architecture",
        "variant",
        "input_size",
        "input_channels",
        "context_length",
        "prediction_length",
        "hidden_size",
        "num_layers",
        "bidirectional",
        "dropout",
        "patch_len",
        "stride",
        "d_model",
        "n_heads",
    ],
    "optimizer": ["name", "learning_rate", "weight_decay"],
    "training": [
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
        "mixed_precision",
        "seed",
        "teacher_forcing_ratio",
        "loss",
        "skip_if_checkpoint_exists",
        "force_retrain",
        "resume_from_checkpoint",
    ],
}


def format_trial_start(
    *,
    index: int,
    total: int,
    run_id: str,
    trial_id: str,
    device: object,
    parameters: Mapping[str, Any],
) -> str:
    """Return a compact multi-line description for one trial start."""

    width = max(len(str(total)), 3)
    lines = [
        f"[{index:0{width}d}/{total:0{width}d}] {run_id} | {trial_id} | device={device}"
    ]
    architecture = _first_present(
        parameters,
        "architecture",
        section="model",
    )
    variant = _first_present(parameters, "variant", section="model")
    if architecture is not None or variant is not None:
        lines.append(
            "  run: "
            + _format_pairs(
                {
                    key: value
                    for key, value in {
                        "architecture": architecture,
                        "variant": variant,
                    }.items()
                    if value is not None
                }
            )
        )
    for section in ("model", "optimizer", "training"):
        values = parameters.get(section)
        if isinstance(values, Mapping):
            display = _without_keys(values, {"architecture", "variant"})
            if display:
                lines.append(f"  {section}: {_format_section(section, display)}")
    return "\n".join(lines)


def _first_present(
    parameters: Mapping[str, Any],
    key: str,
    *,
    section: str,
) -> Any:
    if key in parameters:
        return parameters[key]
    section_values = parameters.get(section)
    if isinstance(section_values, Mapping):
        return section_values.get(key)
    return None


def _without_keys(values: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key not in keys}


def _format_section(section: str, values: Mapping[str, Any]) -> str:
    ordered = []
    seen = set()
    for key in SECTION_ORDERS.get(section, []):
        if key in values:
            ordered.append((key, values[key]))
            seen.add(key)
    for key in sorted(set(values) - seen):
        ordered.append((key, values[key]))
    if section == "optimizer" and ordered and ordered[0][0] == "name":
        name = _format_value(ordered[0][1])
        rest = _format_pairs(dict(ordered[1:]))
        return f"{name}, {rest}" if rest else name
    return _format_pairs(dict(ordered))


def _format_pairs(values: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{key}={_format_value(value)}"
        for key, value in values.items()
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
