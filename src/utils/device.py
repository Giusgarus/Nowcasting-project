"""Centralized hardware-device selection."""


def select_device() -> str:
    """Return the best available PyTorch device name.

    PyTorch is optional during initial repository setup. If it is unavailable,
    CPU is returned so non-neural utilities remain usable.
    """

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def available_accelerator_devices() -> list[str]:
    """Return all available accelerators in priority order, then CPU fallback.

    Grid-search runners should use this helper so independent trials can be
    spread across all available CUDA devices. If CUDA is unavailable, use MPS
    when available, otherwise CPU.
    """

    try:
        import torch
    except ModuleNotFoundError:
        return ["cpu"]

    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return ["mps"]

    return ["cpu"]
