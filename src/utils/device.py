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
