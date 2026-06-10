"""Reproducibility helpers for standard and optional scientific libraries."""

import random


def set_seed(seed: int) -> None:
    """Seed Python and installed numerical or deep-learning libraries."""

    random.seed(seed)

    try:
        import numpy
    except ModuleNotFoundError:
        pass
    else:
        numpy.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
