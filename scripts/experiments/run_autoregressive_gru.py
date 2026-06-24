"""Compatibility wrapper for the task-scoped GRU autoregressive entrypoint."""

from scripts.experiments.autoregressive.run_autoregressive_gru import *  # noqa: F401,F403
from scripts.experiments.autoregressive.run_autoregressive_gru import main


if __name__ == "__main__":
    main()

