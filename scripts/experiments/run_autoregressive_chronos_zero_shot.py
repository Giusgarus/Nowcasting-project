"""Compatibility wrapper for the task-scoped Chronos autoregressive entrypoint."""

from scripts.experiments.autoregressive.run_autoregressive_chronos_zero_shot import *  # noqa: F401,F403
from scripts.experiments.autoregressive.run_autoregressive_chronos_zero_shot import main


if __name__ == "__main__":
    main()

