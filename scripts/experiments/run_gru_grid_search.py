"""Compatibility wrapper for the task-scoped GRU grid-search entrypoint."""

from scripts.experiments.autoregressive.run_gru_grid_search import *  # noqa: F401,F403
from scripts.experiments.autoregressive.run_gru_grid_search import main


if __name__ == "__main__":
    main()

