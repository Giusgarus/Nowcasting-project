"""Compatibility wrapper for the task-scoped PatchTST autoregressive entrypoint."""

from scripts.experiments.autoregressive.run_autoregressive_patchtst import *  # noqa: F401,F403
from scripts.experiments.autoregressive.run_autoregressive_patchtst import main


if __name__ == "__main__":
    main()

