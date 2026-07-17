"""Reusable device-aware parallel execution for independent tuning trials."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from typing import Any
from warnings import warn

from src.utils.device import available_accelerator_devices, select_device


def prepare_trial_device(device_name: str):
    """Activate and return one explicit PyTorch worker device."""

    import torch

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def choose_trial_devices(
    parallel_config: Mapping[str, Any],
    *,
    cuda_count: int | None = None,
    fallback_device: str | None = None,
) -> list[str]:
    """Choose currently available devices for independent trial workers."""

    fallback = fallback_device or available_accelerator_devices()[0]
    if not bool(parallel_config.get("enabled", False)):
        return [fallback]

    if cuda_count is None:
        try:
            import torch
        except ModuleNotFoundError:
            cuda_count = 0
        else:
            cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if cuda_count < 1:
        return [fallback]

    configured_ids = parallel_config.get("cuda_device_ids", "auto")
    if configured_ids == "auto":
        device_ids = list(range(cuda_count))
    elif isinstance(configured_ids, list):
        device_ids = [
            int(device_id)
            for device_id in configured_ids
            if 0 <= int(device_id) < cuda_count
        ]
    else:
        raise ValueError("parallel.cuda_device_ids must be 'auto' or a list.")
    if not device_ids:
        return [fallback]
    device_ids = list(dict.fromkeys(device_ids))

    max_workers = parallel_config.get("max_workers", "auto")
    if max_workers != "auto":
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("parallel.max_workers must be positive or 'auto'.")
        device_ids = device_ids[:max_workers]
    return [f"cuda:{device_id}" for device_id in device_ids]


def iter_parallel_trial_results(
    jobs: Iterable[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Any],
    devices: list[str],
) -> Iterator[Any]:
    """Yield trial results while keeping at most one active job per device."""

    if not devices:
        raise ValueError("At least one trial device is required.")
    if len(set(devices)) != len(devices):
        raise ValueError("Trial devices must be unique.")
    job_iterator = iter(jobs)
    if len(devices) == 1:
        for job in job_iterator:
            yield worker({**job, "device": devices[0]})
        return

    context = get_context("spawn")
    executors = {}
    try:
        for device in devices:
            executors[device] = ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
            )
    except (NotImplementedError, OSError, PermissionError) as error:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
        warn(
            "Parallel worker processes are unavailable; running trials "
            f"sequentially instead. Reason: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        for index, job in enumerate(job_iterator):
            yield worker({**job, "device": devices[index % len(devices)]})
        return
    pending: dict[Future, tuple[str, ProcessPoolExecutor]] = {}
    try:
        for device, executor in executors.items():
            job = next(job_iterator, None)
            if job is None:
                break
            pending[executor.submit(worker, {**job, "device": device})] = (
                device,
                executor,
            )

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                device, executor = pending.pop(future)
                yield future.result()
                job = next(job_iterator, None)
                if job is not None:
                    pending[executor.submit(worker, {**job, "device": device})] = (
                        device,
                        executor,
                    )
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
