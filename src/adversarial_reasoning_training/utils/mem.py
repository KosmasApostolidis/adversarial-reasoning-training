"""GPU memory probe for gate T0 and periodic training monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryStats:
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float
    peak_reserved_gb: float
    device: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
            "peak_allocated_gb": self.peak_allocated_gb,
            "peak_reserved_gb": self.peak_reserved_gb,
            "device": self.device,
        }


def current_memory_stats(device: int | str | None = None) -> MemoryStats:
    """Return current + peak memory for the selected CUDA device."""
    if not torch.cuda.is_available():
        _log.warning(
            "current_memory_stats called but CUDA unavailable — "
            "returning zero stats with device='cpu'. GPU memory "
            "accounting in train_meta.json will be 0.0 GiB."
        )
        return MemoryStats(0.0, 0.0, 0.0, 0.0, "cpu")
    dev = torch.device(f"cuda:{device}") if isinstance(device, int) else (
        torch.device(device) if device else torch.device("cuda")
    )
    alloc = torch.cuda.memory_allocated(dev) / 1e9
    reserved = torch.cuda.memory_reserved(dev) / 1e9
    peak_alloc = torch.cuda.max_memory_allocated(dev) / 1e9
    peak_reserved = torch.cuda.max_memory_reserved(dev) / 1e9
    return MemoryStats(alloc, reserved, peak_alloc, peak_reserved, str(dev))


def reset_peak_memory(device: int | str | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def assert_peak_under(gb_limit: float, device: int | str | None = None) -> MemoryStats:
    """Raise if peak allocated memory exceeds `gb_limit`. Return stats either way."""
    stats = current_memory_stats(device)
    if stats.peak_allocated_gb > gb_limit:
        raise RuntimeError(
            f"Peak GPU memory {stats.peak_allocated_gb:.2f} GB exceeds limit {gb_limit:.2f} GB"
        )
    return stats
