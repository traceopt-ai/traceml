"""
CUDA Event Pool for efficient event reuse.

This module provides a thread-safe pool of reusable CUDA events
to avoid the overhead of creating new events on every hook call.
"""

from collections import deque
from typing import Optional
import torch
import threading


class CUDAEventPool:
    """
    Thread-safe pool of reusable CUDA events.
    """

    def __init__(self, max_size: int = 200):
        self._pool = deque(maxlen=max_size)
        self.max_size = max_size
        self._lock = threading.Lock()

    def acquire(self) -> torch.cuda.Event:
        """
        Get a CUDA event from the pool, or create a new one if pool is empty.
        """
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return torch.cuda.Event(enable_timing=True)

    def release(self, event: Optional[torch.cuda.Event]) -> None:
        """
        Return a CUDA event to the pool for reuse.
        """
        if event is None:
            return

        with self._lock:
            if event is not None and len(self._pool) < self.max_size:
              self._pool.append(event)

    def clear(self) -> None:
        """Clear all events from the pool."""
        with self._lock:
            self._pool.clear()


# Global event pool instance
_event_pool = CUDAEventPool(max_size=2000)


def get_cuda_event() -> torch.cuda.Event:
    """
    Acquire a CUDA event from the global pool.
    """
    return _event_pool.acquire()


def return_cuda_event(event: Optional[torch.cuda.Event]) -> None:
    """
    Return a CUDA event to the global pool.
    """
    _event_pool.release(event)


def clear_event_pool() -> None:
    """Clear the global event pool."""
    _event_pool.clear()