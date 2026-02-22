from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings


class BaseDisplayDriver(ABC):
    """
    Base class for display drivers.

    A display driver is responsible for:
    - starting/stopping its UI resources
    - updating the UI on each aggregator tick

    The aggregator relies ONLY on this interface:
      - start()
      - tick()
      - stop()
    """

    def __init__(self, logger: Any, store: RemoteDBStore, settings: TraceMLSettings) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings

    @abstractmethod
    def start(self) -> None:
        """Initialize UI resources (best effort)."""
        raise NotImplementedError

    @abstractmethod
    def tick(self) -> None:
        """Perform one UI update cycle (must be safe to call repeatedly)."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Release UI resources (best effort)."""
        raise NotImplementedError