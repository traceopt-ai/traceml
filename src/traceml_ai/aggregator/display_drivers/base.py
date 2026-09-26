from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from traceml_ai.runtime.settings import TraceMLSettings


class BaseDisplayDriver(ABC):
    """
    Base class for display drivers.

    A display driver is responsible for:
    - starting/stopping its UI resources
    - updating the UI on each aggregator tick

    The aggregator relies ONLY on this interface:
      - start()
      - tick()
      - run_finished() (optional; a no-op unless overridden)
      - stop()
    """

    def __init__(self, logger: Any, settings: TraceMLSettings) -> None:
        self._logger = logger
        self._settings = settings

    @abstractmethod
    def start(self) -> None:
        """Initialize UI resources (best effort)."""
        raise NotImplementedError

    @abstractmethod
    def tick(self) -> None:
        """Perform one UI update cycle (must be safe to call repeatedly)."""
        raise NotImplementedError

    def run_finished(self) -> None:
        """Every rank of the run has reported finished (best effort).

        Called from the aggregator's loop thread, once per run, when every
        rank the run expects has sent its finish marker. Ticks continue
        after it, and a later run on the same aggregator (``traceml
        serve``) sends data again and calls this again when it finishes.
        A driver that shows no staleness has nothing to do.
        """
        return None

    @abstractmethod
    def stop(self) -> None:
        """Release UI resources (best effort)."""
        raise NotImplementedError
