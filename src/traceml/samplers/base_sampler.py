"""
Shared sampler base class for TraceML.

This module provides the common infrastructure used by all samplers:
- in-memory database creation
- incremental sender creation
- logger setup
- table naming
- small persistence helpers

It intentionally does not impose lifecycle or aggregation behavior on
concrete samplers. Domain-specific logic stays in each sampler subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from traceml.database.database import Database
from traceml.database.database_sender import DBIncrementalSender
from traceml.loggers.error_log import get_error_logger


class BaseSampler(ABC):
    """
    Abstract base class for all TraceML samplers.

    Parameters
    ----------
    sampler_name:
        Logical sampler identity used for the local database and transport
        payloads.
    table_name:
        Default table used by `_add_record()`. Concrete samplers may still
        override the table per call when needed.
    max_rows_per_flush:
        Sender-side transport cap. `-1` preserves the default "send all new
        rows since the last flush" behavior.
    """

    def __init__(
        self,
        sampler_name: str,
        table_name: Optional[str] = None,
        max_rows_per_flush: int = -1,
    ) -> None:
        self.sampler_name = str(sampler_name)
        self.table_name = table_name
        self.logger = get_error_logger(self.sampler_name)

        self.db = Database(sampler_name=self.sampler_name)
        self.sender = DBIncrementalSender(
            db=self.db,
            sampler_name=self.sampler_name,
        )
        if max_rows_per_flush != -1:
            self.sender.max_rows_per_flush = int(max_rows_per_flush)

        self.enable_send: bool = True

    def _add_record(
        self,
        payload: dict[str, Any],
        table_name: Optional[str] = None,
    ) -> None:
        """
        Persist one record into the sampler-owned in-memory database.
        """
        target_table = table_name or self.table_name
        if not target_table:
            raise ValueError(
                f"{self.sampler_name} has no default table name configured."
            )
        self.db.add_record(target_table, payload)

    @abstractmethod
    def sample(self) -> None:
        """
        Collect one best-effort telemetry sample.

        Concrete samplers must swallow their own failures and never interfere
        with training.
        """
        raise NotImplementedError("Must be implemented by subclasses.")
