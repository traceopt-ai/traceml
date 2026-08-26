# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Optional aggregator-side OTLP Logs export.

TraceML owns only the mapping from its normalized records to OpenTelemetry log
records. Queueing, batching, background delivery, and overflow behavior are
delegated to OpenTelemetry's official ``BatchLogRecordProcessor``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional, Sequence

from traceml_ai import __version__
from traceml_ai.telemetry_export.records import ExportRecord
from traceml_ai.telemetry_export.window import WindowProcessor

_HTTP_PROTOCOLS = frozenset({"http/protobuf", "http"})
_GRPC_PROTOCOLS = frozenset({"grpc"})
_DEFAULT_SHUTDOWN_TIMEOUT_SEC = 2.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def otlp_is_configured() -> bool:
    """Return whether a standard OTLP endpoint was explicitly configured."""
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )


def _configured_protocol() -> str:
    return (
        (
            os.environ.get("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL")
            or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
            or "http/protobuf"
        )
        .strip()
        .lower()
    )


class OtlpLogPipeline:
    """Aggregate records and adapt them to the official OTLP processor."""

    def __init__(
        self,
        *,
        protocol: Optional[str] = None,
        shutdown_timeout_sec: Optional[float] = None,
        window_processor: Optional[WindowProcessor] = None,
        logger: Optional[Any] = None,
    ) -> None:
        # Imports stay local so the default TraceML installation has no
        # OpenTelemetry import or dependency requirement.
        from opentelemetry._logs import LogRecord
        from opentelemetry.sdk._logs import LogRecordLimits, ReadWriteLogRecord
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.util.instrumentation import InstrumentationScope

        selected = str(protocol or _configured_protocol()).lower()
        if selected in _HTTP_PROTOCOLS:
            from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                OTLPLogExporter,
            )
        elif selected in _GRPC_PROTOCOLS:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter,
            )
        else:
            raise ValueError(
                "unsupported OTLP Logs protocol "
                f"{selected!r}; use 'http/protobuf' or 'grpc'"
            )

        self._api_log_record = LogRecord
        self._read_write_log_record = ReadWriteLogRecord
        self._log_record_limits = LogRecordLimits()
        self._resource_type = Resource
        self._processor_type = BatchLogRecordProcessor
        self._scope = InstrumentationScope(
            name="traceml",
            version=str(__version__),
        )
        self._exporter = OTLPLogExporter()
        self._resources: dict[tuple[tuple[str, Any], ...], Any] = {}
        self._processor: Optional[Any] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._windows = window_processor or WindowProcessor.from_env(
            logger=logger
        )
        configured_timeout = (
            _env_float(
                "TRACEML_OTLP_SHUTDOWN_TIMEOUT_SEC",
                _DEFAULT_SHUTDOWN_TIMEOUT_SEC,
            )
            if shutdown_timeout_sec is None
            else float(shutdown_timeout_sec)
        )
        self._shutdown_timeout_sec = max(0.0, configured_timeout)

    @property
    def shutdown_timeout_sec(self) -> float:
        """Return TraceML's bounded OTLP drain budget."""
        return self._shutdown_timeout_sec

    def start(self) -> None:
        """Create the SDK batch processor once."""
        with self._lock:
            if self._processor is not None or self._stopped:
                return
            # Constructor defaults intentionally read the standard
            # OTEL_BLRP_* environment variables directly.
            self._processor = self._processor_type(self._exporter)

    def enqueue(self, records: Sequence[ExportRecord]) -> None:
        """Aggregate source records and queue newly completed windows."""
        if not records:
            return
        with self._lock:
            processor = self._processor
            if processor is None or self._stopped:
                return
            for record in self._windows.process(records):
                processor.on_emit(self._to_read_write(record))

    def stop(self, timeout_sec: Optional[float] = None) -> None:
        """Shut down the SDK processor within the aggregator's drain budget."""
        timeout = (
            self._shutdown_timeout_sec
            if timeout_sec is None
            else max(0.0, float(timeout_sec))
        )
        with self._lock:
            if self._stopped:
                return
            processor = self._processor
            if processor is not None:
                for record in self._windows.flush():
                    processor.on_emit(self._to_read_write(record))
            else:
                self._windows.flush()
            self._stopped = True

        # BatchLogRecordProcessor owns all draining and exporter shutdown, but
        # its public shutdown method has no timeout argument. Run that official
        # lifecycle method on a daemon helper so an unavailable collector can
        # never hold TraceML finalization past its own budget.
        shutdown = (
            self._exporter.shutdown
            if processor is None
            else processor.shutdown
        )
        thread = threading.Thread(
            target=shutdown,
            name="TraceMLOtlpShutdown",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout)

    def _to_read_write(self, record: ExportRecord) -> Any:
        api_record = self._api_log_record(
            timestamp=record.timestamp_unix_ns,
            observed_timestamp=record.observed_timestamp_unix_ns,
            body=dict(record.data),
            attributes={"traceml.schema.version": record.schema_version},
            event_name=record.event_name,
        )
        return self._read_write_log_record(
            log_record=api_record,
            resource=self._resource(record),
            instrumentation_scope=self._scope,
            limits=self._log_record_limits,
        )

    def _resource(self, record: ExportRecord) -> Any:
        key = tuple(sorted(record.resource.items()))
        resource = self._resources.get(key)
        if resource is None:
            resource = self._resource_type.create(dict(record.resource))
            self._resources[key] = resource
        return resource


def build_otlp_pipeline(
    *, logger: Optional[Any] = None
) -> Optional[OtlpLogPipeline]:
    """Build the optional pipeline, or return ``None`` when unconfigured."""
    if not otlp_is_configured():
        return None

    try:
        return OtlpLogPipeline(logger=logger)
    except ModuleNotFoundError as exc:
        log = getattr(logger, "warning", None)
        if callable(log):
            log(
                "[TraceML] OTLP endpoint configured but optional dependencies "
                "are missing. Install with: pip install 'traceml-ai[otlp]' "
                "(%s)",
                exc,
            )
        return None
    except Exception as exc:
        log = getattr(logger, "warning", None)
        if callable(log):
            log("[TraceML] OTLP exporter disabled: %s", exc)
        return None


__all__ = ["OtlpLogPipeline", "build_otlp_pipeline", "otlp_is_configured"]
