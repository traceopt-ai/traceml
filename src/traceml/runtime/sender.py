# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry publishing for the per-rank runtime agent."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from traceml.loggers.error_log import get_error_logger


class TelemetryPublisher:
    """
    Best-effort publisher for sampler payloads.

    The publisher owns the rank identity attached to outbound sampler senders.
    That identity must be globally unique for multi-node jobs; samplers can
    still receive local rank separately when they need node-local device state.
    """

    def __init__(
        self,
        *,
        tcp_client: Any,
        global_rank: Optional[int] = None,
        rank: Optional[int] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self._tcp_client = tcp_client
        self._rank = self._resolve_sender_rank(
            global_rank=global_rank,
            legacy_rank=rank,
        )
        self._logger = logger or get_error_logger("TelemetryPublisher")

    def attach_senders(self, samplers: Iterable[Any]) -> None:
        """Attach sampler senders to the runtime TCP client."""
        for sampler in samplers:
            sender = getattr(sampler, "sender", None)
            if sender is None:
                continue
            try:
                sender.sender = self._tcp_client
                sender.rank = self._rank
            except Exception as exc:
                self._log_exception(
                    f"{self._sampler_name(sampler)}.sender attach failed",
                    exc,
                )

    def publish(self, samplers: Iterable[Any]) -> None:
        """Flush sampler writers, collect payloads, and send one batch."""
        sampler_list = list(samplers)
        self.flush_writers(sampler_list)
        batch = self.collect_payloads(sampler_list)
        self.send_batch(batch)

    def flush_writers(self, samplers: Iterable[Any]) -> None:
        """Flush local sampler DB writers, if present."""
        for sampler in samplers:
            db = getattr(sampler, "db", None)
            writer = getattr(db, "writer", None) if db is not None else None
            flush = getattr(writer, "flush", None)
            if flush is None:
                continue
            try:
                flush()
            except Exception as exc:
                self._log_exception(
                    f"{self._sampler_name(sampler)}.writer.flush failed",
                    exc,
                )

    def collect_payloads(self, samplers: Iterable[Any]) -> List[Any]:
        """Collect incremental payloads from sampler senders."""
        batch: List[Any] = []
        for sampler in samplers:
            sender = getattr(sampler, "sender", None)
            collect_payload = getattr(sender, "collect_payload", None)
            if collect_payload is None:
                continue
            try:
                payload = collect_payload()
            except Exception as exc:
                self._log_exception(
                    f"{self._sampler_name(sampler)}.collect_payload failed",
                    exc,
                )
                continue
            if payload is not None:
                batch.append(payload)
        return batch

    def send_batch(self, batch: List[Any]) -> None:
        """Send a batch of telemetry payloads."""
        if not batch:
            return
        try:
            self._tcp_client.send_batch(batch)
        except Exception as exc:
            self._log_exception("TCPClient.send_batch failed", exc)

    def close(self) -> None:
        """Close the underlying TCP client best-effort."""
        try:
            self._tcp_client.close()
        except Exception as exc:
            self._log_exception("TCPClient.close failed", exc)

    def _log_exception(self, label: str, exc: Exception) -> None:
        """Log an exception without raising."""
        log = getattr(self._logger, "exception", None)
        if callable(log):
            log("[TraceML] %s: %s", label, exc)
            return

        fallback = getattr(self._logger, "error", None)
        if callable(fallback):
            fallback(f"[TraceML] {label}: {exc}")

    @staticmethod
    def _sampler_name(sampler: Any) -> str:
        return str(
            getattr(
                sampler,
                "sampler_name",
                sampler.__class__.__name__,
            )
        )

    @staticmethod
    def _resolve_sender_rank(
        *,
        global_rank: Optional[int],
        legacy_rank: Optional[int],
    ) -> int:
        """
        Return the rank value attached to outbound telemetry.

        ``global_rank`` is the canonical argument. ``rank`` is accepted as a
        compatibility shim for existing internal tests/callers and should not be
        used by new code.
        """
        if global_rank is not None:
            return int(global_rank)
        if legacy_rank is not None:
            return int(legacy_rank)
        raise ValueError("TelemetryPublisher requires global_rank")


__all__ = [
    "TelemetryPublisher",
]
