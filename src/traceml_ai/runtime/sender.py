# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry publishing for the per-rank runtime agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from traceml_ai.loggers.error_log import get_error_logger


@dataclass(frozen=True)
class SenderIdentity:
    """
    Runtime identity attached to outbound sampler payloads.

    ``global_rank`` is the stable cross-job worker identity used by storage and
    aggregation. ``local_rank`` and ``local_world_size`` describe node-local
    placement, which is useful for device mapping and multi-node debugging.
    """

    global_rank: int
    local_rank: int
    world_size: int = 1
    local_world_size: int = 1
    node_rank: int = 0
    hostname: str = ""
    pid: int = 0

    @property
    def rank(self) -> int:
        """Compatibility rank used by existing aggregator/storage code."""
        return self.global_rank

    def to_payload_fields(self) -> Dict[str, Any]:
        """Return flat identity fields for the current transport envelope."""
        return {
            "rank": self.rank,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_rank": self.node_rank,
            "hostname": self.hostname,
            "pid": self.pid,
        }


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
        identity: SenderIdentity,
        logger: Optional[Any] = None,
    ) -> None:
        self._tcp_client = tcp_client
        self._identity = identity
        self._logger = logger or get_error_logger("TelemetryPublisher")

    def attach_senders(self, samplers: Iterable[Any]) -> None:
        """Attach sampler senders to the runtime TCP client."""
        for sampler in samplers:
            sender = getattr(sampler, "sender", None)
            if sender is None:
                continue
            try:
                sender.sender = self._tcp_client
                sender.identity = self._identity
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

    def send_control(self, payload: Dict[str, Any]) -> None:
        """Send one internal control payload best-effort."""
        try:
            self._tcp_client.send(payload)
        except Exception as exc:
            self._log_exception("TCPClient.send control failed", exc)

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


__all__ = [
    "SenderIdentity",
    "TelemetryPublisher",
]
