"""
Telemetry publishing for the per-rank runtime agent.

The runtime samples telemetry; this module publishes it. Keeping those
responsibilities separate makes the runtime loop easier to reason about and
lets contributors test transport behavior without starting torchrun.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from traceml.loggers.error_log import get_error_logger


class TelemetryPublisher:
    """
    Attach sampler senders and publish incremental telemetry batches.

    Reliability contract
    --------------------
    Publishing is best-effort and fail-open. Telemetry errors are logged through
    TraceML's error logger and never propagate to the user training script.

    Responsibilities
    ----------------
    - Attach each sampler's incremental sender to the shared TCP client.
    - Flush sampler-owned local writers after sampling.
    - Collect incremental sender payloads.
    - Send all ready payloads as one TCP batch.
    - Close the transport during runtime shutdown.
    """

    def __init__(
        self,
        *,
        tcp_client: Any,
        rank: int,
        logger: Optional[Any] = None,
    ) -> None:
        self._tcp_client = tcp_client
        self._rank = int(rank)
        self._logger = logger or get_error_logger("TelemetryPublisher")

    def attach_senders(self, samplers: Iterable[Any]) -> None:
        """
        Attach sampler senders to the runtime TCP client.

        Samplers without a sender are valid. For example, nonzero-rank
        stdout/stderr samplers persist logs locally but intentionally do not
        send those lines to the aggregator.
        """
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
        """
        Flush sampler writers, collect ready payloads, and send one batch.

        This method is safe to call every runtime tick. It tolerates individual
        sampler failures and still attempts to publish payloads from the
        remaining samplers.
        """
        sampler_list = list(samplers)
        self.flush_writers(sampler_list)
        batch = self.collect_payloads(sampler_list)
        self.send_batch(batch)

    def flush_writers(self, samplers: Iterable[Any]) -> None:
        """
        Flush local sampler DB writers, if present.
        """
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
        """
        Collect incremental payloads from sampler senders.

        Returns only non-empty payloads. If one sender fails, the failure is
        logged and collection continues for the rest.
        """
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
        """
        Send a batch of telemetry payloads.
        """
        if not batch:
            return
        try:
            self._tcp_client.send_batch(batch)
        except Exception as exc:
            self._log_exception("TCPClient.send_batch failed", exc)

    def close(self) -> None:
        """
        Close the underlying TCP client best-effort.
        """
        try:
            self._tcp_client.close()
        except Exception as exc:
            self._log_exception("TCPClient.close failed", exc)

    def _log_exception(self, label: str, exc: Exception) -> None:
        """
        Log an exception without raising.
        """
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
    "TelemetryPublisher",
]
