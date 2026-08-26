# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from traceml_ai.telemetry_export.otlp import (
    build_otlp_pipeline,
    otlp_is_configured,
)


def test_otlp_is_disabled_without_explicit_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", raising=False)

    assert not otlp_is_configured()
    assert build_otlp_pipeline() is None


def test_signal_endpoint_enables_otlp(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "http://collector:4318/v1/logs",
    )

    assert otlp_is_configured()
