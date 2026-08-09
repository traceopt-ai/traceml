# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The error log must actually receive errors logged by TraceML code.

Regression test for the logger-name mismatch where the rotating file
handler lived on ``"traceml"`` while every call site logged under
``"traceml_ai.*"``, leaving every ``traceml_errors.log`` empty.
"""

import logging

import pytest

from traceml_ai.loggers.error_log import get_error_logger, setup_error_logger


@pytest.fixture
def clean_error_logger():
    """Detach handlers before and after so logger state cannot leak."""

    def _reset():
        logger = logging.getLogger("traceml_ai")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = True

    _reset()
    yield
    _reset()


def test_error_from_call_site_lands_in_file(
    tmp_path, monkeypatch, clean_error_logger
):
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "session_test")

    setup_error_logger(is_aggregator=True)
    get_error_logger("Probe").error("[TraceML] probe-error-message")

    log_file = tmp_path / "session_test" / "aggregator" / "traceml_errors.log"
    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "probe-error-message" in content
    assert "traceml_ai.Probe" in content


def test_rank_process_writes_its_own_error_file(
    tmp_path, monkeypatch, clean_error_logger
):
    """Each rank keeps its own file, so ranks cannot overwrite each other."""
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "session_test")
    monkeypatch.setenv("RANK", "3")

    setup_error_logger(is_aggregator=False)
    get_error_logger("Sampler").error("[TraceML] rank-scoped-error")

    log_file = tmp_path / "session_test" / "rank_3" / "traceml_errors.log"
    assert log_file.is_file()
    assert "rank-scoped-error" in log_file.read_text(encoding="utf-8")


def test_setup_is_idempotent(tmp_path, monkeypatch, clean_error_logger):
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "session_test")

    first = setup_error_logger(is_aggregator=True)
    second = setup_error_logger(is_aggregator=True)

    assert first is second
    assert len(first.handlers) == 1
