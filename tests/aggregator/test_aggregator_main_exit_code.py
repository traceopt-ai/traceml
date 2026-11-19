"""Exit-code handling of ``run_aggregator``."""

from __future__ import annotations

import ast
import inspect
from unittest.mock import Mock

from traceml_ai.aggregator import aggregator_main
from traceml_ai.aggregator.aggregator_main import run_aggregator
from traceml_ai.runtime.settings import AggregatorEndpoint, TraceMLSettings


def test_run_aggregator_exits_zero_after_a_clean_shutdown(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        aggregator_main, "_install_signal_handlers", lambda _event: None
    )
    handle = Mock()
    handle.endpoint = AggregatorEndpoint("127.0.0.1", 1234, "run")

    def _start(_settings, *, logger, stop_event):
        stop_event.set()
        return handle

    monkeypatch.setattr(aggregator_main, "start_aggregator", _start)
    logger = Mock()

    rc = run_aggregator(
        TraceMLSettings(
            mode="summary",
            logs_dir=str(tmp_path),
            session_id="run",
            finalize_timeout_sec=0.0,
        ),
        logger=logger,
    )

    assert rc == 0
    handle.stop.assert_called_once()
    logger.error.assert_not_called()
    assert "Aggregator stopped." in capsys.readouterr().err


def test_run_aggregator_does_not_return_from_a_finally_block():
    """PEP 765: Python 3.14 warns about ``return`` in ``finally`` at compile
    time, and the aggregator is compiled from source on every ``traceml run``.
    """
    tree = ast.parse(inspect.getsource(run_aggregator))

    returns_in_finally = [
        node.lineno
        for try_node in ast.walk(tree)
        if isinstance(try_node, ast.Try)
        for statement in try_node.finalbody
        for node in ast.walk(statement)
        if isinstance(node, ast.Return)
    ]

    assert returns_in_finally == []
