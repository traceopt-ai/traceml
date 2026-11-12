"""Tests for publishing the TraceML diagnosis into a W&B run.

The real W&B client is never used here: a stand-in ``wandb`` module records
what the helper writes, and its ``init``/``login``/``finish`` raise so a test
fails if the helper ever owns the run lifecycle.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import traceml_ai
from tests.reporting.html.conftest import _make_payload
from traceml_ai.integrations import wandb as helper
from traceml_ai.sdk.summary_projection import compact_summary

_SRC_ROOT = str(Path(traceml_ai.__file__).resolve().parents[1])

_KEYS = (
    "traceml/primary/status",
    "traceml/step_time/step_time_ms",
    "traceml/step_time/input_wait_ms",
)


class _FakeHtml:
    def __init__(self, data, inject=True):
        self.data = data
        self.inject = inject


class _FakeSummary:
    def __init__(self, error=None):
        self.updates = []
        self._error = error

    def update(self, fields):
        if self._error is not None:
            raise self._error
        self.updates.append(dict(fields))


class _FakeRun:
    def __init__(self, *, summary_error=None, log_error=None):
        self.summary = _FakeSummary(summary_error)
        self.logged = []
        self._log_error = log_error

    def log(self, data, **kwargs):
        if self._log_error is not None:
            raise self._log_error
        self.logged.append((data, kwargs))


def _forbidden(*args, **kwargs):
    raise AssertionError("the helper must not own the W&B run lifecycle")


def _fake_wandb(run=None):
    fake = types.ModuleType("wandb")
    fake.Html = _FakeHtml
    fake.run = run
    fake.init = _forbidden
    fake.login = _forbidden
    fake.finish = _forbidden
    return fake


def _payload():
    return _make_payload(
        primary_diagnosis={
            "kind": "INPUT_BOUND",
            "status": "INPUT-BOUND",
            "severity": "warning",
            "summary": "Input wait dominates the step.",
            "action": "Add DataLoader workers.",
        }
    )


@pytest.fixture
def session(monkeypatch):
    """Primary rank with local history and a recorded summary request."""
    state = types.SimpleNamespace(payload=_payload(), calls=[])

    def fake_final_summary(**kwargs):
        state.calls.append(kwargs)
        return state.payload

    monkeypatch.setattr(helper, "is_primary_rank", lambda: True)
    monkeypatch.setattr(
        helper,
        "resolve_session_context_from_env",
        lambda: types.SimpleNamespace(history_enabled=True),
    )
    monkeypatch.setattr(
        helper.summary_client, "final_summary", fake_final_summary
    )
    return state


@pytest.fixture
def run(monkeypatch):
    active = _FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(active))
    return active


def _card(run):
    assert len(run.logged) == 1
    data, kwargs = run.logged[0]
    assert set(data) == {helper.WANDB_PANEL_KEY}
    assert kwargs == {"commit": True}
    return data[helper.WANDB_PANEL_KEY]


def test_module_import_needs_neither_wandb_nor_torch():
    # ``traceml_ai.sdk`` imports torch when it is installed, so torch is
    # blocked here (the tests/runtime/test_torch_free.py pattern): the
    # import must succeed without it and must not import wandb.
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import traceml_ai.integrations.wandb; "
        "print('wandb' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": _SRC_ROOT},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False"]


def test_writes_the_three_fields_and_one_card(session, run, capsys):
    assert helper.log_training_report() is True

    assert run.summary.updates == [
        {
            "traceml/primary/status": "INPUT-BOUND",
            "traceml/step_time/step_time_ms": 100.0,
            "traceml/step_time/input_wait_ms": 20.0,
        }
    ]
    card = _card(run)
    assert isinstance(card, _FakeHtml)
    assert card.inject is False
    assert card.data.startswith("<!DOCTYPE html>")
    assert "Input wait dominates the step." in card.data
    assert session.calls == [{}]
    assert capsys.readouterr().err == ""


def test_explicit_run_uses_the_imported_html_and_never_inits(
    session, monkeypatch
):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run=None))
    explicit = _FakeRun()

    assert helper.log_training_report(run=explicit) is True

    assert isinstance(_card(explicit), _FakeHtml)
    assert len(explicit.summary.updates) == 1


def test_no_active_run_warns_without_fetching(session, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run=None))

    assert helper.log_training_report() is False

    err = capsys.readouterr().err
    assert err.startswith("[TraceML] W&B: ")
    assert "wandb.init()" in err and "run=" in err
    assert len(err.strip().splitlines()) == 1
    assert session.calls == []


def test_missing_wandb_prints_the_install_hint(session, monkeypatch, capsys):
    # None in sys.modules makes ``import wandb`` raise ImportError.
    monkeypatch.setitem(sys.modules, "wandb", None)

    assert helper.log_training_report() is False

    assert capsys.readouterr().err == (
        "[TraceML] W&B: wandb is not installed; skipping the diagnosis "
        'report. Install it with pip install "traceml-ai[wandb]".\n'
    )
    assert session.calls == []


def test_non_primary_rank_is_silent_and_does_not_fetch(
    session, run, monkeypatch, capsys
):
    monkeypatch.setattr(helper, "is_primary_rank", lambda: False)

    assert helper.log_training_report() is False

    assert capsys.readouterr().err == ""
    assert session.calls == []
    assert run.logged == [] and run.summary.updates == []


def test_telemetry_mode_prints_one_line_and_does_not_fetch(
    session, run, monkeypatch, capsys
):
    monkeypatch.setattr(
        helper,
        "resolve_session_context_from_env",
        lambda: types.SimpleNamespace(history_enabled=False),
    )

    assert helper.log_training_report() is False

    assert capsys.readouterr().err == (
        "[TraceML] W&B: the diagnosis report needs local history; telemetry "
        "mode keeps none, use the wandb exporter for live export.\n"
    )
    assert session.calls == []
    assert run.logged == []


def _no_session_context():
    raise RuntimeError("TraceML session context is not available.")


def test_missing_session_without_run_warns_and_returns_false(
    session, monkeypatch, capsys
):
    monkeypatch.setattr(
        helper, "resolve_session_context_from_env", _no_session_context
    )
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run=None))

    assert helper.log_training_report() is False

    err = capsys.readouterr().err
    assert "wandb.init()" in err and "run=" in err
    assert len(err.strip().splitlines()) == 1
    assert session.calls == []


def test_missing_session_with_run_raises_from_final_summary(
    session, run, monkeypatch
):
    monkeypatch.setattr(
        helper, "resolve_session_context_from_env", _no_session_context
    )

    def no_session(**kwargs):
        session.calls.append(kwargs)
        raise RuntimeError("TraceML session context is not available.")

    monkeypatch.setattr(helper.summary_client, "final_summary", no_session)

    with pytest.raises(RuntimeError, match="session context"):
        helper.log_training_report()

    assert session.calls == [{}]
    assert run.logged == [] and run.summary.updates == []


def test_missing_final_summary_returns_false(session, run):
    session.payload = None

    assert helper.log_training_report() is False

    assert run.logged == [] and run.summary.updates == []


def test_final_summary_runtime_error_propagates(session, run, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Timed out waiting for TraceML final summary")

    monkeypatch.setattr(helper.summary_client, "final_summary", fail)

    with pytest.raises(RuntimeError, match="Timed out"):
        helper.log_training_report()

    assert run.logged == [] and run.summary.updates == []


def test_summary_keys_are_the_projection_keys(session, run):
    helper.log_training_report()

    projected = compact_summary(session.payload)
    (written,) = run.summary.updates
    assert set(written) == set(_KEYS)
    assert written == {key: projected[key] for key in _KEYS}


def test_nan_average_is_omitted_and_zero_is_kept(session, run):
    average = session.payload["step_time"]["global"]["average"]
    average["step_time_ms"] = math.nan
    average["input_wait_ms"] = 0.0

    assert helper.log_training_report() is True

    (written,) = run.summary.updates
    assert "traceml/step_time/step_time_ms" not in written
    assert written["traceml/step_time/input_wait_ms"] == 0.0


def test_summary_failure_still_logs_the_card(session, monkeypatch, capsys):
    failing = _FakeRun(summary_error=ValueError("summary is read-only"))
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(failing))

    assert helper.log_training_report() is False

    assert isinstance(_card(failing), _FakeHtml)
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [
        "[TraceML] W&B: could not write the run summary fields: "
        "ValueError: summary is read-only"
    ]


def test_log_failure_keeps_the_summary_fields(session, monkeypatch, capsys):
    failing = _FakeRun(log_error=ValueError("run is finished"))
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(failing))

    assert helper.log_training_report() is False

    assert len(failing.summary.updates) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [
        "[TraceML] W&B: could not log the diagnosis card: "
        "ValueError: run is finished"
    ]


def test_payload_without_primary_diagnosis_still_logs_the_card(session, run):
    session.payload = _make_payload()

    assert helper.log_training_report() is True

    assert "NO PRIMARY DIAGNOSIS" in _card(run).data
    (written,) = run.summary.updates
    assert "traceml/primary/status" not in written
    assert written["traceml/step_time/step_time_ms"] == 100.0


def test_explicit_payload_skips_the_session_checks(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run=None))
    monkeypatch.setattr(helper, "is_primary_rank", _forbidden)
    monkeypatch.setattr(helper, "resolve_session_context_from_env", _forbidden)
    monkeypatch.setattr(helper.summary_client, "final_summary", _forbidden)
    explicit = _FakeRun()

    assert (
        helper.log_training_report(explicit, final_summary=_payload()) is True
    )
    assert len(explicit.logged) == 1


def test_summary_without_fields_logs_the_card_only(session, run, capsys):
    average = session.payload["step_time"]["global"]["average"]
    average["step_time_ms"] = None
    average["input_wait_ms"] = None
    session.payload.pop("primary_diagnosis", None)

    assert helper.log_training_report() is True

    assert isinstance(_card(run), _FakeHtml)
    assert run.summary.updates == []
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "payload", [{}, {"schema_version": 2.0}], ids=["empty", "schema_2"]
)
def test_old_or_malformed_payload_never_raises(payload, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run=None))
    explicit = _FakeRun()

    result = helper.log_training_report(explicit, final_summary=payload)

    err = capsys.readouterr().err.strip().splitlines()
    if result:
        assert isinstance(_card(explicit), _FakeHtml)
        assert err == []
    else:
        assert len(err) == 1 and err[0].startswith("[TraceML] W&B: ")


def test_unrenderable_payload_returns_false_and_writes_nothing(
    monkeypatch, capsys
):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run=None))
    explicit = _FakeRun()
    payload = _payload()
    # An int beyond float range: the card renderer raises OverflowError.
    payload["step_time"]["global"]["average"]["step_time_ms"] = 10**400

    assert helper.log_training_report(explicit, final_summary=payload) is False

    assert explicit.logged == [] and explicit.summary.updates == []
    assert capsys.readouterr().err.strip().splitlines() == [
        "[TraceML] W&B: could not build the diagnosis report: "
        "OverflowError: int too large to convert to float"
    ]


def test_summary_fields_drop_an_int_beyond_float_range():
    payload = _payload()
    payload["step_time"]["global"]["average"]["input_wait_ms"] = 10**400

    fields = helper._summary_fields(payload)

    assert fields == {
        "traceml/primary/status": "INPUT-BOUND",
        "traceml/step_time/step_time_ms": 100.0,
    }
