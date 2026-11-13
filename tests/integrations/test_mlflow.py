"""Tests for publishing the TraceML diagnosis into an MLflow run.

The real MLflow client is never used here: a stand-in ``mlflow`` module
records what the helper writes, and its fluent run-lifecycle and logging
functions raise so a test fails if the helper ever uses them.
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
from traceml_ai.integrations import mlflow as helper
from traceml_ai.sdk.summary_projection import tracker_comparison_fields

_SRC_ROOT = str(Path(traceml_ai.__file__).resolve().parents[1])

_STEP_KEY = "traceml/step_time/step_time_ms"
_WAIT_KEY = "traceml/step_time/input_wait_ms"
_STATUS_KEY = "traceml/primary/status"


class _Metric:
    def __init__(self, key, value, timestamp, step):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.step = step


class _RunTag:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class _MlflowException(Exception):
    pass


def _run(run_id="run-1", lifecycle_stage="active", status="RUNNING"):
    info = types.SimpleNamespace(
        run_id=run_id, lifecycle_stage=lifecycle_stage, status=status
    )
    return types.SimpleNamespace(info=info)


class _FakeClient:
    """Records writes; ``get_run`` returns ``run`` or raises ``error``."""

    instances: list = []

    def __init__(
        self,
        tracking_uri=None,
        *,
        run=None,
        get_error=None,
        batch_error=None,
        text_error=None,
    ):
        self.tracking_uri = tracking_uri
        self.run = run if run is not None else _run()
        self.get_error = get_error
        self.batch_error = batch_error
        self.text_error = text_error
        self.got = []
        self.batches = []
        self.texts = []
        _FakeClient.instances.append(self)

    def get_run(self, run_id):
        self.got.append(run_id)
        if self.get_error is not None:
            raise self.get_error
        return self.run

    def log_batch(self, run_id, metrics=(), params=(), tags=(), **kwargs):
        self.batches.append((run_id, list(metrics), list(tags), kwargs))
        if self.batch_error is not None:
            raise self.batch_error

    def log_text(self, run_id, text, artifact_file):
        self.texts.append((run_id, text, artifact_file))
        if self.text_error is not None:
            raise self.text_error


def _forbidden(*args, **kwargs):
    raise AssertionError("the helper must write through the client only")


def _fake_mlflow(active=None, last=None, client_cls=_FakeClient):
    fake = types.ModuleType("mlflow")
    fake.MlflowClient = client_cls
    fake.entities = types.SimpleNamespace(Metric=_Metric, RunTag=_RunTag)
    fake.exceptions = types.SimpleNamespace(MlflowException=_MlflowException)
    fake.active_run = lambda: active
    # A callable ``last`` stands in for last_active_run itself.
    fake.last_active_run = last if callable(last) else (lambda: last)
    for name in (
        "start_run",
        "end_run",
        "set_terminated",
        "delete_run",
        "log_metric",
        "log_metrics",
        "log_text",
        "log_artifact",
        "log_dict",
        "set_tag",
        "set_tags",
    ):
        setattr(fake, name, _forbidden)
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


@pytest.fixture(autouse=True)
def _reset_clients():
    _FakeClient.instances = []
    yield
    _FakeClient.instances = []


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
def mlflow(monkeypatch):
    fake = _fake_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def _only_batch(client):
    assert len(client.batches) == 1
    run_id, metrics, tags, kwargs = client.batches[0]
    assert kwargs == {"synchronous": True}
    return run_id, {m.key: m for m in metrics}, {t.key: t.value for t in tags}


def _only_text(client):
    assert len(client.texts) == 1
    return client.texts[0]


def test_module_import_needs_neither_mlflow_nor_torch():
    # ``traceml_ai.sdk`` imports torch when it is installed, so torch is
    # blocked here (the tests/runtime/test_torch_free.py pattern): the
    # import must succeed without it and must not import mlflow.
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import traceml_ai.integrations.mlflow; "
        "print('mlflow' in sys.modules)"
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


def test_explicit_run_and_client_write_one_batch_and_one_card(
    session, mlflow, capsys
):
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is True

    assert client.got == ["run-1"]
    run_id, metrics, tags = _only_batch(client)
    assert run_id == "run-1"
    assert set(metrics) == {_STEP_KEY, _WAIT_KEY}
    assert metrics[_STEP_KEY].value == 100.0
    assert metrics[_WAIT_KEY].value == 20.0
    for metric in metrics.values():
        assert isinstance(metric.value, float)
        assert metric.step == 0
        assert isinstance(metric.timestamp, int)
    assert tags == {_STATUS_KEY: "INPUT-BOUND"}

    run_id, html, path = _only_text(client)
    assert run_id == "run-1"
    assert path == helper.MLFLOW_REPORT_ARTIFACT
    assert path == "traceml/training-report.html"
    assert html.startswith("<!DOCTYPE html>")
    assert "Input wait dominates the step." in html
    assert session.calls == [{}]
    assert capsys.readouterr().err == ""
    # The caller's client is used; none is constructed.
    assert _FakeClient.instances == [client]


def test_client_is_built_from_the_tracking_uri(session, mlflow):
    assert (
        helper.log_training_report("run-1", tracking_uri="sqlite:///x.db")
        is True
    )

    (client,) = _FakeClient.instances
    assert client.tracking_uri == "sqlite:///x.db"
    assert len(client.batches) == 1 and len(client.texts) == 1


def test_store_that_cannot_be_built_warns(session, monkeypatch, capsys):
    def broken_client(tracking_uri=None):
        raise _MlflowException("file store is not supported")

    monkeypatch.setitem(
        sys.modules, "mlflow", _fake_mlflow(client_cls=broken_client)
    )

    assert helper.log_training_report("run-1") is False

    assert capsys.readouterr().err == (
        "[TraceML] MLflow: could not open the MLflow tracking store: "
        "file store is not supported\n"
    )
    assert session.calls == []


def test_active_run_is_used_when_no_id_is_given(session, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        _fake_mlflow(active=_run("active-1"), last=_forbidden),
    )

    assert helper.log_training_report() is True

    (client,) = _FakeClient.instances
    assert client.got == ["active-1"]
    assert _only_batch(client)[0] == "active-1"
    assert _only_text(client)[0] == "active-1"


def test_last_active_run_is_used_after_the_run_ended(session, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mlflow", _fake_mlflow(last=_run("ended-1"))
    )

    assert helper.log_training_report() is True

    (client,) = _FakeClient.instances
    assert client.got == ["ended-1"]
    assert _only_text(client)[0] == "ended-1"


def test_last_active_run_failure_warns(session, monkeypatch, capsys):
    def unreachable():
        raise _MlflowException("connection refused")

    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow(last=unreachable))

    assert helper.log_training_report() is False

    assert capsys.readouterr().err == (
        "[TraceML] MLflow: could not look up the last active run: "
        "connection refused\n"
    )
    assert session.calls == []


def test_no_run_warns_with_the_three_ways(session, mlflow, capsys):
    assert helper.log_training_report() is False

    err = capsys.readouterr().err
    assert err.startswith("[TraceML] MLflow: no run to write into")
    assert "run_id=" in err and "mlflow.start_run()" in err
    assert "ended" in err
    assert len(err.strip().splitlines()) == 1
    assert session.calls == []
    (client,) = _FakeClient.instances
    assert client.got == []


def test_deleted_run_is_refused(session, mlflow, capsys):
    client = _FakeClient(run=_run(lifecycle_stage="deleted"))

    assert helper.log_training_report("run-1", client=client) is False

    assert capsys.readouterr().err == (
        "[TraceML] MLflow: run run-1 is deleted; restore it or pass another "
        "run_id=.\n"
    )
    assert client.batches == [] and client.texts == []
    assert session.calls == []


def test_unreadable_run_is_refused(session, mlflow, capsys):
    client = _FakeClient(get_error=_MlflowException("Run 'bogus' not found"))

    assert helper.log_training_report("bogus", client=client) is False

    assert capsys.readouterr().err == (
        "[TraceML] MLflow: could not read run bogus: Run 'bogus' not found\n"
    )
    assert client.batches == [] and client.texts == []
    assert session.calls == []


def test_finished_run_is_still_written(session, mlflow):
    client = _FakeClient(run=_run(status="FINISHED"))

    assert helper.log_training_report("run-1", client=client) is True

    assert len(client.batches) == 1 and len(client.texts) == 1


def test_nan_average_is_omitted_and_zero_is_kept(session, mlflow):
    average = session.payload["step_time"]["global"]["average"]
    average["step_time_ms"] = math.nan
    average["input_wait_ms"] = 0.0
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is True

    _, metrics, _ = _only_batch(client)
    assert set(metrics) == {_WAIT_KEY}
    assert metrics[_WAIT_KEY].value == 0.0


def test_comparison_fields_skip_an_int_beyond_float_range():
    # math.isfinite raises OverflowError on such an int; the shared
    # projection both tracker helpers use must skip it instead.
    payload = _payload()
    payload["step_time"]["global"]["average"]["step_time_ms"] = 10**400

    fields = tracker_comparison_fields(payload)

    assert _STEP_KEY not in fields
    assert fields[_WAIT_KEY] == 20.0
    assert fields[_STATUS_KEY] == "INPUT-BOUND"


def test_card_render_failure_warns_and_writes_nothing(session, mlflow, capsys):
    # The card renderer cannot draw an int beyond float range. The render
    # runs before either write, so the helper writes nothing.
    average = session.payload["step_time"]["global"]["average"]
    average["step_time_ms"] = 10**400
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is False

    assert client.batches == [] and client.texts == []
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert err[0].startswith(
        "[TraceML] MLflow: could not render the diagnosis card "
        "(OverflowError): "
    )


def test_status_is_cut_to_the_tag_value_limit(session, mlflow):
    session.payload["primary_diagnosis"]["status"] = "X" * 6000
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is True

    _, _, tags = _only_batch(client)
    assert tags[_STATUS_KEY] == "X" * 5000


def test_batch_failure_still_logs_the_card(session, mlflow, capsys):
    client = _FakeClient(batch_error=_MlflowException("store is read-only"))

    assert helper.log_training_report("run-1", client=client) is False

    assert len(client.batches) == 1
    assert len(client.texts) == 1
    assert capsys.readouterr().err.strip().splitlines() == [
        "[TraceML] MLflow: could not write the run metrics and tag: "
        "store is read-only"
    ]


def test_card_failure_keeps_the_batch(session, mlflow, capsys):
    client = _FakeClient(text_error=OSError("artifact store is full"))

    assert helper.log_training_report("run-1", client=client) is False

    assert len(client.batches) == 1
    assert len(client.texts) == 1
    assert capsys.readouterr().err.strip().splitlines() == [
        "[TraceML] MLflow: could not log the diagnosis card artifact: "
        "artifact store is full"
    ]


def test_non_primary_rank_is_silent_and_does_not_fetch(
    session, mlflow, monkeypatch, capsys
):
    monkeypatch.setattr(helper, "is_primary_rank", lambda: False)
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is False

    assert capsys.readouterr().err == ""
    assert session.calls == []
    assert client.got == [] and client.batches == [] and client.texts == []


def test_telemetry_mode_prints_one_line_and_does_not_fetch(
    session, mlflow, monkeypatch, capsys
):
    monkeypatch.setattr(
        helper,
        "resolve_session_context_from_env",
        lambda: types.SimpleNamespace(history_enabled=False),
    )
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is False

    assert capsys.readouterr().err == (
        "[TraceML] MLflow: the diagnosis report needs local history; "
        "telemetry mode keeps none.\n"
    )
    assert session.calls == []
    assert client.got == []


def test_missing_mlflow_prints_the_install_hint(session, monkeypatch, capsys):
    # None in sys.modules makes ``import mlflow`` raise ImportError.
    monkeypatch.setitem(sys.modules, "mlflow", None)

    assert helper.log_training_report("run-1") is False

    assert capsys.readouterr().err == (
        "[TraceML] MLflow: mlflow is not installed; skipping the diagnosis "
        'report. Install it with pip install "traceml-ai[mlflow]".\n'
    )
    assert session.calls == []


def test_final_summary_runtime_error_propagates(session, mlflow, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Timed out waiting for TraceML final summary")

    monkeypatch.setattr(helper.summary_client, "final_summary", fail)
    client = _FakeClient()

    with pytest.raises(RuntimeError, match="Timed out"):
        helper.log_training_report("run-1", client=client)

    assert client.got == ["run-1"]
    assert client.batches == [] and client.texts == []


def test_missing_final_summary_returns_false(session, mlflow):
    session.payload = None
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is False

    assert client.batches == [] and client.texts == []


def test_payload_without_primary_diagnosis_logs_the_card_and_no_tag(
    session, mlflow
):
    session.payload = _make_payload()
    client = _FakeClient()

    assert helper.log_training_report("run-1", client=client) is True

    _, metrics, tags = _only_batch(client)
    assert tags == {}
    assert metrics[_STEP_KEY].value == 100.0
    _, html, _ = _only_text(client)
    assert "NO PRIMARY DIAGNOSIS" in html


def test_explicit_payload_skips_the_session_checks(mlflow, monkeypatch):
    monkeypatch.setattr(helper, "is_primary_rank", _forbidden)
    monkeypatch.setattr(helper, "resolve_session_context_from_env", _forbidden)
    monkeypatch.setattr(helper.summary_client, "final_summary", _forbidden)
    client = _FakeClient()

    assert (
        helper.log_training_report(
            "run-1", client=client, final_summary=_payload()
        )
        is True
    )
    assert len(client.texts) == 1
