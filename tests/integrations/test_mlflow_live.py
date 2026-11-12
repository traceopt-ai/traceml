"""Publish the TraceML diagnosis into a real, SQLite-backed MLflow store.

Skipped unless the full ``mlflow`` package is installed; ``mlflow-skinny``
has no SQLite store. No tracking server is started.
"""

from __future__ import annotations

import os

import pytest

# Silence MLflow's agent banner, printed when mlflow is first imported, and
# keep its usage telemetry off the network.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
pytest.importorskip("sqlalchemy")
mlflow = pytest.importorskip("mlflow")

from tests.reporting.html.conftest import _make_payload  # noqa: E402
from traceml_ai.integrations import mlflow as helper  # noqa: E402

_ARTIFACT = "traceml/training-report.html"


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
def store(tmp_path):
    """A real client and a run that already finished, as after training."""
    client = mlflow.MlflowClient(f"sqlite:///{tmp_path}/mlflow.db")
    experiment_id = client.create_experiment(
        "traceml-live", artifact_location=(tmp_path / "art").as_uri()
    )
    run = client.create_run(experiment_id)
    client.set_terminated(run.info.run_id, "FINISHED")
    return client, run.info.run_id


def test_finished_run_receives_metrics_tag_and_card(store, tmp_path):
    client, run_id = store

    assert (
        helper.log_training_report(
            run_id, client=client, final_summary=_payload()
        )
        is True
    )

    data = client.get_run(run_id).data
    assert data.metrics == {
        "traceml/step_time/step_time_ms": 100.0,
        "traceml/step_time/input_wait_ms": 20.0,
    }
    assert data.tags["traceml/primary/status"] == "INPUT-BOUND"
    history = client.get_metric_history(
        run_id, "traceml/step_time/step_time_ms"
    )
    assert [m.step for m in history] == [0]

    local = client.download_artifacts(run_id, _ARTIFACT, str(tmp_path))
    with open(local, encoding="utf-8") as handle:
        html = handle.read()
    assert html.startswith("<!DOCTYPE html>")
    assert "Input wait dominates the step." in html


def test_async_logging_env_is_overridden(store, monkeypatch):
    client, run_id = store
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")

    assert (
        helper.log_training_report(
            run_id, client=client, final_summary=_payload()
        )
        is True
    )

    # Visible immediately, without flushing an async queue.
    data = client.get_run(run_id).data
    assert data.metrics["traceml/step_time/step_time_ms"] == 100.0
    assert data.tags["traceml/primary/status"] == "INPUT-BOUND"


def test_deleted_run_is_refused(store, capsys):
    client, run_id = store
    client.delete_run(run_id)

    assert (
        helper.log_training_report(
            run_id, client=client, final_summary=_payload()
        )
        is False
    )

    assert "is deleted" in capsys.readouterr().err
    assert client.list_artifacts(run_id) == []
    assert client.get_run(run_id).data.metrics == {}


def test_bogus_run_id_is_refused(store, capsys):
    client, _ = store

    assert (
        helper.log_training_report(
            "0" * 32, client=client, final_summary=_payload()
        )
        is False
    )

    err = capsys.readouterr().err
    assert err.startswith(f"[TraceML] MLflow: could not read run {'0' * 32}")


def test_non_primary_rank_writes_nothing(store, monkeypatch, capsys):
    client, run_id = store
    monkeypatch.setattr(helper, "is_primary_rank", lambda: False)

    assert helper.log_training_report(run_id, client=client) is False

    assert capsys.readouterr().err == ""
    assert client.list_artifacts(run_id) == []
    assert client.get_run(run_id).data.metrics == {}
