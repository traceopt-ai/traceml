# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Publish TraceML's end-of-run diagnosis into an MLflow run.

``log_training_report()`` writes the compact diagnosis card and three
comparison values into the MLflow run the user already owns. It writes
through ``MlflowClient`` only, and never creates, resumes, ends or deletes
that run. It reads no credentials beyond what MLflow itself resolves.

The helper is best-effort: a missing ``mlflow`` install, a tracking store
that cannot be reached, a missing or deleted run or a failed write prints
one ``[TraceML]`` line to stderr and returns ``False``. Only the documented
``RuntimeError`` of ``traceml.final_summary()`` propagates once a run is
resolved, as it does when that API is called directly.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Mapping, Optional

from traceml_ai.loggers.error_log import (
    get_error_logger,
    log_internal_exception,
)
from traceml_ai.reporting.html.training_report import render_training_report
from traceml_ai.sdk import summary_client
from traceml_ai.sdk.protocol import (
    is_primary_rank,
    resolve_session_context_from_env,
)
from traceml_ai.sdk.summary_projection import tracker_comparison_fields

# The run-relative artifact path of the card. The MLflow UI renders a
# ``.html`` artifact inline.
MLFLOW_REPORT_ARTIFACT = "traceml/training-report.html"

_STATUS_KEY = "traceml/primary/status"
# The tag value length every MLflow store accepts.
_MAX_TAG_VALUE = 5000

_INSTALL_HINT = 'pip install "traceml-ai[mlflow]"'


def _warn(message: str) -> None:
    try:
        print(f"[TraceML] MLflow: {message}", file=sys.stderr)
    except Exception:
        pass


def _log_mlflow_error(message: str, exc: Exception) -> None:
    """Record a failed MLflow write without interrupting the caller."""
    try:
        log_internal_exception(
            get_error_logger("MlflowIntegration"),
            f"[TraceML] {message}",
            exc,
        )
    except Exception:
        pass
    _warn(f"{message}: {exc}")


def _history_disabled() -> bool:
    """Return whether this session keeps no local history.

    Uses the same session context ``final_summary()`` checks, so the helper
    can skip before requesting a summary that would only raise. A missing
    session context is not decided here: it falls through to run
    resolution, and ``final_summary()`` reports it when a run exists.
    """
    try:
        context = resolve_session_context_from_env()
    except RuntimeError:
        return False
    return not context.history_enabled


def _resolve_client(
    mlflow: Any, client: Any, tracking_uri: Optional[str]
) -> Any:
    """Return the caller's client, else a new one, else None."""
    if client is not None:
        return client
    try:
        # Resolves the tracking store now, so a bad URI fails here.
        return mlflow.MlflowClient(tracking_uri)
    except Exception as exc:
        _warn(f"could not open the MLflow tracking store: {exc}")
        return None


def _resolve_run_id(mlflow: Any, run_id: Optional[str]) -> Optional[str]:
    """Return the caller's run id, else the active or last active run's."""
    if run_id is not None:
        return run_id
    active = mlflow.active_run()
    if active is None:
        try:
            # Finds a run that ended in this thread, as HF Trainer's does.
            active = mlflow.last_active_run()
        except Exception as exc:
            _warn(f"could not look up the last active run: {exc}")
            return None
    if active is None:
        _warn(
            "no run to write into; pass run_id=, call log_training_report() "
            "inside mlflow.start_run(), or call it after a run that ended in "
            "this thread."
        )
        return None
    return active.info.run_id


def _run_is_writable(client: Any, run_id: str) -> bool:
    """Return whether the run exists and is not deleted."""
    try:
        stage = client.get_run(run_id).info.lifecycle_stage
    except Exception as exc:
        _warn(f"could not read run {run_id}: {exc}")
        return False
    # Artifact writes skip the lifecycle check, so refuse a deleted run here.
    if stage != "active":
        _warn(f"run {run_id} is deleted; restore it or pass another run_id=.")
        return False
    return True


def _write_batch(
    mlflow: Any, client: Any, run_id: str, fields: Dict[str, Any]
) -> bool:
    if not fields:
        return True
    try:
        entities = mlflow.entities
        now_ms = int(time.time() * 1000)
        metrics = [
            entities.Metric(key, float(value), now_ms, 0)
            for key, value in fields.items()
            if key != _STATUS_KEY
        ]
        tags = []
        if _STATUS_KEY in fields:
            status = str(fields[_STATUS_KEY])[:_MAX_TAG_VALUE]
            tags.append(entities.RunTag(_STATUS_KEY, status))
        # Explicit, so MLFLOW_ENABLE_ASYNC_LOGGING cannot defer the write.
        client.log_batch(run_id, metrics=metrics, tags=tags, synchronous=True)
    except Exception as exc:
        _log_mlflow_error("could not write the run metrics and tag", exc)
        return False
    return True


def _render_card(payload: Mapping[str, Any]) -> Optional[str]:
    try:
        return render_training_report(payload)
    except Exception as exc:
        _log_mlflow_error(
            f"could not render the diagnosis card ({type(exc).__name__})", exc
        )
        return None


def _write_card(client: Any, run_id: str, html: str) -> bool:
    try:
        client.log_text(run_id, html, MLFLOW_REPORT_ARTIFACT)
    except Exception as exc:
        _log_mlflow_error("could not log the diagnosis card artifact", exc)
        return False
    return True


def log_training_report(
    run_id: Optional[str] = None,
    *,
    client: Any = None,
    tracking_uri: Optional[str] = None,
    final_summary: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Publish the TraceML diagnosis card and comparison values into an MLflow
    run.

    Call it after training finishes. It logs the compact diagnosis card as
    the artifact ``traceml/training-report.html``, the metrics
    ``traceml/step_time/step_time_ms`` and
    ``traceml/step_time/input_wait_ms`` at step 0, and the tag
    ``traceml/primary/status``. A run that already finished, failed or was
    killed still accepts them.

    The helper writes through ``MlflowClient`` only. It never creates,
    resumes, ends or deletes a run. Problems it can recover from print one
    ``[TraceML]`` line to stderr and return ``False``; it does not raise for
    them.

    Parameters
    ----------
    run_id:
        The run to write into. Defaults to ``mlflow.active_run()``, then
        ``mlflow.last_active_run()``, which finds the run Hugging Face
        Trainer ends when training returns. Pass it explicitly when a
        framework creates the run through a client, as PyTorch Lightning's
        ``MLFlowLogger`` does.
    client:
        The ``MlflowClient`` to write with. Defaults to a new client for
        ``tracking_uri``.
    tracking_uri:
        The tracking URI for the new client. Defaults to MLflow's own
        resolution. Ignored when ``client`` is given.
    final_summary:
        A ``final_summary.json`` payload to publish. Defaults to
        ``traceml.final_summary()`` for the current session, requested on
        the primary rank only.

    Returns
    -------
    bool
        ``True`` when every write that had something to write succeeded.
        A summary with no comparison values (no diagnosis and no Step Time
        averages) gets the card only, and the result is still ``True``.
        ``False`` on non-primary ranks, in telemetry mode, without
        ``mlflow``, without a reachable store, without a writable run, when
        the card cannot be rendered, or when either write failed.

    Raises
    ------
    RuntimeError
        Propagated unchanged from ``traceml.final_summary()``, which is
        requested only once the run has been read: when the payload is
        omitted and no TraceML session context is available, the
        aggregator reports an error, or the request times out.
    """
    if final_summary is None:
        if not is_primary_rank():
            return False
        if _history_disabled():
            _warn(
                "the diagnosis report needs local history; telemetry mode "
                "keeps none."
            )
            return False

    try:
        import mlflow
    except ImportError:
        _warn(
            "mlflow is not installed; skipping the diagnosis report. "
            f"Install it with {_INSTALL_HINT}."
        )
        return False

    target = _resolve_client(mlflow, client, tracking_uri)
    if target is None:
        return False
    resolved_id = _resolve_run_id(mlflow, run_id)
    if resolved_id is None:
        return False
    if not _run_is_writable(target, resolved_id):
        return False

    payload = final_summary
    if payload is None:
        payload = summary_client.final_summary()
    if payload is None:
        return False

    html = _render_card(payload)
    if html is None:
        return False
    fields = tracker_comparison_fields(payload)

    batch_ok = _write_batch(mlflow, target, resolved_id, fields)
    card_ok = _write_card(target, resolved_id, html)
    return batch_ok and card_ok


__all__ = ["MLFLOW_REPORT_ARTIFACT", "log_training_report"]
