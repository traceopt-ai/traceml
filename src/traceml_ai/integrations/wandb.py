# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Publish TraceML's end-of-run diagnosis into a Weights & Biases run.

``log_training_report()`` writes the compact diagnosis card and three
comparison fields into the W&B run the user already owns. It never creates,
configures or finishes that run, and it reads no credentials. The live
per-step export is a separate concern, owned by the ``wandb`` exporter.

The helper is best-effort: a missing ``wandb`` install, a missing run, a
failed summary request, a payload it cannot render or a failed write prints
one ``[TraceML]`` line to stderr and returns ``False``.
"""

from __future__ import annotations

import sys
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

# The workspace media key the card is logged under.
WANDB_PANEL_KEY = "traceml/diagnosis_card"

_INSTALL_HINT = 'pip install "traceml-ai[wandb]"'


def _warn(message: str) -> None:
    try:
        print(f"[TraceML] W&B: {message}", file=sys.stderr)
    except Exception:
        pass


def _log_wandb_error(message: str, exc: Exception) -> None:
    """Record a failed W&B write without interrupting the caller."""
    try:
        log_internal_exception(
            get_error_logger("WandbIntegration"),
            f"[TraceML] {message}",
            exc,
        )
    except Exception:
        pass
    detail = type(exc).__name__
    text = str(exc)
    if text:
        detail = f"{detail}: {text}"
    _warn(f"{message}: {detail}")


def _skip_failed_summary(exc: Exception) -> None:
    """Record a failed summary request and warn that the report is skipped."""
    try:
        log_internal_exception(
            get_error_logger("WandbIntegration"),
            "[TraceML] could not request the final summary",
            exc,
        )
    except Exception:
        pass
    try:
        text = " ".join(str(exc).split())
    except Exception:
        text = ""
    text = text or type(exc).__name__
    if not text.endswith("."):
        text = f"{text}."
    _warn(f"{text} Skipping the diagnosis report.")


def _history_disabled() -> bool:
    """Return whether this session keeps no local history.

    Uses the same session context ``final_summary()`` checks, so the helper
    can skip before requesting a summary that would only fail. A missing
    session context is not decided here: it falls through to run
    resolution, and the failed summary request reports it when a run
    exists.
    """
    try:
        context = resolve_session_context_from_env()
    except RuntimeError:
        return False
    return not context.history_enabled


def _resolve_run(run: Any) -> Any:
    """Return the caller's run, else the active ``wandb.run``, else None."""
    if run is not None:
        return run
    try:
        import wandb
    except ImportError:
        _warn(
            "wandb is not installed; skipping the diagnosis report. "
            f"Install it with {_INSTALL_HINT}."
        )
        return None
    active = getattr(wandb, "run", None)
    if active is None:
        _warn(
            "no active run; call wandb.init() before log_training_report(), "
            "or pass the run with run=."
        )
    return active


def _write_summary(run: Any, fields: Dict[str, Any]) -> bool:
    if not fields:
        return True
    try:
        run.summary.update(fields)
    except Exception as exc:
        _log_wandb_error("could not write the run summary fields", exc)
        return False
    return True


def _write_card(run: Any, html: str) -> bool:
    try:
        import wandb

        run.log(
            {WANDB_PANEL_KEY: wandb.Html(html, inject=False)},
            commit=True,
        )
    except Exception as exc:
        _log_wandb_error("could not log the diagnosis card", exc)
        return False
    return True


def log_training_report(
    run: Any = None,
    *,
    final_summary: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Publish the TraceML diagnosis card and comparison fields into a W&B run.

    Call it after training finishes, while the run is still open. It logs
    the compact diagnosis card under ``traceml/diagnosis_card`` and copies
    ``traceml/primary/status``, ``traceml/step_time/step_time_ms`` and
    ``traceml/step_time/input_wait_ms`` into the run summary.

    The helper never creates or finishes a run and reads no credentials.
    Problems it can recover from print one ``[TraceML]`` line to stderr and
    return ``False``; it does not raise for them. A report requested during
    training is a best-effort snapshot and can cover fewer steps than the
    run has reached.

    Parameters
    ----------
    run:
        The W&B run to write into. Defaults to the active ``wandb.run``.
        Pass it explicitly when a framework owns the run, for example
        ``trainer.logger.experiment`` under PyTorch Lightning.
    final_summary:
        A ``final_summary.json`` payload to publish. Defaults to
        ``traceml.final_summary()`` for the current session, requested on
        the primary rank only.

    Returns
    -------
    bool
        ``True`` when every write that had something to write succeeded.
        A summary with no diagnosis and no Step Time averages gets the
        card only and still returns ``True``. ``False`` on non-primary
        ranks, in telemetry mode, without a run, without ``wandb``, when
        the summary request fails, when the payload cannot be rendered, or
        when a write failed.

        The helper publishes on global rank 0 only, also for a payload
        passed with ``final_summary=``. The telemetry-mode check runs only
        when the helper fetches the summary itself. That request is made
        once a run has been resolved; it fails when no TraceML session
        context is available, the aggregator reports an error, or the
        request times out.
    """
    if not is_primary_rank():
        return False
    if final_summary is None:
        if _history_disabled():
            _warn(
                "the diagnosis report needs local history; telemetry mode "
                "keeps none, use the wandb exporter for live export."
            )
            return False

    target = _resolve_run(run)
    if target is None:
        return False

    payload = final_summary
    if payload is None:
        try:
            payload = summary_client.final_summary()
        except Exception as exc:
            _skip_failed_summary(exc)
            return False
        if payload is None:
            _warn(
                "the TraceML summary is unavailable. Skipping the diagnosis "
                "report."
            )
            return False

    try:
        html = render_training_report(payload)
        fields = tracker_comparison_fields(payload)
    except Exception as exc:
        _log_wandb_error("could not build the diagnosis report", exc)
        return False

    summary_ok = _write_summary(target, fields)
    card_ok = _write_card(target, html)
    return summary_ok and card_ok


__all__ = ["WANDB_PANEL_KEY", "log_training_report"]
