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
payload it cannot render or a failed write prints one ``[TraceML]`` line to
stderr and returns ``False``.
Only the documented ``RuntimeError`` of ``traceml.final_summary()``
propagates once a run is resolved, as it does when that API is called
directly.
"""

from __future__ import annotations

import math
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
from traceml_ai.sdk.summary_projection import TRACKER_PREFIX, compact_summary

# The workspace media key the card is logged under.
WANDB_PANEL_KEY = "traceml/diagnosis_card"

# The compact-summary fields copied into the run summary, which W&B shows as
# runs-table columns. Names come from the projection, never retyped.
_SUMMARY_KEYS = (
    f"{TRACKER_PREFIX}/primary/status",
    f"{TRACKER_PREFIX}/step_time/step_time_ms",
    f"{TRACKER_PREFIX}/step_time/input_wait_ms",
)
_NUMERIC_KEYS = frozenset(_SUMMARY_KEYS[1:])

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


def _is_finite_number(value: Any) -> bool:
    """Return whether ``value`` is a finite int or float, never raising.

    An int beyond float range makes ``math.isfinite`` raise
    ``OverflowError``, so the conversion is guarded.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _summary_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the three comparison fields present in the compact summary."""
    compact = compact_summary(payload) or {}
    fields: Dict[str, Any] = {}
    for key in _SUMMARY_KEYS:
        if key not in compact:
            continue
        value = compact[key]
        if key in _NUMERIC_KEYS and not _is_finite_number(value):
            continue
        fields[key] = value
    return fields


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
    return ``False``; it does not raise for them.

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
        the payload cannot be rendered, or when a write failed.

        The primary-rank and telemetry-mode checks run only when the
        helper fetches the summary itself. A payload passed with
        ``final_summary=`` skips both checks and is written wherever the
        helper is called, so pass it from one rank only.

    Raises
    ------
    RuntimeError
        Propagated unchanged from ``traceml.final_summary()``, which is
        requested only once a run has been resolved: when the payload is
        omitted and no TraceML session context is available, the
        aggregator reports an error, or the request times out. Without a
        run, a missing session context gives the usual missing-run warning
        and ``False``.
    """
    if final_summary is None:
        if not is_primary_rank():
            return False
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
        payload = summary_client.final_summary()
    if payload is None:
        return False

    try:
        html = render_training_report(payload)
        fields = _summary_fields(payload)
    except Exception as exc:
        _log_wandb_error("could not build the diagnosis report", exc)
        return False

    summary_ok = _write_summary(target, fields)
    card_ok = _write_card(target, html)
    return summary_ok and card_ok


__all__ = ["WANDB_PANEL_KEY", "log_training_report"]
