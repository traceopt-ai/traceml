"""
TraceML — Weights & Biases (W&B) Integration
=============================================

Logs TraceML end-of-run summaries into an existing W&B run.

Two entry points
----------------
1. **`WandbSummaryExporter.export(summary, run)`** — low-level; accepts the
   already-parsed summary dict returned by ``generate_summary``.

2. **`log_traceml_summary_to_wandb(summary_json_path, run, ...)`** — high-level
   convenience; reads the JSON file written by TraceML and delegates to
   ``WandbSummaryExporter.export()``.

Design goals
------------
- **Optional**: ``wandb`` is only imported inside the function body.  If the
  package is not installed the call silently returns ``False`` and logs a
  warning; it never raises.
- **Stable metric schema**: all keys are prefixed with ``traceml/`` so they
  appear in their own namespace in W&B and remain consistent across runs for
  easy comparison.
- **Full artifact upload**: the raw JSON is uploaded as a W&B Artifact so the
  full detail (per-rank breakdown, etc.) is always preserved.
- **Graceful failure**: any exception inside ``export()`` is caught, logged as
  a warning, and returns ``False``.  Training / post-processing code should
  never crash because of W&B.

Usage
-----
::

    import wandb
    from traceml.integrations.wandb import log_traceml_summary_to_wandb

    wandb.init(project="my-project", name="my-run")

    # ... your training loop with trace_step(model) ...

    log_traceml_summary_to_wandb(
        summary_json_path="./logs/session_xyz_summary_card.json",
        run=wandb.run,
    )
    wandb.finish()

Or pass the run into ``generate_summary`` directly::

    from traceml.aggregator.final_summary import generate_summary

    generate_summary(db_path, wandb_run=wandb.run)

Metric schema (W&B summary key → JSON source)
----------------------------------------------
See ``docs/wandb.md`` for the full reference table.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stable metric keys (never rename these — breaks cross-run comparisons)
# ---------------------------------------------------------------------------


def _flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the structured TraceML summary dict into a stable set of W&B
    metric keys prefixed with ``traceml/``.

    Only scalar-valued, numeric (or bool) fields are emitted.  Nested objects
    like ``per_rank`` or the rendered ``card`` string are intentionally
    excluded from the flat metrics view; they are preserved in the artifact.

    Parameters
    ----------
    summary:
        The parsed JSON dict written by ``generate_summary``.

    Returns
    -------
    Dict[str, Any]
        Flat dict of ``{wandb_key: value}`` pairs.  Values are always numeric
        (``float`` / ``int`` / ``bool``), never ``None`` — keys with ``None``
        source are silently dropped.
    """
    out: Dict[str, Any] = {}

    def _put(key: str, value: Any) -> None:
        """Add key only when value is a usable scalar."""
        if value is None:
            return
        if isinstance(value, (int, float, bool)):
            out[key] = value

    # ── system summary ──────────────────────────────────────────────────────
    sys = summary.get("system", {}) or {}
    _put("traceml/system/duration_s", sys.get("duration_s"))
    _put("traceml/system/cpu_avg_percent", sys.get("cpu_avg_percent"))
    _put("traceml/system/cpu_peak_percent", sys.get("cpu_peak_percent"))
    _put("traceml/system/ram_avg_gb", sys.get("ram_avg_gb"))
    _put("traceml/system/ram_peak_gb", sys.get("ram_peak_gb"))
    _put("traceml/system/ram_total_gb", sys.get("ram_total_gb"))
    _put("traceml/system/gpu_available", sys.get("gpu_available"))
    _put("traceml/system/gpu_count", sys.get("gpu_count"))
    _put(
        "traceml/system/gpu_util_avg_percent", sys.get("gpu_util_avg_percent")
    )
    _put(
        "traceml/system/gpu_util_peak_percent",
        sys.get("gpu_util_peak_percent"),
    )
    _put("traceml/system/gpu_mem_avg_gb", sys.get("gpu_mem_avg_gb"))
    _put("traceml/system/gpu_mem_peak_gb", sys.get("gpu_mem_peak_gb"))
    _put("traceml/system/gpu_temp_avg_c", sys.get("gpu_temp_avg_c"))
    _put("traceml/system/gpu_temp_peak_c", sys.get("gpu_temp_peak_c"))
    _put("traceml/system/gpu_power_avg_w", sys.get("gpu_power_avg_w"))
    _put("traceml/system/gpu_power_peak_w", sys.get("gpu_power_peak_w"))

    # ── step-time summary ───────────────────────────────────────────────────
    st = summary.get("step_time", {}) or {}
    _put("traceml/step_time/training_steps", st.get("training_steps"))
    _put("traceml/step_time/ranks_seen", st.get("ranks_seen"))
    _put("traceml/step_time/worst_avg_step_ms", st.get("worst_avg_step_ms"))
    _put("traceml/step_time/median_avg_step_ms", st.get("median_avg_step_ms"))
    _put(
        "traceml/step_time/worst_vs_median_pct", st.get("worst_vs_median_pct")
    )

    median_ms = st.get("median_split_ms") or {}
    _put("traceml/step_time/median_dataloader_ms", median_ms.get("dataloader"))
    _put("traceml/step_time/median_forward_ms", median_ms.get("forward"))
    _put("traceml/step_time/median_backward_ms", median_ms.get("backward"))
    _put("traceml/step_time/median_optimizer_ms", median_ms.get("optimizer"))

    worst_ms = st.get("worst_split_ms") or {}
    _put("traceml/step_time/worst_dataloader_ms", worst_ms.get("dataloader"))
    _put("traceml/step_time/worst_forward_ms", worst_ms.get("forward"))
    _put("traceml/step_time/worst_backward_ms", worst_ms.get("backward"))
    _put("traceml/step_time/worst_optimizer_ms", worst_ms.get("optimizer"))

    return out


# ---------------------------------------------------------------------------
# Main exporter class
# ---------------------------------------------------------------------------


class WandbSummaryExporter:
    """
    Exports a TraceML end-of-run summary to Weights & Biases.

    All methods are **stateless** — this class exists purely as a namespace.
    Use the module-level ``log_traceml_summary_to_wandb`` function for the
    most ergonomic API.

    Failure contract
    ----------------
    Every public method catches all exceptions and returns ``False`` on error.
    The caller is never expected to wrap the call in a try/except.
    """

    @staticmethod
    def export(
        summary: Dict[str, Any],
        run: Any = None,
        *,
        artifact_name: str = "traceml_summary",
        artifact_type: str = "traceml",
        json_filename: str = "traceml_summary.json",
        log_as_charts: bool = False,
    ) -> bool:
        """
        Log flat summary metrics to the W&B run summary panel and upload the
        full JSON as a W&B Artifact.

        Parameters
        ----------
        summary:
            The parsed TraceML summary dict.  Must have at least a ``system``
            or ``step_time`` key (the dict written by ``generate_summary``).
        run:
            Active ``wandb.Run`` object.  If ``None``, the function attempts to
            use ``wandb.run`` (the global active run).  If neither is available,
            the function logs a warning and returns ``False``.
        artifact_name:
            Name for the W&B Artifact.
        artifact_type:
            W&B Artifact type tag (used for filtering in the UI).
        json_filename:
            Filename used inside the artifact.
        log_as_charts:
            If ``True``, also call ``wandb.log(flat_metrics, commit=True)`` so
            the TraceML metrics appear as panels in the W&B **Charts** tab
            (in addition to appearing in the **Overview → Summary** panel).

            Use this when you want per-run bars/columns or to compare values
            visually across runs in a W&B Report.  When ``False`` (default),
            only ``run.summary`` is updated, which is sufficient for the
            runs table and sweep comparisons.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on any failure (never raises).
        """
        try:
            import wandb  # noqa: PLC0415 (intentional lazy import)
        except ImportError:
            logger.warning(
                "[TraceML] W&B integration skipped: 'wandb' is not installed. "
                "Run `pip install wandb` or `pip install 'traceml-ai[wandb]'`."
            )
            return False

        try:
            active_run = run if run is not None else wandb.run
            if active_run is None:
                logger.warning(
                    "[TraceML] W&B integration skipped: no active wandb.Run. "
                    "Call wandb.init() before training and pass the run, or "
                    "set TRACEML_WANDB_AUTO=1 (see docs/wandb.md)."
                )
                return False

            flat = _flatten_summary(summary)

            # ── 1. Always update run.summary (Overview tab) ────────────────
            active_run.summary.update(flat)
            logger.info(
                f"[TraceML] Logged {len(flat)} metric(s) to W&B run summary."
            )

            # ── 2. Optionally also emit via wandb.log (Charts tab) ─────────
            # wandb.log() with commit=True adds a single point visible in the
            # Charts tab.  We use the active run's log method so it is scoped
            # correctly even in multi-run scripts.
            if log_as_charts:
                active_run.log(flat, commit=True)
                logger.info(
                    "[TraceML] Logged TraceML metrics via wandb.log() "
                    "(Charts tab)."
                )

            # ── 3. Upload full JSON as artifact ────────────────────────────
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description="TraceML end-of-run summary (full detail).",
                metadata={"source": "traceml"},
            )

            import tempfile  # noqa: PLC0415

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / json_filename
                tmp_path.write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                artifact.add_file(str(tmp_path), name=json_filename)
                active_run.log_artifact(artifact)

            logger.info(
                f"[TraceML] Uploaded full summary as W&B Artifact "
                f"'{artifact_name}' (type='{artifact_type}')."
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[TraceML] W&B export failed (training is unaffected): {exc}",
                exc_info=True,
            )
            return False


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------


def log_traceml_summary_to_wandb(
    summary_json_path: str,
    run: Any = None,
    *,
    artifact_name: str = "traceml_summary",
    artifact_type: str = "traceml",
    log_as_charts: bool = False,
) -> bool:
    """
    Read a TraceML ``*_summary_card.json`` file and log it to W&B.

    This is the recommended entry point for manual / post-run integration.
    For automatic export at the end of every TraceML run, pass
    ``wandb_run=wandb.run`` to ``generate_summary()`` instead.

    Parameters
    ----------
    summary_json_path:
        Path to the ``*_summary_card.json`` file written by TraceML.
    run:
        Active ``wandb.Run`` object.  Defaults to ``wandb.run`` (the global
        active run).
    artifact_name:
        Name for the W&B Artifact (default: ``"traceml_summary"``).
    artifact_type:
        W&B Artifact type (default: ``"traceml"``).
    log_as_charts:
        If ``True``, also call ``wandb.log()`` so metrics appear as panels in
        the W&B **Charts** tab in addition to the **Overview → Summary** panel.
        See ``WandbSummaryExporter.export`` for details.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any failure (never raises).

    Example
    -------
    ::

        import wandb
        from traceml.integrations.wandb import log_traceml_summary_to_wandb

        wandb.init(project="my-project")
        # ... training ...
        log_traceml_summary_to_wandb(
            "./logs/session_summary_card.json",
            log_as_charts=True,   # also show in Charts tab
        )
        wandb.finish()
    """
    try:
        path = Path(summary_json_path)
        if not path.is_file():
            logger.warning(
                f"[TraceML] W&B export skipped: summary file not found at "
                f"'{summary_json_path}'."
            )
            return False

        summary = json.loads(path.read_text(encoding="utf-8"))

        return WandbSummaryExporter.export(
            summary,
            run=run,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            log_as_charts=log_as_charts,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"[TraceML] W&B export failed (training is unaffected): {exc}",
            exc_info=True,
        )
        return False
