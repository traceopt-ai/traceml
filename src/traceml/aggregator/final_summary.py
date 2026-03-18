import logging
from typing import Any, Optional

from traceml.aggregator.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml.aggregator.summaries.system import generate_system_summary_card

logger = logging.getLogger(__name__)


def generate_summary(
    db_path: str,
    *,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    End-of-run summary hook.

    Generates system and step-time summary cards and (optionally) exports them
    to Weights & Biases.

    Parameters
    ----------
    db_path:
        Path to the session SQLite database file.  Summary cards are written
        alongside it as ``<db_path>_summary_card.{json,txt}``.
    wandb_run:
        An active ``wandb.Run`` object.  When provided, the summary JSON is
        logged to W&B as metrics (``run.summary``) and uploaded as an
        Artifact.  Defaults to ``None`` (no W&B export).

        Use ``upload_traceml_summary()`` from
        ``traceml.integrations.wandb`` as the recommended way to trigger
        W&B export from inside your training script.

    Notes
    -----
    - Both summary card generation and the W&B export are best-effort:
      failures are logged as warnings and never propagate to the caller.
    - The W&B export requires ``wandb`` to be installed
      (``pip install 'traceml-ai[wandb]'``).  If it is not installed, a
      warning is emitted and the function returns normally.
    """
    generate_system_summary_card(db_path)
    generate_step_time_summary_card(db_path)

    # Optional W&B export
    # Only triggered when the caller explicitly passes a wandb_run object.
    # Use upload_traceml_summary() from traceml.integrations.wandb
    # to trigger W&B export from inside a training script.
    if wandb_run is not None:
        try:
            from traceml.integrations.wandb import (  # noqa: PLC0415
                log_traceml_summary_to_wandb,
            )

            summary_json_path = db_path + "_summary_card.json"
            log_traceml_summary_to_wandb(
                summary_json_path=summary_json_path,
                run=wandb_run,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[TraceML] W&B export failed (run unaffected): {exc}",
                exc_info=True,
            )
