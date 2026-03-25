from traceml.aggregator.summaries.process import generate_process_summary_card
from traceml.aggregator.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml.aggregator.summaries.system import generate_system_summary_card


def generate_summary(db_path: str) -> None:
    """
    End-of-run summary hook.

    Parameters
    ----------
    db_path:
        Path to the session SQLite database file.

    Notes
    -----
    Implement later. Keep this function side-effecting (write summary.json, etc.)
    and make it safe to call at shutdown.
    """
    generate_system_summary_card(db_path)
    generate_process_summary_card(db_path)
    generate_step_time_summary_card(db_path)
    return
