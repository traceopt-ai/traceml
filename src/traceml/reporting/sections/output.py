"""Output helpers for generated final-report sections."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.summaries.summary_io import (
    append_text,
    load_json_or_empty,
    write_json,
)


def persist_section_summary(
    db_path: str,
    *,
    section_name: str,
    text: str,
    payload: Dict[str, Any],
    replace_text: bool = False,
) -> None:
    """Persist one generated section beside the SQLite database."""
    text_path = db_path + "_summary_card.txt"
    json_path = db_path + "_summary_card.json"

    if replace_text:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        append_text(text_path, text)

    existing = load_json_or_empty(json_path)
    existing[section_name] = payload
    write_json(json_path, existing)


__all__ = ["persist_section_summary"]
