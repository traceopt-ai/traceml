# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Top-level summary view command implementation for TraceML."""

from __future__ import annotations

from pathlib import Path

from traceml_ai.reporting.summary_artifact import (
    extract_summary_text,
    load_summary_artifact,
)


def view_summary(
    summary_path: str | Path,
    *,
    print_to_stdout: bool = True,
    re_render: bool = False,
) -> str:
    """
    Print and return the terminal summary for a summary JSON artifact.

    By default this prints the stored card verbatim, so an artifact always
    reads back exactly as it was written.

    With ``re_render=True`` the card is rebuilt from the payload with the
    current renderer, which is how an artifact written before a card change
    can be read in the current layout. The stored text is never modified.
    The profile is inferred from the stored card's own title; artifacts
    written before watch had its own card therefore re-render as run cards,
    which is what they already stated.

    Either way this is read-only: it does not regenerate diagnostics or read
    telemetry databases. Rendering a derived HTML report is opt-in and
    handled separately by ``traceml view <summary.json> --html`` (see
    ``reporting.html.render_html_report_from_file``).
    """
    payload = load_summary_artifact(summary_path)
    text = extract_summary_text(payload, path=summary_path)

    if re_render:
        text = _re_rendered_card(payload, stored_text=text)

    if print_to_stdout:
        print(text)

    return text


def _re_rendered_card(payload: dict, *, stored_text: str) -> str:
    """Rebuild the card from the payload, falling back to the stored text.

    A payload old or partial enough to defeat the current renderer must not
    turn a read-only command into an error, so any failure degrades to the
    text the artifact already carries.
    """
    try:
        from traceml_ai.reporting.summary_card import (
            build_card_from_payload,
            card_profile_from_text,
            card_to_plain,
        )

        return card_to_plain(
            build_card_from_payload(
                payload,
                profile=card_profile_from_text(stored_text),
            )
        )
    except Exception:
        return stored_text


__all__ = ["view_summary"]
