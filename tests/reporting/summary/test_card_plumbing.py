# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Plumbing tests for the terminal summary card.

Covers the pieces that live outside the card renderer itself: profile
threading, artifact/stdout parity, the TTY colour gate, and the best-effort
fallback that keeps a rendering failure from breaking shutdown.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

from tests.sqlite_fixtures import (
    insert_process_sample,
    insert_step_time_sample,
    insert_system_sample,
    summary_database,
)
from traceml_ai.reporting import final as reporting_final
from traceml_ai.reporting.final import generate_summary
from traceml_ai.reporting.terminal_card import card as terminal_card
from traceml_ai.reporting.terminal_card.card import card_profile_from_text
from traceml_ai.sdk import summary_client

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def test_terminal_card_facade_keeps_its_public_symbols() -> None:
    """The module split must not change the established card import surface."""
    assert all(hasattr(terminal_card, name) for name in terminal_card.__all__)
    rendered = terminal_card.card_to_plain(terminal_card.build_fallback_card())
    assert "TraceML Run Summary" in rendered


class _TTYBuffer(io.StringIO):
    """A stdout stand-in that claims to be an interactive terminal."""

    def isatty(self) -> bool:
        return True


@pytest.fixture()
def session(tmp_path: Path) -> Path:
    """Build a small session database with system, process, and steps."""
    db_path = tmp_path / "telemetry.db"
    with summary_database(db_path) as conn:
        insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util=None,
        )
        insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
        )
        for step in range(1, 6):
            insert_step_time_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                traced_step_time=10.0,
            )
    return db_path


def _run(session: Path, tmp_path: Path, **kwargs) -> dict:
    """Generate a summary into a session root under ``tmp_path``."""
    session_root = tmp_path / "logs" / "session_test"
    session_root.mkdir(parents=True, exist_ok=True)
    return generate_summary(
        str(session),
        session_root=str(session_root),
        print_to_stdout=False,
        **kwargs,
    )


def test_text_matches_txt_artifact(session: Path, tmp_path: Path) -> None:
    payload = _run(session, tmp_path)
    txt = (tmp_path / "logs" / "session_test" / "final_summary.txt").read_text(
        encoding="utf-8"
    )
    assert txt == payload["text"] + "\n"
    assert "\x1b[" not in payload["text"]


def test_run_profile_is_the_default(session: Path, tmp_path: Path) -> None:
    text = _run(session, tmp_path)["text"]
    assert "TraceML Run Summary" in text
    assert "TraceML Watch Summary" not in text


def test_watch_profile_renders_the_watch_card(
    session: Path, tmp_path: Path
) -> None:
    text = _run(session, tmp_path, profile="watch")["text"]
    assert "TraceML Watch Summary" in text
    assert "Step Time" not in text
    assert "Step Memory" not in text
    assert "steps analyzed" not in text
    assert "traceml run <your-script>.py" in text


def test_footer_names_the_session_relative_artifact(
    session: Path, tmp_path: Path
) -> None:
    text = _run(session, tmp_path)["text"]
    assert "logs/session_test/final_summary.json" in text
    assert "(--html-report)" in text


def test_footer_names_the_html_artifact_when_written(
    session: Path, tmp_path: Path
) -> None:
    text = _run(session, tmp_path, write_html=True)["text"]
    assert "logs/session_test/final_summary.html" in text
    assert "(--html-report)" not in text


def _print_summary(session: Path, tmp_path: Path, monkeypatch) -> str:
    """Run with printing enabled and capture what reached stdout."""
    buffer = _TTYBuffer()
    monkeypatch.setattr(sys, "stdout", buffer)
    session_root = tmp_path / "logs" / "session_test"
    session_root.mkdir(parents=True, exist_ok=True)
    generate_summary(
        str(session),
        session_root=str(session_root),
        print_to_stdout=True,
    )
    return buffer.getvalue()


def test_tty_output_is_colorized(
    session: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    printed = _print_summary(session, tmp_path, monkeypatch)
    assert "\x1b[" in printed
    payload_text = (
        tmp_path / "logs" / "session_test" / "final_summary.txt"
    ).read_text(encoding="utf-8")
    assert _ANSI_RE.sub("", printed) == payload_text


@pytest.mark.parametrize(
    ("env_key", "env_value"),
    [("NO_COLOR", "1"), ("TERM", "dumb")],
)
def test_color_is_disabled_by_environment(
    session: Path,
    tmp_path: Path,
    monkeypatch,
    env_key: str,
    env_value: str,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv(env_key, env_value)
    assert "\x1b[" not in _print_summary(session, tmp_path, monkeypatch)


def test_non_tty_output_is_plain(
    session: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    session_root = tmp_path / "logs" / "session_test"
    session_root.mkdir(parents=True, exist_ok=True)
    payload = generate_summary(
        str(session),
        session_root=str(session_root),
        print_to_stdout=True,
    )
    printed = capsys.readouterr().out
    assert "\x1b[" not in printed
    assert printed == payload["text"] + "\n"


def _sdk_print(session_root: Path, monkeypatch) -> str:
    """Print a stored summary through the SDK read path, capturing stdout."""
    buffer = _TTYBuffer()
    monkeypatch.setattr(sys, "stdout", buffer)
    summary_client._load_existing_final_summary(session_root, print_text=True)
    return buffer.getvalue()


def _stored(session_root: Path) -> str:
    return (session_root / "final_summary.txt").read_text(encoding="utf-8")


def test_sdk_print_path_is_colorized_on_a_tty(
    session: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    _run(session, tmp_path)
    session_root = tmp_path / "logs" / "session_test"

    printed = _sdk_print(session_root, monkeypatch)

    assert "\x1b[" in printed
    assert _ANSI_RE.sub("", printed) == _stored(session_root) + "\n"


def test_sdk_print_path_infers_the_watch_profile(
    session: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    _run(session, tmp_path, profile="watch")
    session_root = tmp_path / "logs" / "session_test"
    stored = _stored(session_root)
    assert card_profile_from_text(stored) == "watch"

    printed = _sdk_print(session_root, monkeypatch)

    # A run-profile rebuild would not reproduce the watch card, so colour
    # here also proves the profile was inferred correctly.
    assert "\x1b[" in printed
    assert "TraceML Watch Summary" in _ANSI_RE.sub("", printed)
    assert _ANSI_RE.sub("", printed) == stored + "\n"


def test_sdk_print_path_keeps_stored_text_when_rebuild_differs(
    session: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    _run(session, tmp_path)
    session_root = tmp_path / "logs" / "session_test"
    txt_path = session_root / "final_summary.txt"
    tampered = _stored(session_root).replace("Verdict:", "VERDICT!", 1)
    txt_path.write_text(tampered, encoding="utf-8")

    printed = _sdk_print(session_root, monkeypatch)

    assert "\x1b[" not in printed
    assert printed == tampered + "\n"


def test_sdk_print_path_is_plain_without_a_tty(
    session: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    _run(session, tmp_path)
    session_root = tmp_path / "logs" / "session_test"

    summary_client._load_existing_final_summary(session_root, print_text=True)

    printed = capsys.readouterr().out
    assert "\x1b[" not in printed
    assert printed == _stored(session_root) + "\n"


def test_card_failure_degrades_to_a_minimal_card(
    session: Path, tmp_path: Path, monkeypatch
) -> None:
    def _boom(**kwargs):
        raise RuntimeError("card exploded")

    monkeypatch.setattr(reporting_final, "build_card_from_payload", _boom)

    text = _run(session, tmp_path)["text"]

    lines = text.splitlines()
    assert lines[0].startswith("+---")
    assert "TraceML Run Summary" in text
    assert "Verdict:" in text
    assert "final_summary.json" in text
    for line in lines:
        assert len(line) == 156
