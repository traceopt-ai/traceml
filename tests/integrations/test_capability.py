"""Unit tests for the fail-loud capability assertion (warn, never raise)."""

import logging

from traceml_ai.integrations._capability import warn_if_missing_streams
from traceml_ai.sdk.initial import TraceMLInitConfig

HF_REQUIRES = {"dataloader_fetch", "forward", "backward", "h2d"}


def _cfg(mode: str, **patches) -> TraceMLInitConfig:
    return TraceMLInitConfig(
        mode=mode,
        patch_dataloader=patches.get("patch_dataloader", False),
        patch_forward=patches.get("patch_forward", False),
        patch_backward=patches.get("patch_backward", False),
        patch_h2d=patches.get("patch_h2d", False),
    )


def _messages(caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records)


def test_warns_when_not_initialized(monkeypatch, caplog):
    monkeypatch.setattr("traceml_ai.sdk.initial.get_init_config", lambda: None)
    with caplog.at_level(logging.WARNING):
        warn_if_missing_streams("HF", HF_REQUIRES)
    assert "dataloader_fetch" in _messages(caplog)


def test_warns_in_manual_mode_names_all_dark_streams(monkeypatch, caplog):
    monkeypatch.setattr(
        "traceml_ai.sdk.initial.get_init_config", lambda: _cfg("manual")
    )
    with caplog.at_level(logging.WARNING):
        warn_if_missing_streams("HF", HF_REQUIRES)
    msg = _messages(caplog)
    for stream in ("dataloader_fetch", "forward", "backward", "h2d"):
        assert stream in msg, f"{stream} not named in warning: {msg!r}"


def test_silent_in_auto_mode(monkeypatch, caplog):
    monkeypatch.setattr(
        "traceml_ai.sdk.initial.get_init_config",
        lambda: _cfg(
            "auto",
            patch_dataloader=True,
            patch_forward=True,
            patch_backward=True,
            patch_h2d=True,
        ),
    )
    with caplog.at_level(logging.WARNING):
        warn_if_missing_streams("HF", HF_REQUIRES)
    assert "will NOT be captured" not in _messages(caplog)


def test_disabled_env_is_silent(monkeypatch, caplog):
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    monkeypatch.setattr("traceml_ai.sdk.initial.get_init_config", lambda: None)
    with caplog.at_level(logging.WARNING):
        warn_if_missing_streams("HF", HF_REQUIRES)
    assert _messages(caplog) == ""


def test_never_raises_on_partial_config(monkeypatch, caplog):
    # selective mode with only dataloader on: forward/backward/h2d are dark.
    monkeypatch.setattr(
        "traceml_ai.sdk.initial.get_init_config",
        lambda: _cfg("selective", patch_dataloader=True),
    )
    with caplog.at_level(logging.WARNING):
        warn_if_missing_streams("HF", HF_REQUIRES)
    msg = _messages(caplog)
    assert "dataloader_fetch" not in msg  # captured
    assert "forward" in msg and "backward" in msg and "h2d" in msg  # dark
