"""One-shot, lazy launcher hook for the standard Hugging Face Trainer."""

from __future__ import annotations

import sys

from traceml_ai.runtime import _import_hook


def _activate(module) -> None:
    try:
        from traceml_ai.integrations.huggingface import _enable_auto_attach

        _enable_auto_attach(module.Trainer)
    except Exception as exc:
        # Instrumentation must not change whether the user's import succeeds.
        try:
            print(
                f"[TraceML] Hugging Face auto-instrumentation unavailable: {exc}",
                file=sys.stderr,
            )
        except Exception:
            pass


def install():
    """Observe Trainer loading without importing Transformers ourselves."""
    return _import_hook.install(("transformers.trainer",), _activate)


uninstall = _import_hook.uninstall
