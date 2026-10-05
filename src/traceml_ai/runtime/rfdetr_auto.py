"""Lazy launcher activation for RF-DETR's existing factory adapter."""

import sys

from traceml_ai.runtime import _import_hook


def _activate(module):
    try:
        from traceml_ai.integrations import rfdetr

        rfdetr._enable_auto_attach(module)
    except Exception as exc:
        try:
            print(
                f"[TraceML] RF-DETR automatic attachment unavailable: {exc}",
                file=sys.stderr,
            )
        except Exception:
            pass


def install():
    return _import_hook.install(
        ("rfdetr.training",), _activate, ready_attribute="build_trainer"
    )


uninstall = _import_hook.uninstall
