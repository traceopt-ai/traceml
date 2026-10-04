"""One-shot, lazy launcher hook for the standard Hugging Face Trainer."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys


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


class _TrainerLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:
        self.original.exec_module(module)
        _activate(module)


class _TrainerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "transformers.trainer":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _TrainerLoader(spec.loader)
        # The loader owns activation now. Other imports use their normal path.
        uninstall(self)
        return spec


def install():
    """Observe Trainer loading without importing Transformers ourselves."""
    module = sys.modules.get("transformers.trainer")
    if module is not None and hasattr(module, "Trainer"):
        _activate(module)
        return None
    finder = _TrainerFinder()
    sys.meta_path.insert(0, finder)
    return finder


def uninstall(finder) -> None:
    if finder is not None and finder in sys.meta_path:
        sys.meta_path.remove(finder)
