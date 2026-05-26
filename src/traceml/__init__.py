"""
Deprecated compatibility package for the former ``traceml`` import path.

Use ``import traceml_ai as tml`` in new code. The ``traceml`` CLI command is
unchanged.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_NEW_PACKAGE_NAME = "traceml_ai"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, new_name: str):
        self._new_name = new_name

    def create_module(self, spec):
        module = importlib.import_module(self._new_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    _traceml_alias_finder = True

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("traceml."):
            return None

        new_name = f"{_NEW_PACKAGE_NAME}.{fullname[len('traceml.'):]}"
        new_spec = importlib.util.find_spec(new_name)
        if new_spec is None:
            return None

        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(new_name),
            origin=new_spec.origin,
            is_package=new_spec.submodule_search_locations is not None,
        )


def _install_alias_finder() -> None:
    for finder in sys.meta_path:
        if getattr(finder, "_traceml_alias_finder", False):
            return
    sys.meta_path.insert(0, _AliasFinder())


warnings.warn(
    "The 'traceml' Python import path is deprecated and will be removed in a "
    "future release. Use 'import traceml_ai as tml' instead.",
    FutureWarning,
    stacklevel=2,
)

_new_package = importlib.import_module(_NEW_PACKAGE_NAME)

_install_alias_finder()
sys.modules[__name__] = _new_package
