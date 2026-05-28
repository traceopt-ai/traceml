import importlib
import sys

import pytest


def _drop_legacy_modules() -> None:
    for name in list(sys.modules):
        if name == "traceml" or name.startswith("traceml."):
            sys.modules.pop(name, None)


def test_new_import_path_is_primary():
    module = importlib.import_module("traceml_ai")

    assert module.__name__ == "traceml_ai"
    assert hasattr(module, "init")


def test_legacy_import_path_warns_and_aliases_new_package():
    _drop_legacy_modules()

    with pytest.warns(FutureWarning, match="traceml_ai"):
        legacy = importlib.import_module("traceml")

    primary = importlib.import_module("traceml_ai")
    assert legacy is primary


def test_legacy_submodule_import_still_works():
    _drop_legacy_modules()

    with pytest.warns(FutureWarning, match="traceml_ai"):
        legacy_cli = importlib.import_module("traceml.launcher.cli")

    primary_cli = importlib.import_module("traceml_ai.launcher.cli")
    assert legacy_cli.build_parser is primary_cli.build_parser
