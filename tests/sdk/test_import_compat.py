import importlib
import sys

import pytest


def _drop_short_import_modules() -> None:
    for name in list(sys.modules):
        if name == "traceml" or name.startswith("traceml."):
            sys.modules.pop(name, None)


def test_new_import_path_is_primary():
    module = importlib.import_module("traceml_ai")

    assert module.__name__ == "traceml_ai"
    assert hasattr(module, "init")


def test_short_import_path_aliases_implementation_package():
    _drop_short_import_modules()

    with pytest.warns(FutureWarning, match="deprecated"):
        short = importlib.import_module("traceml")

    primary = importlib.import_module("traceml_ai")
    assert short is primary


def test_short_submodule_import_still_works():
    _drop_short_import_modules()

    with pytest.warns(FutureWarning, match="deprecated"):
        short_cli = importlib.import_module("traceml.launcher.cli")

    primary_cli = importlib.import_module("traceml_ai.launcher.cli")
    assert short_cli.build_parser is primary_cli.build_parser
