import importlib

import pytest


def test_new_import_path_is_primary():
    module = importlib.import_module("traceml_ai")

    assert module.__name__ == "traceml_ai"
    assert hasattr(module, "init")


@pytest.mark.parametrize("module_name", ["traceml", "traceml.launcher.cli"])
def test_unsupported_import_path_raises_migration_error(module_name):
    with pytest.raises(ImportError) as error:
        importlib.import_module(module_name)

    assert str(error.value) == (
        "The 'traceml' import path is not supported. "
        "Use 'import traceml_ai as traceml' instead."
    )
