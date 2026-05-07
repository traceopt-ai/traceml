import importlib

import pytest


def test_canonical_instrumentation_namespace_imports():
    optimizer_hooks = importlib.import_module(
        "traceml.instrumentation.hooks.optimizer_hooks"
    )
    forward_patch = importlib.import_module(
        "traceml.instrumentation.patches.forward_auto_timer_patch"
    )
    all_reduce_patch = importlib.import_module(
        "traceml.instrumentation.patches.all_reduce_auto_timer_patch"
    )

    assert optimizer_hooks.ensure_optimizer_timing_installed is not None
    assert forward_patch.patch_forward is not None
    assert all_reduce_patch.patch_all_reduce is not None


@pytest.mark.parametrize(
    "module_name",
    (
        "traceml.hooks.optimizer_hooks",
        "traceml.patches.forward_auto_timer_patch",
    ),
)
def test_old_hook_and_patch_namespaces_are_removed(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
