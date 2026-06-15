import importlib

import pytest


def test_canonical_instrumentation_namespace_imports():
    optimizer_hooks = importlib.import_module(
        "traceml_ai.instrumentation.hooks.optimizer_hooks"
    )
    ddp_comm_hook = importlib.import_module(
        "traceml_ai.instrumentation.hooks.ddp_comm_hook"
    )
    forward_patch = importlib.import_module(
        "traceml_ai.instrumentation.patches.forward_auto_timer_patch"
    )

    assert optimizer_hooks.ensure_optimizer_timing_installed is not None
    assert ddp_comm_hook.ensure_ddp_comm_hook_installed is not None
    assert ddp_comm_hook.install_ddp_comm_hook is not None
    assert forward_patch.patch_forward is not None


@pytest.mark.parametrize(
    "module_name",
    (
        "traceml_ai.hooks.optimizer_hooks",
        "traceml_ai.patches.forward_auto_timer_patch",
    ),
)
def test_old_hook_and_patch_namespaces_are_removed(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
