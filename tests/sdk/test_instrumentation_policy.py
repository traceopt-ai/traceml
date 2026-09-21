"""Tests for declared ownership of public manual wrappers."""

from __future__ import annotations

import pytest

from traceml_ai.sdk import instrumentation_policy as policy
from traceml_ai.sdk.initial import TraceMLInitConfig, TraceMLInitMode


def _config(
    mode: TraceMLInitMode,
    *,
    patch_dataloader: bool = False,
    patch_forward: bool = False,
    patch_backward: bool = False,
    patch_h2d: bool = False,
    disabled: bool = False,
) -> TraceMLInitConfig:
    return TraceMLInitConfig(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        patch_h2d=patch_h2d,
        disabled=disabled,
    )


@pytest.mark.parametrize(
    ("feature", "wrapper_name"),
    [
        ("dataloader_fetch", "wrap_dataloader_fetch"),
        ("forward", "wrap_forward"),
        ("backward", "wrap_backward"),
        ("optimizer", "wrap_optimizer"),
        ("h2d", "wrap_h2d"),
    ],
)
def test_manual_wrapper_requires_init(monkeypatch, feature, wrapper_name):
    monkeypatch.setattr(policy.initial, "get_init_config", lambda: None)

    with pytest.raises(RuntimeError, match=r"traceml\.init"):
        policy.require_manual_wrapper_allowed(feature, wrapper_name)


@pytest.mark.parametrize(
    "feature",
    ["dataloader_fetch", "forward", "backward", "optimizer", "h2d"],
)
def test_auto_owns_every_supported_pytorch_feature(monkeypatch, feature):
    config = _config(
        "auto",
        patch_dataloader=True,
        patch_forward=True,
        patch_backward=True,
        patch_h2d=True,
    )
    monkeypatch.setattr(policy.initial, "get_init_config", lambda: config)

    with pytest.raises(RuntimeError, match="already owns"):
        policy.require_manual_wrapper_allowed(feature, f"wrap_{feature}")


def test_auto_allows_source_outside_automatic_coverage(monkeypatch):
    config = _config("auto", patch_dataloader=True)
    monkeypatch.setattr(policy.initial, "get_init_config", lambda: config)

    result = policy.require_manual_wrapper_allowed(
        "dataloader_fetch",
        "wrap_dataloader_fetch",
        covered_by_automatic_instrumentation=False,
    )

    assert result is config


@pytest.mark.parametrize(
    "feature",
    ["dataloader_fetch", "forward", "backward", "optimizer", "h2d"],
)
def test_manual_mode_allows_every_wrapper(monkeypatch, feature):
    config = _config("manual")
    monkeypatch.setattr(policy.initial, "get_init_config", lambda: config)

    assert (
        policy.require_manual_wrapper_allowed(feature, f"wrap_{feature}")
        is config
    )


@pytest.mark.parametrize(
    ("feature", "owned"),
    [
        ("dataloader_fetch", True),
        ("forward", False),
        ("backward", True),
        ("optimizer", False),
        ("h2d", False),
    ],
)
def test_selective_mode_follows_patch_ownership(monkeypatch, feature, owned):
    config = _config(
        "selective",
        patch_dataloader=True,
        patch_backward=True,
    )
    monkeypatch.setattr(policy.initial, "get_init_config", lambda: config)

    if owned:
        with pytest.raises(RuntimeError, match="already owns"):
            policy.require_manual_wrapper_allowed(feature, f"wrap_{feature}")
    else:
        assert (
            policy.require_manual_wrapper_allowed(feature, f"wrap_{feature}")
            is config
        )


def test_disabled_config_never_blocks_training(monkeypatch):
    config = _config(
        "auto",
        patch_dataloader=True,
        patch_forward=True,
        patch_backward=True,
        patch_h2d=True,
        disabled=True,
    )
    monkeypatch.setattr(policy.initial, "get_init_config", lambda: config)

    assert (
        policy.require_manual_wrapper_allowed("forward", "wrap_forward")
        is config
    )
