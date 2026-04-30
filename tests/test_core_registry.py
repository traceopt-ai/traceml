import pytest

from traceml.core import Registry


def test_registry_registers_and_fetches_items() -> None:
    registry: Registry[int] = Registry()

    returned = registry.register("system", 1)

    assert returned == 1
    assert registry.get("system") == 1
    assert "system" in registry
    assert len(registry) == 1


def test_registry_preserves_registration_order() -> None:
    registry = Registry[str](
        [
            ("system", "SystemSampler"),
            ("process", "ProcessSampler"),
        ]
    )
    registry.register("step_time", "StepTimeSampler")

    assert registry.keys() == ("system", "process", "step_time")
    assert registry.all() == (
        "SystemSampler",
        "ProcessSampler",
        "StepTimeSampler",
    )
    assert registry.items() == (
        ("system", "SystemSampler"),
        ("process", "ProcessSampler"),
        ("step_time", "StepTimeSampler"),
    )
    assert tuple(registry) == (
        "SystemSampler",
        "ProcessSampler",
        "StepTimeSampler",
    )


def test_registry_rejects_duplicate_keys() -> None:
    registry = Registry[int]()
    registry.register("system", 1)

    with pytest.raises(KeyError, match="already registered"):
        registry.register("system", 2)


def test_registry_rejects_missing_keys() -> None:
    registry = Registry[int]()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")


def test_registry_rejects_empty_keys() -> None:
    registry = Registry[int]()

    with pytest.raises(ValueError, match="cannot be empty"):
        registry.register("  ", 1)


def test_registry_returns_mapping_copy() -> None:
    registry = Registry[int]([("system", 1)])
    mapping = registry.as_mapping()

    assert mapping == {"system": 1}
    assert mapping is not registry.as_mapping()
