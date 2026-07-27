"""Guard the runtime dependency contract in pyproject.toml.

An upper bound on a runtime dependency silently caps every downstream
environment. The rest of the suite cannot catch it: CI resolves to
whatever the bound allows and stays green forever, while users on a
newer ecosystem get a downgrade or an unsolvable resolution.

A ``numpy<2`` bound shipped this way. Installing traceml-ai downgraded
NumPy 2 in place, and resolvers that refuse to downgrade backsolved to a
two-year-old release instead (issue #263).

Upper bounds remain allowed, but they must be listed in
ALLOWED_UPPER_BOUNDS with a reason, so the decision is explicit and
reviewable rather than incidental.

This parses pyproject.toml directly rather than using tomllib, which is
absent on Python 3.10, the version CI runs for pull requests. A skipped
guard on the one leg that always runs would defeat the purpose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Dependency name -> reason the upper bound is justified.
# Add an entry only for a concrete, current incompatibility.
ALLOWED_UPPER_BOUNDS: dict[str, str] = {}

_UPPER_BOUND_PATTERN = re.compile(r"(<=|<|==|~=)\s*[0-9]")
_DEPENDENCIES_BLOCK = re.compile(
    r"^dependencies\s*=\s*\[(?P<body>.*?)\]", re.MULTILINE | re.DOTALL
)


def _runtime_dependencies() -> list[str]:
    """Return the [project] dependencies list from pyproject.toml."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = _DEPENDENCIES_BLOCK.search(text)
    if match is None:
        raise AssertionError(
            "could not locate the [project] dependencies array in "
            f"{PYPROJECT}"
        )
    return re.findall(r'"([^"]+)"', match.group("body"))


def _requirement_name(requirement: str) -> str:
    """Return the bare distribution name from a requirement string."""
    name = re.split(r"[\[(;<>=!~ ]", requirement.strip(), maxsplit=1)[0]
    return name.strip().lower()


def _version_specifier(requirement: str) -> str:
    """Return the requirement text with any environment marker removed."""
    return requirement.split(";")[0]


def test_runtime_dependencies_are_parsed() -> None:
    """The parser must actually find dependencies, or the guard is inert."""
    dependencies = _runtime_dependencies()

    assert dependencies, "no runtime dependencies parsed from pyproject.toml"
    assert any(
        _requirement_name(item) == "numpy" for item in dependencies
    ), "expected numpy among the runtime dependencies"


@pytest.mark.parametrize("requirement", _runtime_dependencies())
def test_runtime_dependency_has_no_undocumented_upper_bound(
    requirement: str,
) -> None:
    """Fail when a runtime dependency caps the version a user may install."""
    name = _requirement_name(requirement)
    specifier = _version_specifier(requirement)

    operators = {
        match.group(1) for match in _UPPER_BOUND_PATTERN.finditer(specifier)
    }
    if not operators:
        return

    if name in ALLOWED_UPPER_BOUNDS:
        assert ALLOWED_UPPER_BOUNDS[
            name
        ].strip(), f"{name} is allowlisted but records no reason"
        return

    pytest.fail(
        f"Runtime dependency {requirement!r} carries an upper bound "
        f"({', '.join(sorted(operators))}). An upper bound caps every "
        "downstream environment and cannot be caught by the rest of the "
        "suite. Remove it, or add an entry to ALLOWED_UPPER_BOUNDS in "
        f"{Path(__file__).name} recording the concrete incompatibility "
        "that justifies it."
    )


def test_numpy_is_not_capped_below_version_2() -> None:
    """Regression guard for issue #263."""
    for requirement in _runtime_dependencies():
        if _requirement_name(requirement) != "numpy":
            continue

        specifier = _version_specifier(requirement).replace(" ", "")

        assert "<2" not in specifier, (
            f"numpy is capped below 2 by {requirement!r}. This downgrades "
            "NumPy in place for every user on a modern scientific stack."
        )
