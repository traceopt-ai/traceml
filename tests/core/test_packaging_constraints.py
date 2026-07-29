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

The manifest is read with a real TOML parser. An earlier revision matched
quoted strings with a regex, which dropped a dependency whose URL carried
a ``#`` fragment and then swallowed the following entry along with it. A
guard that silently omits a dependency is the failure it exists to catch,
so the parse is never approximated. ``tomllib`` covers 3.11 and newer and
``tomli`` covers 3.10; if neither is importable the guard fails rather
than skipping.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - environment error
        _toml = None  # type: ignore[assignment]

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Dependency name -> reason the upper bound is justified.
# Add an entry only for a concrete, current incompatibility.
ALLOWED_UPPER_BOUNDS: dict[str, str] = {}

_UPPER_BOUND_PATTERN = re.compile(r"(<=|<|==|~=)\s*[0-9]")


def _parse_dependencies(text: str) -> list[str]:
    """Return the [project] dependencies array from pyproject.toml text."""
    if _toml is None:
        raise AssertionError(
            "no TOML parser available. Install tomli on Python 3.10 "
            "(pip install \"tomli; python_version < '3.11'\") so this "
            "guard can read pyproject.toml."
        )

    data = _toml.loads(text)
    try:
        dependencies = data["project"]["dependencies"]
    except KeyError as exc:
        raise AssertionError(
            f"could not locate [project].dependencies: missing {exc}"
        ) from None

    if not isinstance(dependencies, list):
        raise AssertionError("[project].dependencies is not an array")

    return [str(item) for item in dependencies]


def _runtime_dependencies() -> list[str]:
    """Return the [project] dependencies list from pyproject.toml."""
    return _parse_dependencies(PYPROJECT.read_text(encoding="utf-8"))


def _requirement_name(requirement: str) -> str:
    """Return the bare distribution name from a requirement string."""
    name = re.split(r"[\[(;<>=!~ @]", requirement.strip(), maxsplit=1)[0]
    return name.strip().lower()


def _version_specifier(requirement: str) -> str:
    """Return the requirement text with any environment marker removed."""
    return requirement.split(";")[0]


def test_parser_handles_quote_styles_comments_and_url_fragments() -> None:
    """No dependency may disappear because of how it is written.

    A dependency the parser cannot see is one the guard cannot check, so
    the bound would ship unnoticed. The URL fragment is the case that
    broke an earlier regex version: the ``#`` ended the string early and
    the entry after it was lost too.
    """
    text = """
[project]
dependencies = [
    "double>=1.0",
    'literal<3',
    "fragment @ https://example.com/pkg-1.0.whl#sha256=deadbeef",
    "capped<2",
]
"""

    assert _parse_dependencies(text) == [
        "double>=1.0",
        "literal<3",
        "fragment @ https://example.com/pkg-1.0.whl#sha256=deadbeef",
        "capped<2",
    ]


def test_upper_bound_is_detected_in_every_quote_style() -> None:
    """The upper-bound check must fire regardless of how a dep is quoted."""
    text = """
[project]
dependencies = ["basic<2", 'literal<=3']
"""

    for requirement in _parse_dependencies(text):
        assert _UPPER_BOUND_PATTERN.search(requirement) is not None


def test_runtime_dependencies_are_parsed() -> None:
    """The parser must find dependencies, or the guard is inert."""
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


def test_nicegui_floor_covers_ui_context() -> None:
    """Regression guard for issue #270.

    The display driver calls ui.context, which nicegui added in 1.4.23.
    Without a floor, an unrelated constraint elsewhere in the user's
    environment (an old fastapi pin, for example) walks nicegui back to
    a version that imports fine and raises AttributeError at runtime.
    """
    for requirement in _runtime_dependencies():
        if _requirement_name(requirement) != "nicegui":
            continue

        specifier = _version_specifier(requirement).replace(" ", "")

        assert ">=1.4.23" in specifier or ">1.4.22" in specifier, (
            f"nicegui is declared as {requirement!r} without a floor "
            "covering ui.context (added in 1.4.23). Resolvers may select "
            "an older nicegui that breaks the dashboard at runtime."
        )
        return

    pytest.fail("nicegui not found among runtime dependencies")
