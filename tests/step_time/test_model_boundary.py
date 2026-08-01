# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Architecture contracts for the presentation-independent Step Time model."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

from traceml_ai.renderers.step_time import schema as legacy_schema
from traceml_ai.step_time import model
from traceml_ai.utils import step_time_window as legacy_window

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _PROJECT_ROOT / "src" / "traceml_ai" / "step_time"
_PRESENTATION_PREFIXES = (
    "nicegui",
    "plotly",
    "rich",
    "traceml_ai.aggregator",
    "traceml_ai.renderers",
    "traceml_ai.reporting",
)


def _imports(path: Path) -> Iterator[str]:
    """Yield absolute import targets found in one Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    yield node.module
                continue

            package = ["traceml_ai", "step_time"]
            parent = package[: len(package) - node.level + 1]
            if node.module:
                parent.extend(node.module.split("."))
            yield ".".join(parent)


def test_step_time_package_does_not_import_presentation_layers() -> None:
    """Keep the central package below CLI, dashboard, and reporting code."""
    violations = [
        (str(path.relative_to(_PROJECT_ROOT)), dependency)
        for path in _PACKAGE_ROOT.glob("*.py")
        for dependency in _imports(path)
        if dependency.startswith(_PRESENTATION_PREFIXES)
    ]

    assert violations == []


def test_model_depends_only_on_the_standard_library() -> None:
    """Keep data contracts independently importable and inexpensive."""
    dependencies = set(_imports(_PACKAGE_ROOT / "model.py"))

    assert dependencies <= {"__future__", "dataclasses", "typing"}


def test_renderer_schema_reexports_canonical_type_objects() -> None:
    """Preserve historical renderer imports while consumers migrate."""
    names = (
        "StepCombinedTimeCoverage",
        "StepCombinedTimeMetric",
        "StepCombinedTimeResult",
        "StepCombinedTimeSeries",
        "StepCombinedTimeSummary",
    )

    for name in names:
        assert getattr(legacy_schema, name) is getattr(model, name)


def test_window_utility_reexports_canonical_contracts() -> None:
    """Preserve the former window import path without duplicate classes."""
    assert legacy_window.StepTimeWindow is model.StepTimeWindow
    assert legacy_window.DiagnosisClock is model.DiagnosisClock
    assert legacy_window.DIAGNOSIS_CLOCK_KEY == model.DIAGNOSIS_CLOCK_KEY
