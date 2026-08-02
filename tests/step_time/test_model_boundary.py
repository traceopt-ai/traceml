# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Architecture contracts for the presentation-independent Step Time model."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path
from typing import Iterator

import traceml_ai.step_time as step_time_package
from traceml_ai.diagnostics.step_time.context import build_step_time_context
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

    assert dependencies <= {
        "__future__",
        "dataclasses",
        "functools",
        "types",
        "typing",
    }


def test_typed_analysis_facts_are_exported_without_loading_analyzer() -> None:
    """Expose canonical facts while keeping the package root lightweight."""
    for name in (
        "StepTimeValues",
        "StepTimeStepFacts",
        "StepTimeRankFacts",
    ):
        assert getattr(step_time_package, name) is getattr(model, name)

    dependencies = set(_imports(_PACKAGE_ROOT / "__init__.py"))
    assert "traceml_ai.step_time.analysis" not in dependencies
    assert "traceml_ai.step_time.pipeline" not in dependencies


def test_window_utility_reexports_canonical_contracts() -> None:
    """Preserve the former window import path without duplicate classes."""
    assert legacy_window.StepTimeWindow is model.StepTimeWindow
    assert legacy_window.DiagnosisClock is model.DiagnosisClock
    assert legacy_window.DIAGNOSIS_CLOCK_KEY == model.DIAGNOSIS_CLOCK_KEY


def test_diagnosis_context_accepts_only_the_canonical_window() -> None:
    """Do not reintroduce the nested rank map into canonical diagnosis."""
    assert tuple(inspect.signature(build_step_time_context).parameters) == (
        "window",
        "thresholds",
        "training_strategy",
    )
    context_source = (
        _PROJECT_ROOT
        / "src"
        / "traceml_ai"
        / "diagnostics"
        / "step_time"
        / "context.py"
    ).read_text(encoding="utf-8")
    rules_source = (
        _PROJECT_ROOT
        / "src"
        / "traceml_ai"
        / "diagnostics"
        / "step_time"
        / "rules.py"
    ).read_text(encoding="utf-8")

    assert "per_rank_timing" not in context_source
    assert "per_rank_timing" not in rules_source


def test_repository_progress_has_one_stored_source() -> None:
    """Keep latest-step progress only in the source cursor."""
    field_names = {
        field.name for field in fields(model.StepTimeRepositorySnapshot)
    }

    assert "latest_step_observed" not in field_names
    snapshot = model.StepTimeRepositorySnapshot(
        cursor=model.StepTimeSourceCursor(latest_step=42)
    )
    assert snapshot.cursor.latest_step == 42
    assert not hasattr(snapshot, "latest_step_observed")


def test_metric_shape_has_no_repeated_window_wrappers() -> None:
    """Keep metric statistics flat and window metadata on the window."""
    metric_fields = {field.name for field in fields(model.StepTimeMetric)}

    assert "clock" not in metric_fields
    assert "coverage" not in metric_fields
    assert "summary" not in metric_fields
    assert not hasattr(model, "StepTimeSummary")
    assert not hasattr(step_time_package, "StepTimeSummary")


def test_removed_renderer_schema_does_not_return() -> None:
    """The historical renderer-owned domain shim was intentionally retired."""
    assert not (
        _PROJECT_ROOT
        / "src"
        / "traceml_ai"
        / "renderers"
        / "step_time"
        / "schema.py"
    ).exists()
