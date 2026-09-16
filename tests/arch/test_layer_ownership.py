# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Structural guards for the aggregator's read side.

Three rules, checked against the source tree rather than at runtime, so a
change that crosses a boundary fails in CI before a reviewer has to say it:

1. A renderer package reads only the telemetry tables it owns. A surface
   that needs facts from several domains is its own package, declared
   below as a composing package, never a query slipped into another
   domain's reader.
2. Dashboard sections (``nicegui_sections``) format a payload; they never
   open the database.
3. Every dashboard layout key has exactly one producer, and every section
   that subscribes to a key is fed by one.

Background: the review of #388 (a cross-domain query in the System domain's
reader, carried on the System payload and fanned out to another section).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

SRC = Path(__file__).resolve().parents[2] / "src" / "traceml_ai"
RENDERERS = SRC / "renderers"
DISPLAY = SRC / "aggregator" / "display_drivers"
SECTIONS = DISPLAY / "nicegui_sections"
WRITERS = SRC / "aggregator" / "sqlite_writers"

# Which telemetry tables each renderer package may read. Packages listed
# with several domains are composing packages: cross-domain by design, they
# own nothing and exist so no domain reader has to answer for another's
# data. A package missing here fails rule 1 until it is declared.
OWNERS: Dict[str, Set[str]] = {
    "system": {"system_samples", "system_gpu_samples"},
    "process": {"process_samples"},
    "stdout_stderr": {"stdout_stderr_samples"},
    "step_time": {"step_time_samples"},
    # Step memory reads two capacity flags outside its domain
    # (gpu_available, gpu_mem_total_bytes); kept as found at 0.3.7.
    "step_memory": {
        "step_memory_samples",
        "process_samples",
        "system_samples",
    },
    # Composing packages.
    "model_diagnostics": set(),
    "context": {"system_samples", "process_samples", "runtime_environment"},
}

_TABLE_REF = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([A-Za-z_]\w*)")


def _string_literals(path: Path) -> Iterable[str]:
    """Every string constant in a module, f-string parts included."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(
                    part.value, str
                ):
                    yield part.value


def _table_universe() -> Set[str]:
    tables: Set[str] = set()
    for path in WRITERS.glob("*.py"):
        for literal in _string_literals(path):
            tables.update(
                re.findall(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", literal)
            )
    assert tables, "no CREATE TABLE statements found under sqlite_writers"
    return tables


def _renderer_package(path: Path) -> str:
    rel = path.relative_to(RENDERERS)
    if len(rel.parts) > 1:
        return rel.parts[0]
    # Root-level modules: ``stdout_stderr_renderer.py`` -> ``stdout_stderr``.
    return re.sub(r"_renderer$", "", rel.stem)


def _tables_read(path: Path, universe: Set[str]) -> Set[str]:
    found: Set[str] = set()
    for literal in _string_literals(path):
        found.update(
            name for name in _TABLE_REF.findall(literal) if name in universe
        )
    return found


def test_renderer_packages_read_only_their_own_tables() -> None:
    universe = _table_universe()
    offenders: List[str] = []
    undeclared: Set[str] = set()
    for path in sorted(RENDERERS.rglob("*.py")):
        tables = _tables_read(path, universe)
        if not tables:
            continue
        package = _renderer_package(path)
        if package not in OWNERS:
            undeclared.add(package)
            continue
        foreign = tables - OWNERS[package]
        if foreign:
            offenders.append(
                f"{path.relative_to(SRC)} reads {sorted(foreign)}; "
                f"renderers/{package} owns {sorted(OWNERS[package])}"
            )
    assert not undeclared, (
        "renderer packages that read tables but are not declared in "
        f"OWNERS: {sorted(undeclared)}. Declare the tables they own, or "
        "declare them as a composing package."
    )
    assert not offenders, (
        "cross-domain reads; move the query to the owning package or to a "
        "composing package:\n  " + "\n  ".join(offenders)
    )


def test_sections_do_not_query_the_database() -> None:
    offenders: List[str] = []
    for path in sorted(SECTIONS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] == "sqlite3" for alias in node.names
            ):
                offenders.append(f"{path.name}: import sqlite3")
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "sqlite3"
            ):
                offenders.append(f"{path.name}: from sqlite3 import ...")
        for literal in _string_literals(path):
            if re.search(r"\bSELECT\b", literal):
                offenders.append(f"{path.name}: SQL literal {literal[:40]!r}")
    assert not offenders, (
        "sections format payloads; queries belong in a renderer package:\n  "
        + "\n  ".join(offenders)
    )


def _layout_keys() -> Set[str]:
    tree = ast.parse((DISPLAY / "layout.py").read_text(encoding="utf-8"))
    keys = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.endswith("_LAYOUT")
    }
    assert keys, "no *_LAYOUT constants found in layout.py"
    return keys


def _producers() -> Tuple[Dict[str, List[str]], Set[str]]:
    """Renderer classes per layout key, and keys the driver fills itself."""
    by_renderer: Dict[str, List[str]] = {}
    for path in sorted(RENDERERS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "layout_section_name"
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id.endswith("_LAYOUT")
                ):
                    by_renderer.setdefault(kw.value.id, []).append(
                        str(path.relative_to(SRC))
                    )
    driver_fed: Set[str] = set()
    tree = ast.parse((DISPLAY / "nicegui.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Name) and key.id.endswith("_LAYOUT"):
                    driver_fed.add(key.id)
    return by_renderer, driver_fed


def _consumers() -> Set[str]:
    consumers: Set[str] = set()
    for path in sorted(SECTIONS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "subscribe_layout"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                consumers.add(node.args[0].id)
    assert consumers, "no subscribe_layout(...) calls found in sections"
    return consumers


def test_every_layout_key_has_one_producer_and_consumers_are_fed() -> None:
    keys = _layout_keys()
    by_renderer, driver_fed = _producers()
    doubled = {k: v for k, v in by_renderer.items() if len(v) > 1}
    assert not doubled, f"one renderer per layout key; doubled: {doubled}"
    unknown = (set(by_renderer) | driver_fed | _consumers()) - keys
    assert not unknown, f"layout keys not declared in layout.py: {unknown}"
    unfed = _consumers() - set(by_renderer) - driver_fed
    assert not unfed, (
        "sections subscribe to layout keys nothing produces (add a renderer "
        f"with that layout_section_name): {sorted(unfed)}"
    )
