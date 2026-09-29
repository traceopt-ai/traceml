# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Public-schema golden for the Step Time section of the final summary.

Every scenario is written through the production projection writer and read
back by the real summary builder, with the analysis window the final report
resolves for it. The golden pins two things readers depend
on: the exact set of public keys, and, per scenario and per metric, which
values are present and which are null. The null pattern is per metric on
purpose: ``optimizer_step`` and ``h2d`` occur only on some steps and still
count as measured, while forward, backward and the other phases must be
present on every aligned step, so one rule for all metrics would be wrong
for some of them.

A rename, a dropped key or a changed null pattern fails this test with a
diff. Regenerate with ``UPDATE_COMMAND``; it refuses to rewrite a key or
null-pattern change unless ``SCHEMA_VERSION`` was bumped first. The guard
runs on regeneration only: deleting or hand-editing the golden bypasses it.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from tests.step_time.scenarios import (
    ALL_SCENARIOS,
    create_step_time_database,
)
from traceml_ai.reporting.analysis_window import resolve_analysis_window
from traceml_ai.reporting.final import SCHEMA_VERSION
from traceml_ai.reporting.sections.step_time import (
    STEP_TIME_METRIC_NAMES,
    StepTimeSummarySection,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "step_time_public_schema.json"
UPDATE_COMMAND = (
    "pytest tests/step_time/test_public_schema_golden.py --update-golden"
)

# Per-rank rows are keyed by rank id; the schema is the same for every rank.
_RANK = "<rank>"


def _key_paths(value: Any, prefix: str = "") -> set[str]:
    """Every key path in a payload, with rank ids and list items folded."""
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = _RANK if prefix.endswith("groups.rows") else str(key)
            path = f"{prefix}.{name}" if prefix else name
            paths.add(path)
            paths |= _key_paths(child, path)
    elif isinstance(value, list):
        for item in value:
            paths |= _key_paths(item, f"{prefix}[]")
    return paths


def _presence(value: Any) -> str:
    return "null" if value is None else "value"


def _metric_table(payload: Mapping[str, Any]) -> dict[str, str]:
    """For each metric: present or null in the global stats and each row."""
    stats = payload["global"]
    rows = payload["groups"]["rows"]
    table = {}
    for metric in STEP_TIME_METRIC_NAMES:
        cells = [
            f"average={_presence(stats['average'][metric])}",
            f"median={_presence(stats['median'][metric]['value'])}",
            f"worst={_presence(stats['worst'][metric]['value'])}",
        ]
        cells += [
            f"rank{rank}={_presence(rows[rank]['metrics'][metric])}"
            for rank in sorted(rows, key=int)
        ]
        table[metric] = " ".join(cells)
    return table


def build_snapshot(tmp_path: Path) -> dict[str, Any]:
    """Key set and per-metric null table across every scenario."""
    keys: set[str] = set()
    metrics = {}
    for scenario in ALL_SCENARIOS:
        db_path = tmp_path / f"{scenario.name}.db"
        create_step_time_database(db_path, scenario)
        window = resolve_analysis_window(str(db_path))
        section = StepTimeSummarySection(analysis_window=window)
        payload = section.build(str(db_path)).payload
        keys |= _key_paths(payload)
        metrics[scenario.name] = _metric_table(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "keys": sorted(keys),
        "metrics": metrics,
    }


def render(snapshot: Mapping[str, Any]) -> str:
    """Sorted keys, one per line, so a golden diff reads as a schema diff."""
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def schema_change_without_bump(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> list[str]:
    """Why ``new`` may not replace ``old`` at the same schema version.

    Changing the key set, or the null pattern of a scenario the golden
    already has, changes what readers of ``final_summary.json`` see, so it
    needs a ``SCHEMA_VERSION`` bump. A new scenario that brings no new key
    needs none.
    """
    if old.get("schema_version") != new.get("schema_version"):
        return []
    reasons = []
    old_keys, new_keys = set(old["keys"]), set(new["keys"])
    if old_keys != new_keys:
        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)
        reasons.append(f"keys added {added}, removed {removed}")
    for name, table in old["metrics"].items():
        if name in new["metrics"] and new["metrics"][name] != table:
            reasons.append(f"null pattern changed in scenario {name!r}")
    return reasons


def write_golden(path: Path, snapshot: Mapping[str, Any]) -> list[str]:
    """Write ``snapshot`` to ``path`` unless that needs a version bump.

    Returns the reasons it refused, empty when it wrote the file.
    """
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        reasons = schema_change_without_bump(old, snapshot)
        if reasons:
            return reasons
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(snapshot), encoding="utf-8")
    return []


def refusal_message(reasons: list[str]) -> str:
    return (
        f"Refusing to rewrite the golden at schema_version {SCHEMA_VERSION}: "
        + "; ".join(reasons)
        + ". Bump SCHEMA_VERSION in src/traceml_ai/reporting/final.py and "
        "add a CHANGELOG.md entry, then run it again."
    )


def drift_message(expected: str, current: str) -> str:
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile="golden",
            tofile="current",
        )
    )
    return (
        "The Step Time public schema changed:\n"
        f"{diff}\n"
        "If this is intended, bump SCHEMA_VERSION in "
        "src/traceml_ai/reporting/final.py, add a CHANGELOG.md entry, "
        f"and regenerate with: {UPDATE_COMMAND}"
    )


def test_public_schema_matches_golden(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    snapshot = build_snapshot(tmp_path)

    if request.config.getoption("--update-golden"):
        reasons = write_golden(GOLDEN_PATH, snapshot)
        if reasons:
            pytest.fail(refusal_message(reasons), pytrace=False)
        return

    assert GOLDEN_PATH.exists(), f"No golden yet. Run: {UPDATE_COMMAND}"
    expected = GOLDEN_PATH.read_text(encoding="utf-8")
    current = render(snapshot)
    if current != expected:
        pytest.fail(drift_message(expected, current), pytrace=False)


_OLD = {
    "schema_version": 1.8,
    "keys": ["global.average.forward_ms"],
    "metrics": {"a": {"forward_ms": "average=value"}},
}


@pytest.mark.parametrize(
    ("new", "refused"),
    [
        pytest.param(
            {**_OLD, "keys": ["global.average.fwd_ms"]},
            True,
            id="renamed_key",
        ),
        pytest.param(
            {**_OLD, "metrics": {"a": {"forward_ms": "average=null"}}},
            True,
            id="changed_null_pattern",
        ),
        pytest.param(
            {**_OLD, "schema_version": 1.9, "keys": ["global.average.fwd"]},
            False,
            id="bumped_version",
        ),
        pytest.param(
            {**_OLD, "metrics": {**_OLD["metrics"], "b": {"x": "y"}}},
            False,
            id="new_scenario",
        ),
    ],
)
def test_regeneration_needs_a_version_bump_for_a_schema_change(
    new: Mapping[str, Any], refused: bool
) -> None:
    assert bool(schema_change_without_bump(_OLD, new)) is refused


def test_regeneration_writes_and_refuses_as_documented(tmp_path: Path) -> None:
    golden = tmp_path / "golden.json"
    assert write_golden(golden, _OLD) == []
    assert json.loads(golden.read_text(encoding="utf-8")) == _OLD

    renamed = {**_OLD, "keys": ["global.average.fwd_ms"]}
    assert write_golden(golden, renamed)
    assert json.loads(golden.read_text(encoding="utf-8")) == _OLD

    bumped = {**renamed, "schema_version": 1.9}
    assert write_golden(golden, bumped) == []
    assert json.loads(golden.read_text(encoding="utf-8")) == bumped


def test_the_messages_say_what_to_do() -> None:
    drift = drift_message("a\n", "b\n")
    assert "-a" in drift and "+b" in drift
    assert UPDATE_COMMAND in drift
    assert "SCHEMA_VERSION" in drift and "CHANGELOG.md" in drift
    refusal = refusal_message(["keys added ['x'], removed []"])
    assert "SCHEMA_VERSION" in refusal and "CHANGELOG.md" in refusal
