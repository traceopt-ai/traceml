# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Context strip: identity, liveness, coverage, and profile gating."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections.context_section import (  # noqa: E402,E501
    RUN_SECTIONS,
    STEP_SECTIONS,
    abbreviate_path,
    format_coverage,
    format_liveness,
    live_threshold_s,
    resolve_run_identity,
    run_name_label,
    sections_for_profile,
    strategy_token,
)
from traceml_ai.aggregator.display_drivers.nicegui_sections.formatting import (  # noqa: E402,E501
    format_elapsed,
)
from traceml_ai.renderers.shared.freshness import FreshnessPolicy  # noqa: E402


def test_elapsed_formats() -> None:
    assert format_elapsed(47) == "47s"
    assert format_elapsed(146) == "2m 26s"
    assert format_elapsed(3 * 3600 + 12 * 60) == "3h 12m"
    assert format_elapsed(-5) == "0s"


def test_liveness_is_derived_from_data_age() -> None:
    assert format_liveness(None) == ("no data", "waiting for the first sample")
    # Fresh data is just "live": a 2 s sampler would only cycle 0/1/2.
    assert format_liveness(1.2) == ("live", "")
    assert format_liveness(-0.5) == ("live", "")
    assert format_liveness(4.9) == ("live", "")
    assert format_liveness(47.0) == ("stale", "47s since last data")


def test_coverage_line_reads_observed_over_configured() -> None:
    ctx = {
        "world_size": 4,
        "ranks_reporting": 3,
        "gpu_count": 4,
        "node_count": 1,
        "first_data_ts": 1000.0,
        "last_data_ts": 1146.0,
    }
    assert (
        format_coverage(ctx)
        == "ranks 3/4 reporting · 4 GPUs observed · 1 node · 2m 26s"
    )


def test_coverage_never_invents_a_missing_fact() -> None:
    # No process data at all: the funnel part is omitted rather than
    # printing world_size as if it were observed.
    assert format_coverage({"world_size": 4, "gpu_count": 1}) == (
        "1 GPU observed"
    )
    # Observed ranks but no configured world size: numerator only.
    assert format_coverage({"ranks_reporting": 2, "node_count": 2}) == (
        "ranks 2 reporting · 2 nodes"
    )
    assert format_coverage({}) == ""


def test_watch_profile_shows_no_step_sections() -> None:
    watch = sections_for_profile("watch")
    assert not set(watch) & set(STEP_SECTIONS)
    assert {"system", "process"} <= set(watch)
    assert sections_for_profile("run") == RUN_SECTIONS
    assert sections_for_profile("deep") == RUN_SECTIONS
    assert sections_for_profile("") == RUN_SECTIONS


def _manifest(root: Path, run_name: str, script: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": root.name,
                "run": {
                    "run_name": run_name,
                    "session_id": root.name,
                    "identity_source": "run_name",
                },
                "launch": {"script_path": script, "profile": "run"},
            }
        )
    )


def test_identity_from_manifest_uses_script_basename_only(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    _manifest(logs / "bert_ddp", "bert_ddp", "/home/someone/proj/train.py")
    settings = SimpleNamespace(
        profile="watch",
        session_id="bert_ddp",
        logs_dir=str(logs),
        db_path="",
    )
    identity = resolve_run_identity(settings)
    assert identity["profile"] == "watch"
    assert identity["run_name"] == "bert_ddp"
    assert identity["script_name"] == "train.py"
    assert identity["artifact_path"].endswith("bert_ddp/final_summary.json")
    # A full script path leaks directory layout on a shared screen.
    assert "/home/someone" not in " ".join(identity.values())
    # The strip never claims a watch session is training.
    assert "training" not in " ".join(identity.values())


def test_identity_walks_up_from_the_database_for_replays(
    tmp_path: Path,
) -> None:
    root = tmp_path / "M1"
    _manifest(root, "M1", "bert_single_gpu_compare.py")
    db = root / "aggregator" / "telemetry" / "telemetry.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    settings = SimpleNamespace(
        profile="run", session_id="", logs_dir="", db_path=str(db)
    )
    identity = resolve_run_identity(settings)
    assert identity["run_name"] == "M1"
    assert identity["script_name"] == "bert_single_gpu_compare.py"
    assert identity["artifact_path"].endswith("final_summary.json")


def test_identity_degrades_to_session_id_without_a_manifest(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        profile="run",
        session_id="session_123",
        logs_dir=str(tmp_path),
        db_path="",
    )
    identity = resolve_run_identity(settings)
    assert identity["run_name"] == "session_123"
    assert identity["script_name"] == ""
    assert identity["artifact_path"].endswith("session_123/final_summary.json")


def test_strategy_chip_hides_non_classifications() -> None:
    assert strategy_token("ddp") == "DDP"
    assert strategy_token("fsdp") == "FSDP"
    assert strategy_token("distributed_unknown") == ""
    assert strategy_token("single_process") == ""
    assert strategy_token("") == ""
    assert strategy_token(None) == ""


def test_artifact_path_does_not_climb_out_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "far" / "M1"
    _manifest(root, "M1", "train.py")
    db = root / "telemetry.db"
    db.write_bytes(b"")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    identity = resolve_run_identity(
        SimpleNamespace(
            profile="run", session_id="", logs_dir="", db_path=str(db)
        )
    )
    assert not identity["artifact_path"].startswith("..")
    assert identity["artifact_path"].endswith("M1/final_summary.json")
    monkeypatch.chdir(root)
    identity = resolve_run_identity(
        SimpleNamespace(
            profile="run", session_id="", logs_dir="", db_path=str(db)
        )
    )
    assert identity["artifact_path"] == "final_summary.json"


def test_generated_session_id_reads_no_run_name(tmp_path: Path) -> None:
    # Launcher-generated id with the manifest flag set.
    root = tmp_path / "logs" / "session_20260821_123224_1103e8"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": root.name,
                "run": {
                    "run_name": root.name,
                    "session_id": root.name,
                    "identity_source": "generated",
                },
                "launch": {"script_path": "/x/train.py"},
            }
        )
    )
    settings = SimpleNamespace(
        profile="run",
        session_id=root.name,
        logs_dir=str(tmp_path / "logs"),
        db_path="",
    )
    identity = resolve_run_identity(settings)
    assert identity["run_name_source"] == "generated"
    assert run_name_label(identity) == ("no run name", True)
    # The id is still on the artifact-path line, so nothing is hidden.
    assert root.name in identity["artifact_path"]
    # A chosen name is shown as is.
    assert run_name_label(
        {"run_name": "bert_ddp", "run_name_source": "run_name"}
    ) == (
        "bert_ddp",
        False,
    )


def test_generated_id_shape_is_recognised_without_the_flag() -> None:
    settings = SimpleNamespace(
        profile="run",
        session_id="session_20260821_123224_1103e8",
        logs_dir="",
        db_path="",
    )
    identity = resolve_run_identity(settings)
    assert identity["run_name_source"] == "generated"
    assert run_name_label(identity) == ("no run name", True)


def test_artifact_path_is_abbreviated_for_display() -> None:
    long = (
        "/private/tmp/very/deep/tree/logs/strip_demo_watch/final_summary.json"
    )
    assert (
        abbreviate_path(long) == "…/logs/strip_demo_watch/final_summary.json"
    )
    assert abbreviate_path("logs/M1/final_summary.json") == (
        "logs/M1/final_summary.json"
    )
    assert abbreviate_path("final_summary.json") == "final_summary.json"
    assert abbreviate_path("") == ""


def test_strip_threshold_is_the_shared_policy_at_configured_cadence() -> None:
    """The strip holds no threshold of its own, and stales on the same edge.

    Two claims. The number comes from ``FreshnessPolicy`` fed the
    CONFIGURED interval, which is the only cadence a run-wide indicator
    has. And the boundary is inclusive, matching
    ``FreshnessPolicy.state_of``, which stales only once the threshold is
    exceeded. The local ``max(5 s, 2.5 * interval)`` this replaced fails
    both.

    It does NOT claim the strip and the System card report one state for
    one age. The card feeds the same policy the cadence it observed for
    the single node it describes, so the two thresholds legitimately
    differ; the card's own liveness is covered in
    ``tests/renderers/test_system_dashboard_gpu_payload.py``.
    """
    assert live_threshold_s(None) == 6.0
    assert live_threshold_s(2.0) == 6.0  # default sampler: the 5 s floor
    assert live_threshold_s(10.0) == 30.0  # 3 ticks of a slow sampler

    for interval in (None, 2.0, 10.0):
        threshold = live_threshold_s(interval)
        assert (
            threshold == FreshnessPolicy.from_interval(interval).stale_after_s
        )
        # Exactly at the threshold is still live, as ``state_of`` has it.
        assert format_liveness(threshold, threshold) == ("live", "")

    assert format_liveness(12.0, live_threshold_s(10.0)) == ("live", "")
    assert format_liveness(31.0, live_threshold_s(10.0)) == (
        "stale",
        "31s since last data",
    )


def test_coverage_prefers_gpus_observed_across_nodes() -> None:
    ctx = {
        "world_size": 2,
        "ranks_reporting": 2,
        "gpu_count": 1,
        "gpus_observed": 2,
        "node_count": 2,
    }
    assert (
        format_coverage(ctx)
        == "ranks 2/2 reporting · 2 GPUs observed · 2 nodes"
    )
