from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from traceml_ai.samplers import system_manifest


@pytest.mark.parametrize(
    ("global_rank", "should_write"), [(0, True), (4, False)]
)
def test_only_global_rank_zero_writes_root_system_manifest(
    monkeypatch, tmp_path, global_rank: int, should_write: bool
) -> None:
    monkeypatch.setenv("TRACEML_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("TRACEML_SESSION_ID", "shared-run")
    monkeypatch.setenv("RANK", str(global_rank))
    build_manifest = Mock(return_value={"rank": global_rank})
    monkeypatch.setattr(
        system_manifest, "build_system_manifest", build_manifest
    )

    system_manifest.write_system_manifest_if_missing(
        cpu_logical_core_count=1,
        ram_total_memory=1.0,
        gpu_available=False,
        gpu_count=0,
        logger=Mock(),
    )

    manifest_path = tmp_path / "shared-run" / "system_manifest.json"
    assert manifest_path.exists() is should_write
    if should_write:
        build_manifest.assert_called_once()
        assert json.loads(manifest_path.read_text()) == {"rank": global_rank}
    else:
        build_manifest.assert_not_called()
